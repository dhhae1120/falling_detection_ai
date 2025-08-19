# app.py
# -*- coding: utf-8 -*-

"""
Realtime fall detection (4-stream ensemble: joint/bone/joint_motion/bone_motion)

Features
- Video/Webcam/RTSP OR precomputed 3D skeleton JSON playback
- 4-stream ensemble with per-stream weights
- Robust model forward signature (supports forward(x) and forward(x, epoch))
- FP16-unsafe custom ops handled (AMP disabled, FP32 enforced)
- Optional headless mode (--no_show)
- Optional CSV logging of confirmed fall events (--save_csv)

Run examples:
  # JSON playback (headless) and save confirmed events
  python app.py --json ../convert_botsort/converted_formatA.json --no_show --save_csv ./fall_events.csv

  # Webcam (index 0)
  python app.py --source 0
"""

# ---- SAFETY PATCH: avoid UnicodeDecodeError during traceback formatting (when torch formats stack) ----
import linecache as _lc
__orig_updatecache = _lc.updatecache
def __updatecache_safe(filename, module_globals=None):
    try:
        return __orig_updatecache(filename, module_globals)
    except UnicodeDecodeError:
        _lc.cache[filename] = (0, None, ['\n'], filename)
        return _lc.cache[filename]
_lc.updatecache = __updatecache_safe
# -------------------------------------------------------------------------------------------------------

import os
import time
import json
import cv2
import numpy as np
import torch
from collections import defaultdict, deque
from importlib import import_module
import argparse
from pathlib import Path

# =========================
# User config (must match training)
# =========================
T = 32          # window length
V = 17          # #joints
C = 3           # channels (x,y,score) or (x,y,z)
GRAPH_CLASS = 'graph.custom17.Graph'
CLASS_NAMES_PATH = './data/class_names.txt'
FALL_NAME = 'fall'

WEIGHTS = dict(
    joint        = './weight/rgbd_ShiftGCNplus_joint-11.pt',
    bone         = './weight/rgbd_ShiftGCNplus_bone-13.pt',
    joint_motion = './weight/rgbd_ShiftGCNplus_joint_motion-28.pt',
    bone_motion  = './weight/rgbd_ShiftGCNplus_bone_motion-31.pt',
)

ALPHAS = dict(joint=1, joint_motion=0, bone=0, bone_motion=0)

THRESH = 0.5    # probability threshold for fall
STRIDE = 4      # infer every N frames
TOPK   = 6       # prefilter candidate count
DEBOUNCE_N = 2   # consecutive confirmations
TIMEOUT_SEC = 2.0  # person timeout

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP = False  # IMPORTANT: disable AMP because custom CUDA shift op isn't implemented for FP16

# Optional TF32 speed-up on Ampere+
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =========================
# pre_normalization
# =========================
_pre_norm_import_err = None
try:
    # commonly used in Shift-GCN-plus repos
    from data_gen.preprocess import pre_normalization  # (N,C,T,V,M) -> normalized
except Exception as e1:
    try:
        from preprocess import pre_normalization
    except Exception as e2:
        _pre_norm_import_err = (e1, e2)
        pre_normalization = None

# =========================
# Bone edges (Graph.inward) with fallback
# =========================
def load_bone_pairs(v=V):
    try:
        mod_name, cls_name = GRAPH_CLASS.rsplit('.', 1)
        G = getattr(import_module(mod_name), cls_name)()
        pairs = getattr(G, 'inward', [])
        if not pairs:
            raise ValueError("Graph.inward is empty.")
        return [(int(a), int(b)) for (a, b) in pairs]
    except Exception:
        if v != 17:
            raise RuntimeError("Graph load failed and fallback edges only provided for V=17.")  # noqa
        return [
            (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
                    (5, 7), (5, 11), (6, 8), (6, 12), (7, 9), (8, 10), (11, 12), (11, 13),
                    (12, 14), (13, 15), (14, 16)
        ]

BONE_PAIRS = load_bone_pairs(V)

# =========================
# JSON stream loader
# =========================
class JsonSkeletonStream:
    """Produces frame-wise persons list from a JSON file."""
    def __init__(self, json_path, V_expected, C_expected):
        with open(json_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        self.frames = self._to_frames(obj, V_expected, C_expected)

    def __len__(self):
        return len(self.frames)

    def get_frame_persons(self, t):
        return self.frames[t]

    @staticmethod
    def _to_frames(obj, Vexp, Cexp):
        frames = []

        # Format A: list-of-frames -> [[{id,joints(V,C)}, ...], ...]
        if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], list):
            for fr in obj:
                persons = []
                for p in fr:
                    pid = int(p['id'])
                    J = np.asarray(p['joints'], dtype=np.float32)
                    if J.shape != (Vexp, Cexp):
                        raise ValueError(f"JSON joints shape mismatch: {J.shape} != {(Vexp,Cexp)}")
                    persons.append({'id': pid, 'joints': J})
                frames.append(persons)
            return frames

        # Format B: packed (C,T,V,M)
        if isinstance(obj, dict) and all(k in obj for k in ['C','T','V','M','data']):
            C_,T_,V_,M_ = int(obj['C']), int(obj['T']), int(obj['V']), int(obj['M'])
            if C_ != Cexp or V_ != Vexp:
                raise ValueError(f"Packed dims mismatch: (C,V)=({C_},{V_}) != ({Cexp},{Vexp})")
            data = np.asarray(obj['data'], dtype=np.float32)  # (C,T,V,M)
            if data.shape != (C_,T_,V_,M_):
                raise ValueError(f"Packed data shape mismatch: {data.shape} != {(C_,T_,V_,M_)}")
            ids = obj.get('ids', list(range(1, M_+1)))
            if len(ids) != M_:
                raise ValueError("ids length != M")
            for t in range(T_):
                persons = []
                for m in range(M_):
                    J = data[:, t, :, m].T  # (V,C)
                    persons.append({'id': int(ids[m]), 'joints': J})
                frames.append(persons)
            return frames

        # Format C: list of persons with (T,V,C)
        if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict) and 'joints' in obj[0]:
            sequences = []
            maxT = 0
            for p in obj:
                pid = int(p.get('id', len(sequences)+1))
                J = np.asarray(p['joints'], dtype=np.float32)  # (T,V,C)
                if J.ndim != 3 or J.shape[1] != Vexp or J.shape[2] != Cexp:
                    raise ValueError(f"(T,V,C) mismatch: got {J.shape}, expected (*,{Vexp},{Cexp})")
                sequences.append((pid, J))
                maxT = max(maxT, J.shape[0])
            for t in range(maxT):
                persons = []
                for pid, J in sequences:
                    JT = J[t] if t < J.shape[0] else J[-1]
                    persons.append({'id': pid, 'joints': JT})
                frames.append(persons)
            return frames

        raise ValueError("Unsupported JSON schema (use A/B/C).")

# =========================
# Utilities to make bone/motion
# =========================
def joints_to_bones(clip_joint):
    """
    clip_joint: (C,T,V,1)
    return: (C,T,V,1) with bone[:, :, child] = joint[:, :, child] - joint[:, :, parent]
    """
    Cc, Tt, Vv, Mm = clip_joint.shape
    bone = np.zeros_like(clip_joint, dtype=np.float32)
    for child, parent in BONE_PAIRS:
        if child < Vv and parent < Vv:
            bone[:, :, child, :] = clip_joint[:, :, child, :] - clip_joint[:, :, parent, :]
    return bone

def temporal_diff(clip):
    """ clip: (C,T,V,1) -> out[:, t] = clip[:, t+1] - clip[:, t] """
    out = np.zeros_like(clip, dtype=np.float32)
    out[:, :-1, :, :] = clip[:, 1:, :, :] - clip[:, :-1, :, :]
    return out

# =========================
# Model load
# =========================
from model.shiftgcn_plus import Model as ShiftGCNPlus  # num_class, num_point, num_person, graph, in_channels

def load_num_class_and_fall_idx():
    if os.path.isfile(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
            class_names = [ln.strip() for ln in f if ln.strip()]
        num_class = len(class_names)
        if FALL_NAME not in class_names:
            raise RuntimeError(f"'{FALL_NAME}' not in class_names: {class_names}")
        fall_idx = class_names.index(FALL_NAME)
        return num_class, fall_idx, class_names
    # fallback
    return 4, 0, ['fall', 'sit', 'stay', 'walk']

NUM_CLASS, FALL_IDX, CLASS_NAMES = load_num_class_and_fall_idx()

def load_stream_model(weight_path, in_channels=C, num_point=V, num_person=1):
    m = ShiftGCNPlus(
        num_class=NUM_CLASS,
        num_point=num_point,
        num_person=num_person,
        graph=GRAPH_CLASS,
        in_channels=in_channels
    ).to(DEVICE).eval()
    state = torch.load(weight_path, map_location=DEVICE)
    if isinstance(state, dict) and 'state_dict' in state:
        m.load_state_dict(state['state_dict'], strict=True)
    else:
        m.load_state_dict(state, strict=True)
    # enforce FP32 to avoid half-precision custom-op issues
    m = m.float()
    return m

# =========================
# Realtime ensemble
# =========================
class RealtimeFallDetectorEnsemble:
    def __init__(self, weights=WEIGHTS, alphas=ALPHAS):
        # ring buffers: pid -> deque of (V,C)
        self.buffers = defaultdict(lambda: deque(maxlen=T))

        s = sum(max(0.0, v) for v in alphas.values())
        self.alphas = {k: (max(0.0, v)/s if s > 0 else 0.25) for k, v in alphas.items()}

        self.models = {
            'joint':        load_stream_model(weights['joint']),
            'bone':         load_stream_model(weights['bone']),
            'joint_motion': load_stream_model(weights['joint_motion']),
            'bone_motion':  load_stream_model(weights['bone_motion']),
        }
        torch.set_grad_enabled(False)

    def add_frame(self, detections):
        """
        detections: List[{'id': int, 'joints': np.ndarray (V,C)}]
        """
        for det in detections:
            pid = int(det['id'])
            J = det['joints'].astype(np.float32)
            if J.shape != (V, C):
                raise ValueError(f"joints shape expected {(V,C)}, got {J.shape}")
            self.buffers[pid].append(J)

    def _pack_person_joint_clip(self, pid):
        """
        buffer -> (C,T,V,1) + pre_normalization (must match training)
        """
        if pre_normalization is None:
            raise RuntimeError(
                f"Failed to import pre_normalization. Errors: {_pre_norm_import_err}"
            )
        seq = np.stack(list(self.buffers[pid]), axis=0)  # (t,V,C)
        if seq.shape[0] < T:
            pad = np.repeat(seq[-1:], T - seq.shape[0], axis=0)
            seq = np.concatenate([seq, pad], axis=0)
        else:
            seq = seq[-T:]
        clip = np.transpose(seq, (2, 0, 1))   # (C,T,V)
        clip = clip[:, :, :, None]            # (C,T,V,1)
        batch = pre_normalization(clip[None, ...])  # (1,C,T,V,1)
        return batch[0]                        # (C,T,V,1)

    def _make_four_streams(self, clip_joint_norm):
        bone = joints_to_bones(clip_joint_norm)
        jmot = temporal_diff(clip_joint_norm)
        bmot = temporal_diff(bone)
        return dict(joint=clip_joint_norm, bone=bone, joint_motion=jmot, bone_motion=bmot)

    def infer_person_prob(self, pid):
        """
        Return ensembled probability vector for one pid. Shape: (NUM_CLASS,)
        """
        clip_joint = self._pack_person_joint_clip(pid)           # (C,T,V,1)
        stream_dict = self._make_four_streams(clip_joint)

        probs = []
        for key in ('joint', 'bone', 'joint_motion', 'bone_motion'):
            model = self.models[key]
            arr = stream_dict[key].astype(np.float32, copy=False)      # (C,T,V,1)
            x = torch.from_numpy(arr).unsqueeze(0).to(DEVICE).float()  # (1,C,T,V,1)
            try:
                out = model(x)
            except TypeError:
                out = model(x, 0)  # some models require epoch
            # normalize output forms
            if isinstance(out, (tuple, list)):
                logits = out[0]
            elif isinstance(out, dict):
                logits = out.get('logits') or out.get('output') or out.get('scores')
                if logits is None:
                    raise TypeError(f"Unsupported dict output from stream {key}: keys={list(out.keys())}")
            else:
                logits = out
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            pr = torch.softmax(logits, dim=1)[0].detach().float().cpu().numpy()
            probs.append(self.alphas.get(key, 0.25) * pr)

        return np.sum(probs, axis=0)

    def infer_candidates(self, candidate_pids, fall_idx=0, thresh=0.8):
        fallen = []
        min_len = max(4, T // 4)
        for pid in candidate_pids:
            if len(self.buffers[pid]) < min_len:
                continue
            p = self.infer_person_prob(pid)
            if float(p[fall_idx]) >= float(thresh):
                fallen.append(pid)
        return fallen

# =========================
# Prefilter (fall proxy)
# =========================
def select_candidates(buffers, K=TOPK):
    def fall_proxy(dq):
        k = min(16, len(dq), T)
        if k < 4: return 0.0
        arr = np.stack(list(dq)[-k:], axis=0)  # (k,V,C)
        x, y = arr[:, :, 0], arr[:, :, 1]
        H = y.max(1) - y.min(1)
        a = max(1, k // 3)

        Hf = np.median(H[:a]); Hb = np.min(H[-a:])
        height_collapse = max(0.0, (Hf - Hb) / max(Hf, 1e-6))

        cy = y.mean(1); dv = np.diff(cy)
        down_vel = float(np.max(np.clip(-dv, 0, None)) / max(np.median(H), 1e-6))

        W = x.max(1) - x.min(1)
        ratio_front = np.median(W[:a] / (H[:a] + 1e-6))
        ratio_back  = np.median(W[-a:] / (H[-a:] + 1e-6))
        tilt_inc = max(0.0, ratio_back - ratio_front)

        return 0.5*down_vel + 0.35*height_collapse + 0.15*tilt_inc

    scored = [(pid, fall_proxy(dq)) for pid, dq in buffers.items() if len(dq) > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in scored[:K]]

# =========================
# TODO: plug your pose tracker here
# =========================
def get_skeletons(frame):
    """
    Must return:
        [{'id': int_pid, 'joints': np.ndarray shape (V,C)}, ...]
    Joint order (V) must match training graph.
    """
    return []

# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Realtime 4-stream ensemble fall detection")
    p.add_argument("--source", type=str, default="0",
                   help="0=webcam, path=video file, rtsp://..., used when --json is not set")
    p.add_argument("--json", type=str, default=None,
                   help="3D skeleton JSON path (Format A/B/C). If set, video is ignored.")
    p.add_argument("--no_show", action="store_true",
                   help="Do not open OpenCV windows (headless).")
    p.add_argument("--save_csv", type=str, default=None,
                   help="CSV path to log confirmed fall events.")
    return p.parse_args()

# =========================
# Main
# =========================
def main():
    args = parse_args()
    print(f"[INFO] DEVICE={DEVICE}, USE_AMP={USE_AMP}, CLASSES={CLASS_NAMES}, FALL_IDX={FALL_IDX}")

    for k, pth in WEIGHTS.items():
        if not os.path.isfile(pth):
            raise FileNotFoundError(f"Missing weight: {k} -> {pth}")

    det = RealtimeFallDetectorEnsemble(weights=WEIGHTS, alphas=ALPHAS)

    # CSV logger
    log_f = None
    log = None
    if args.save_csv:
        import csv
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        log_f = open(args.save_csv, "w", newline="", encoding="utf-8-sig")
        log = csv.writer(log_f)
        log.writerow(["ts","frame","pid","fall_prob"])
    events_cnt = {}

    last_seen = {}
    stable = {}
    frame_idx = 0

    # ---------- JSON mode ----------
    if args.json is not None:
        stream = JsonSkeletonStream(args.json, V_expected=V, C_expected=C)
        print(f"[INFO] JSON frames: {len(stream)}")
        H, W = 720, 1280  # canvas for optional display
        for t in range(len(stream)):
            frame_idx += 1
            now = time.time()

            persons = stream.get_frame_persons(t)
            det.add_frame(persons)

            for p in persons:
                last_seen[int(p['id'])] = now

            do_infer = (frame_idx % STRIDE == 0)
            confirmed, candidates = [], []
            if do_infer:
                candidates = select_candidates(det.buffers, K=TOPK)
                fallen = det.infer_candidates(candidates, fall_idx=FALL_IDX, thresh=THRESH)
                for pid in candidates:
                    if pid in fallen:
                        stable[pid] = stable.get(pid, 0) + 1
                        if stable[pid] >= DEBOUNCE_N:
                            confirmed.append(pid)
                    else:
                        stable[pid] = 0

            # logging
            if confirmed:
                for pid in confirmed:
                    prob = float(det.infer_person_prob(pid)[FALL_IDX])
                    print(f"[DETECT] frame={frame_idx} pid={pid} p_fall={prob:.3f}")
                    events_cnt[pid] = events_cnt.get(pid, 0) + 1
                    if log:
                        log.writerow([now, frame_idx, pid, f"{prob:.4f}"])

            # cleanup
            for pid in list(det.buffers.keys()):
                if now - last_seen.get(pid, 0) > TIMEOUT_SEC:
                    det.buffers.pop(pid, None)
                    last_seen.pop(pid, None)
                    stable.pop(pid, None)

            # simple display
            if not args.no_show:
                frame = np.zeros((H, W, 3), dtype=np.uint8)
                y = 60
                for detp in persons:
                    pid = int(detp['id'])
                    color = (0, 0, 255) if pid in confirmed else (0, 255, 0)
                    text  = f"FALL {pid}" if pid in confirmed else f"ID {pid}"
                    cv2.putText(frame, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    y += 30
                cv2.imshow('Fall Detection (Ensemble) [JSON]', frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break

        if log_f: log_f.close()
        if not args.no_show:
            cv2.destroyAllWindows()
        print("[INFO] Exit (JSON)")
        if events_cnt:
            print("[SUMMARY] confirmed events per pid:", events_cnt)
        else:
            print("[SUMMARY] no confirmed fall events")
        return

    # ---------- Video/RTSP mode ----------
    cap = cv2.VideoCapture(int(args.source)) if args.source.isdigit() else cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        now = time.time()

        persons = get_skeletons(frame)
        det.add_frame(persons)

        for p in persons:
            last_seen[int(p['id'])] = now

        do_infer = (frame_idx % STRIDE == 0)
        confirmed, candidates = [], []
        if do_infer:
            candidates = select_candidates(det.buffers, K=TOPK)
            fallen = det.infer_candidates(candidates, fall_idx=FALL_IDX, thresh=THRESH)
            for pid in candidates:
                if pid in fallen:
                    stable[pid] = stable.get(pid, 0) + 1
                    if stable[pid] >= DEBOUNCE_N:
                        confirmed.append(pid)
                else:
                    stable[pid] = 0

        # logging
        if confirmed:
            for pid in confirmed:
                prob = float(det.infer_person_prob(pid)[FALL_IDX])
                print(f"[DETECT] frame={frame_idx} pid={pid} p_fall={prob:.3f}")
                events_cnt[pid] = events_cnt.get(pid, 0) + 1
                if log:
                    log.writerow([now, frame_idx, pid, f"{prob:.4f}"])

        # timeout cleanup
        for pid in list(det.buffers.keys()):
            if now - last_seen.get(pid, 0) > TIMEOUT_SEC:
                det.buffers.pop(pid, None)
                last_seen.pop(pid, None)
                stable.pop(pid, None)

        # draw
        if not args.no_show:
            for detp in persons:
                pid = int(detp['id'])
                joints = detp['joints']
                cx, cy = joints[:, 0].mean(), joints[:, 1].mean()
                color = (0, 0, 255) if pid in confirmed else (0, 255, 0)
                text  = f"FALL {pid}" if pid in confirmed else f"ID {pid}"
                cv2.putText(frame, text, (int(cx), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow('Fall Detection (Ensemble)', frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

    cap.release()
    if not args.no_show:
        cv2.destroyAllWindows()
    if log_f: log_f.close()
    print("[INFO] Exit")
    if events_cnt:
        print("[SUMMARY] confirmed events per pid:", events_cnt)
    else:
        print("[SUMMARY] no confirmed fall events")

if __name__ == '__main__':
    main()
