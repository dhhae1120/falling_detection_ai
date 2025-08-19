# -*- coding: utf-8 -*-
import json
import argparse

def to_format_A(botsort_obj, V_expected=17, C_expected=3, score_min=None):
    """
    botsort_obj: [
      {"frame_id": int, "instances": [
         {"track_id": int, "keypoints": [[x,y,z],...V], "keypoint_scores": [...V], ...},
         ...
      ]},
      ...
    ]
    return Format-A: list[ list[ {"id": int, "joints": list[list[float]]} ] ]
    """
    out_frames = []

    for f in botsort_obj:
        instances = f.get("instances", [])
        persons = []
        for inst in instances:
            pid = int(inst.get("track_id", inst.get("id", 0)))
            kps = inst.get("keypoints", None)
            if kps is None:
                continue

            # 간단한 유효성 검사
            if len(kps) != V_expected:
                raise ValueError(f"V mismatch: got {len(kps)}, expected {V_expected}")
            if any(len(j) != C_expected for j in kps):
                raise ValueError("C mismatch: expected {} per joint".format(C_expected))

            # (선택) 스코어 기반 필터링
            if score_min is not None and "keypoint_scores" in inst:
                scores = inst["keypoint_scores"]
                if len(scores) == V_expected:
                    # 모든 관절 평균 스코어가 너무 낮으면 스킵(옵션)
                    if sum(scores)/len(scores) < float(score_min):
                        continue

            persons.append({"id": pid, "joints": kps})
        out_frames.append(persons)

    return out_frames


def main():
    ap = argparse.ArgumentParser(description="Convert BoT-SORT pose JSON -> Format-A (frames list)")
    ap.add_argument("--in",  dest="in_path", default = "3113249_botsort.json", help="BoT-SORT JSON path")
    ap.add_argument("--out", dest="out_path", default = "result.json", help="Output JSON path (Format-A)")
    ap.add_argument("--V", type=int, default=17, help="# joints (default: 17)")
    ap.add_argument("--C", type=int, default=3,  help="# channels per joint (default: 3, e.g., x,y,z)")
    ap.add_argument("--score-min", type=float, default=None,
                    help="(optional) drop person if mean(keypoint_scores) < score-min")
    args = ap.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        botsort = json.load(f)

    out_frames = to_format_A(botsort, V_expected=args.V, C_expected=args.C, score_min=args.score_min)

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(out_frames, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {args.out_path}  (frames={len(out_frames)})")


if __name__ == "__main__":
    main()
