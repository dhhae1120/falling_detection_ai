

from typing import Optional, Tuple
import numpy as np
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def _norm(v: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.linalg.norm(v) + eps)

def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v) + eps)
    return (v / n).astype(np.float32, copy=False)

def angle_between(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> float:
    uu = _unit(u, eps)
    vv = _unit(v, eps)
    d = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
    return float(np.arccos(d))

def rotation_matrix(axis: np.ndarray, angle: float, eps: float = 1e-8) -> np.ndarray:
    a = _unit(axis, eps).astype(np.float32)
    x, y, z = a
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    C = 1.0 - c
    R = np.array([
        [c + x*x*C,     x*y*C - z*s,  x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,    y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s,  c + z*z*C]
    ], dtype=np.float32)
    return R

def pre_normalization(
    data: np.ndarray,                      # (N,C,T,V,M)
    *,
    
    third_channel: str = "z",          # "score" (2D+score) | "z" (3D)
    data_is_centered: bool = False,        
   
    pad_null_frames: bool = True,
    align_axes_mode: str = "clip",        
    do_center: bool = True,                
    center_xy_only: bool = False,           
    scale_by: Optional[str] = "shoulder",        

    L_HIP: int = 11, R_HIP: int = 12,
    L_SH: int = 5,  R_SH: int = 6,

    target_z: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    target_x: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    eps: float = 1e-8,
    verbose: bool = True
) -> np.ndarray:

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be numpy.ndarray (N,C,T,V,M)")
    if data.ndim != 5:
        raise ValueError(f"data.ndim must be 5, got {data.ndim}")
    N, C, T, V, M = data.shape
    if C < 2:
        raise ValueError(f"C must be >=2 (x,y[,z/score]); got C={C}")

    is_3d = (third_channel.lower() == "z") and (C >= 3)


    if not is_3d:
        if align_axes_mode != "none" and verbose:
            print("[WARN] align_axes_mode is ignored for 2D+score.")
        align_axes_mode = "none"
        if scale_by and verbose:
            print("[WARN] scale_by is ignored for 2D+score.")
        scale_by = None


    if data_is_centered:
        do_center = False


    s = np.transpose(data, (0, 4, 2, 3, 1)).astype(np.float32, copy=True)

    # 1) Null-frame padding
    if verbose:
        print('pad null frames' if pad_null_frames else 'skip null-frame padding')
    if pad_null_frames:
        for i_n in range(N):
            for i_p in range(M):
                last_valid = None
                for t in range(T):
                    fr = s[i_n, i_p, t]  # (V,C)
                    if not np.isfinite(fr).all() or fr.sum() == 0.0:
                        if last_valid is not None:
                            s[i_n, i_p, t] = last_valid
                    else:
                        last_valid = fr


    if align_axes_mode in ("clip", "frame") and is_3d:
        tgt_z = np.array(target_z, dtype=np.float32)
        tgt_x = np.array(target_x, dtype=np.float32)

        iterator = range(N)
        if verbose and tqdm is not None:
            iterator = tqdm(iterator, desc='axis align (3D)')
        elif verbose and tqdm is None:
            print(f'axis align (3D): {N} clips')

        for i_n in iterator:
            sk = s[i_n]  # (M,T,V,C)
            if not np.isfinite(sk).any() or sk.sum() == 0.0:
                continue

            # ref person & frame (ù non-empty)
            ref_p, ref_t = 0, 0
            if sk[ref_p, ref_t].sum() == 0.0:
                found = False
                for tt in range(T):
                    for pp in range(M):
                        if sk[pp, tt].sum() != 0.0:
                            ref_p, ref_t = pp, tt
                            found = True
                            break
                    if found:
                        break
                if not found:
                    continue


            def calc_R(fr_: np.ndarray):
                pel = 0.5 * (fr_[L_HIP, :3] + fr_[R_HIP, :3])
                shm = 0.5 * (fr_[L_SH,  :3] + fr_[R_SH,  :3])
                v_z = shm - pel
                if not (np.isfinite(v_z).all() and np.linalg.norm(v_z) > eps):
                    return np.eye(3, dtype=np.float32)

                axis_z = np.cross(_unit(v_z, eps), _unit(tgt_z, eps))
                ang_z  = angle_between(v_z, tgt_z, eps)
                Rz = rotation_matrix(axis_z, ang_z, eps) if (np.linalg.norm(axis_z) > eps and np.isfinite(ang_z)) else np.eye(3, dtype=np.float32)

                rsh = fr_[R_SH, :3] @ Rz.T
                lsh = fr_[L_SH, :3] @ Rz.T
                v_x = lsh - rsh
                if not (np.isfinite(v_x).all() and np.linalg.norm(v_x) > eps):
                    Rx = np.eye(3, dtype=np.float32)
                else:
                    axis_x = np.cross(_unit(v_x, eps), _unit(tgt_x, eps))
                    ang_x  = angle_between(v_x, tgt_x, eps)
                    Rx = rotation_matrix(axis_x, ang_x, eps) if (np.linalg.norm(axis_x) > eps and np.isfinite(ang_x)) else np.eye(3, dtype=np.float32)

                return (Rx @ Rz).astype(np.float32)

            if align_axes_mode == "clip":
                R = calc_R(sk[ref_p, ref_t])
                if not np.allclose(R, np.eye(3), atol=1e-6):
                    for p in range(M):
                        for t in range(T):
                            fr = sk[p, t]
                            if fr.sum() == 0.0: 
                                continue
                            fr[:, :3] = fr[:, :3] @ R.T

            elif align_axes_mode == "frame":
                for t in range(T):

                    p0 = next((pp for pp in range(M) if sk[pp, t].sum() != 0.0), None)
                    if p0 is None:
                        continue
                    R = calc_R(sk[p0, t])
                    if np.allclose(R, np.eye(3), atol=1e-6):
                        continue
                    for p in range(M):
                        fr = sk[p, t]
                        if fr.sum() == 0.0:
                            continue
                        fr[:, :3] = fr[:, :3] @ R.T

            s[i_n] = sk

    # 3) Pelvis-relative centering
    if do_center:
        dims = 2 if (third_channel.lower() == "score" and center_xy_only) else min(3, C)
        for i_n in range(N):
            for p in range(M):
                for t in range(T):
                    fr = s[i_n, p, t]
                    if fr.sum() == 0.0:
                        continue
                    pel = 0.5 * (fr[L_HIP, :dims] + fr[R_HIP, :dims])
                    fr[:, :dims] -= pel

    # 4) Scale normalization
    if scale_by in ("shoulder", "torso"):
        if not is_3d:

            dims = 2
        else:
            dims = min(3, C)

        for i_n in range(N):
            sk = s[i_n]
            if sk.sum() == 0.0:
                continue

            ref_p, ref_t = 0, 0
            if sk[ref_p, ref_t].sum() == 0.0:
                found = False
                for tt in range(T):
                    for pp in range(M):
                        if sk[pp, tt].sum() != 0.0:
                            ref_p, ref_t = pp, tt
                            found = True
                            break
                    if found: break
                if not found:
                    continue

            fr_ref = sk[ref_p, ref_t]
            if scale_by == "shoulder":
                width = _norm(fr_ref[L_SH, :dims] - fr_ref[R_SH, :dims], eps)
            else:  # "torso"
                pel = 0.5 * (fr_ref[L_HIP, :dims] + fr_ref[R_HIP, :dims])
                shm = 0.5 * (fr_ref[L_SH,  :dims] + fr_ref[R_SH,  :dims])
                width = _norm(shm - pel, eps)

            if not (np.isfinite(width) and width > 0):
                continue

            for p in range(M):
                for t in range(T):
                    fr = sk[p, t]
                    if fr.sum() == 0.0: 
                        continue
                    fr[:, :dims] /= width

            s[i_n] = sk

    # back to (N,C,T,V,M)
    return np.transpose(s, (0, 4, 2, 3, 1))

__all__ = ["pre_normalization", "angle_between", "rotation_matrix"]
