import os
import numpy as np
from numpy.lib.format import open_memmap
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = (BASE / '..' / 'data').resolve()

sets = {
    'train', 'val'
}

datasets = {
    'rgbd'
}

parts = {
    'joint', 'bone'
}

from tqdm import tqdm

for dataset in datasets:
    for set in sets:
        for part in parts:
            print(dataset, set, part)
            in_path  = DATA / dataset / f'{set}_data_{part}.npy'
            out_path = DATA / dataset / f'{set}_data_{part}_motion.npy'

            if not in_path.exists():
                print(f"[WARN] File not found: {in_path} (skipped)")
                continue

            try:
                data = np.load(in_path)  # expected shape: (N, C, T, V, M)
            except Exception as e:
                print(f"[WARN] Failed to load: {in_path} -> {e} (skipped)")
                continue

            if data.ndim != 5:
                print(f"[WARN] Unexpected shape {data.shape} for {in_path} (skipped)")
                continue

            N, C, T, V, M = data.shape
            if T < 2:
                print(f"[WARN] T={T} < 2, nothing to diff: {in_path} (skipped)")
                continue

            fp_sp = open_memmap(out_path, dtype='float32', mode='w+', shape=(N, C, T, V, M))
            fp_sp[:] = 0
            # motion: x[t] - x[t-1]
            fp_sp[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]

            print(f"[OK] Saved motion: {out_path} (shape={fp_sp.shape})")
