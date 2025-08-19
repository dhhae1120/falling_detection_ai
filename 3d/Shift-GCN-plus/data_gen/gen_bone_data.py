import os
import numpy as np
from numpy.lib.format import open_memmap

from pathlib import Path
BASE = Path(__file__).resolve().parent
DATA = (BASE / '..' / 'data').resolve()

pairs = {
    'rgbd': (
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
        (5, 7), (5, 11), (6, 8), (6, 12), (7, 9), (8, 10), (11, 12), (11, 13),
        (12, 14), (13, 15), (14, 16)
    )
}

sets = {
    'train', 'val'
}

datasets = {
    'rgbd'
}

from tqdm import tqdm

for dataset in datasets:
    # Use dataset-specific pairs if available; otherwise fall back to 'rgbd'
    local_pairs = pairs.get(dataset, pairs['rgbd'])

    for set in sets:
        print(dataset, set)
        path = DATA / dataset / f'{set}_data_joint.npy'
        if not path.exists():
            print(f"[WARN] File not found: {path} (skipped)")
            continue

        try:
            data = np.load(path)
        except Exception as e:
            print(f"[WARN] Failed to load: {path} -> {e} (skipped)")
            continue

        N, C, T, V, M = data.shape
        out_dir = (DATA / dataset)
        out_path = out_dir / f'{set}_data_bone.npy'
        fp_sp = open_memmap(
            out_path, dtype='float32', mode='w+', shape=(N, 3, T, V, M)
        )

        fp_sp[:] = 0
        for v1, v2 in local_pairs:
            fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
