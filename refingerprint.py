"""Recompute the fingerprint stored in each optimized_actions/*.npz file."""
from pathlib import Path

import numpy as np

from cem import _segment_fingerprint


def main():
    actions_dir = Path("optimized_actions")
    data_dir = Path("data")
    files = sorted(actions_dir.glob("*.npz"))
    print(f"Refingerprinting {len(files)} files...")

    fps_seen = {}
    collisions = 0
    for i, f in enumerate(files):
        seg_id = f.stem
        data_path = data_dir / f"{seg_id}.csv"
        if not data_path.exists():
            print(f"  skip (no data): {seg_id}")
            continue
        d = np.load(f)
        fp_new = _segment_fingerprint(str(data_path))
        if fp_new in fps_seen:
            collisions += 1
            print(f"  collision: {seg_id} vs {fps_seen[fp_new]}  fp={fp_new[:12]}")
        fps_seen[fp_new] = seg_id

        np.savez(
            f,
            actions=d["actions"],
            best_cost=d["best_cost"],
            baseline_cost=d["baseline_cost"],
            fingerprint=fp_new,
        )
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)} done  unique_fps={len(fps_seen)}")

    print(f"\nFinal: {len(fps_seen)} unique fingerprints from {len(files)} files")
    print(f"Collisions: {collisions}")


if __name__ == "__main__":
    main()
