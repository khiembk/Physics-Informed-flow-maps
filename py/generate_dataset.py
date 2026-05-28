"""
PDE dataset generator for Physics-Informed Flow Maps.

Usage:
    python py/generate_dataset.py \\
        --config py/configs/data/navier_stokes_2d.yaml \\
        --output_dir /scratch/user/u.kt348068/physics_informedPDE

Generates:
    <output_dir>/<system>/train.npz
    <output_dir>/<system>/test.npz
    <output_dir>/<system>/meta.json

Each NPZ contains:
    x0     : float32 [N, C, H, W]  initial states
    xT     : float32 [N, C, H, W]  final states at time T
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from solvers import SOLVERS


# ---------------------------------------------------------------------------
# Batch generator (runs solver in chunks to avoid OOM on large N)
# ---------------------------------------------------------------------------

def _generate_split(system: str, cfg: dict, N: int,
                    seed: int, batch_size: int = 64) -> dict:
    """Call the solver in batches and concatenate results."""
    solver = SOLVERS[system]
    all_x0, all_xT = [], []

    n_batches = (N + batch_size - 1) // batch_size
    for b in range(n_batches):
        n_b = min(batch_size, N - b * batch_size)
        batch_cfg = {**cfg, "N": n_b}
        result = solver(batch_cfg, seed=seed + b * 1000)
        all_x0.append(result["x0"])
        all_xT.append(result["xT"])
        print(f"    batch {b+1}/{n_batches} done  ({n_b} samples)")

    return {
        **result,   # carry metadata from last batch
        "x0": np.concatenate(all_x0, axis=0),
        "xT": np.concatenate(all_xT, axis=0),
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save(out_dir: str, split: str, data: dict):
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"{split}.npz")
    np.savez_compressed(
        npz_path,
        x0=data["x0"],
        xT=data["xT"],
    )
    print(f"  Saved {npz_path}  "
          f"x0={data['x0'].shape}  xT={data['xT'].shape}  "
          f"dtype={data['x0'].dtype}")


def _save_meta(out_dir: str, data: dict, cfg: dict):
    meta = {
        "t0": data["t0"],
        "tT": data["tT"],
        "dt": data["dt"],
        "channel_names": data["channel_names"],
        "params": data["params"],
        "N_train": cfg["N_train"],
        "N_test":  cfg["N_test"],
        "H": cfg.get("H", 64),
        "W": cfg.get("W", 64),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True,
                        help="Path to YAML config (e.g. configs/data/navier_stokes_2d.yaml)")
    parser.add_argument("--output_dir", required=True,
                        help="Root directory for saved datasets")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Samples per solver batch (reduce if OOM)")
    parser.add_argument("--systems",    nargs="+", default=None,
                        help="Override system list (default: from config)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    system   = cfg["system"]
    N_train  = cfg["N_train"]
    N_test   = cfg["N_test"]
    base_seed = cfg.get("seed", 0)

    if system not in SOLVERS:
        raise ValueError(f"Unknown system '{system}'. Available: {list(SOLVERS)}")

    out_dir = os.path.join(args.output_dir, system)
    print(f"\n{'='*60}")
    print(f"System  : {system}")
    print(f"Output  : {out_dir}")
    print(f"N_train : {N_train}   N_test : {N_test}")
    print(f"{'='*60}")

    # Training split
    print(f"\n[train] Generating {N_train} samples ...")
    t0 = time.time()
    train_data = _generate_split(system, cfg, N_train,
                                  seed=base_seed, batch_size=args.batch_size)
    print(f"  Elapsed: {time.time()-t0:.1f}s")
    _save(out_dir, "train", train_data)

    # Test split
    print(f"\n[test]  Generating {N_test} samples ...")
    t0 = time.time()
    test_data = _generate_split(system, cfg, N_test,
                                 seed=base_seed + 9999, batch_size=args.batch_size)
    print(f"  Elapsed: {time.time()-t0:.1f}s")
    _save(out_dir, "test", test_data)

    _save_meta(out_dir, train_data, cfg)
    print(f"\nDone. Meta written to {out_dir}/meta.json")


if __name__ == "__main__":
    main()
