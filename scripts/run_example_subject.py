#!/usr/bin/env python3
"""
Run one synthetic/example subject through the CRRP HCP model core.

This script is meant as a lightweight smoke test for the repository. It does not
use raw HCP data. Instead, it creates synthetic cortical time series, a toy
cortical Laplacian, and three route-specific cortex-to-subcortex observation
matrices, then writes example CSV files in the same schema as the manuscript
analysis pipeline expects.

Usage:
    python scripts/run_example_subject.py
    python scripts/run_example_subject.py --output-dir outputs/example_subject --n-cortex 30 --t-tr 150

Outputs:
    example_subject_global_biomarkers.csv
    example_subject_parcel_biomarkers.csv
    example_subject_network_biomarkers.csv
    example_model_arrays.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from either repo root or scripts/.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
for p in [REPO_ROOT / "src", REPO_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from crrp_hcp_model_core import (  # noqa: E402
    aggregate_network_df,
    result_to_global_row,
    result_to_parcel_df,
    simulate_hierarchical_virtual_crrp,
)


def make_laplacian(rng: np.random.Generator, n: int) -> np.ndarray:
    """Create a small positive graph Laplacian from a random symmetric SC matrix."""
    sc = rng.random((n, n))
    sc = 0.5 * (sc + sc.T)
    np.fill_diagonal(sc, 0.0)
    # Keep a sparse-ish backbone.
    threshold = np.quantile(sc[sc > 0], 0.70)
    sc = np.where(sc >= threshold, sc, 0.0)
    degree = np.diag(sc.sum(axis=1))
    lap = degree - sc
    # Normalize to avoid overly large drift terms.
    scale = np.linalg.norm(lap)
    if scale > 0:
        lap = lap / scale
    # Add small leak for numerical stability.
    lap = lap + 1e-3 * np.eye(n)
    return lap


def make_synthetic_timeseries(rng: np.random.Generator, t: int, n: int) -> np.ndarray:
    """Generate simple correlated AR(1)-like cortical time series."""
    latent_dim = max(3, min(8, n // 4))
    loadings = rng.normal(0, 0.5, size=(latent_dim, n))
    latent = np.zeros((t, latent_dim))
    noise = rng.normal(0, 1, size=(t, latent_dim))
    for k in range(1, t):
        latent[k] = 0.85 * latent[k - 1] + 0.35 * noise[k]
    x = latent @ loadings + 0.25 * rng.normal(size=(t, n))
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)
    return x


def make_observation_matrix(rng: np.random.Generator, n_sub: int, n_cortex: int) -> np.ndarray:
    """Create a nonnegative row-normalized cortex-to-subcortex observation matrix."""
    w = rng.gamma(shape=1.5, scale=1.0, size=(n_sub, n_cortex))
    # Make each subcortical unit observe a subset more strongly.
    mask = rng.random((n_sub, n_cortex)) < 0.35
    w = w * mask
    # Ensure no empty rows.
    for i in range(n_sub):
        if np.all(w[i] == 0):
            w[i, rng.integers(0, n_cortex)] = 1.0
    w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-8)
    return w


def make_network_mapping(parcel_labels: list[str]) -> dict[str, str]:
    """Assign example parcels to canonical Yeo-7-style network labels."""
    networks = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    return {label: networks[i % len(networks)] for i, label in enumerate(parcel_labels)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one synthetic CRRP HCP example subject.")
    parser.add_argument("--output-dir", default="outputs/example_subject", help="Directory for example outputs.")
    parser.add_argument("--subject-id", default="example_subject", help="Subject ID to write into output CSVs.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--n-cortex", type=int, default=24, help="Number of cortical parcels in the toy example.")
    parser.add_argument("--t-emp", type=int, default=240, help="Length of synthetic empirical time series.")
    parser.add_argument("--t-tr", type=int, default=120, help="Recorded simulation TRs after warmup.")
    parser.add_argument("--warmup-tr", type=int, default=20, help="Warmup TRs before recording biomarkers.")
    parser.add_argument("--tr", type=float, default=0.72, help="TR in seconds.")
    parser.add_argument("--dt", type=float, default=0.02, help="Integration step. Must divide TR exactly.")
    parser.add_argument("--g-lr", type=float, default=50.0, help="CRRP route plasticity learning rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    nc = args.n_cortex

    p_emp = make_synthetic_timeseries(rng, args.t_emp, nc)
    lap = make_laplacian(rng, nc)

    # Route sizes are arbitrary for the synthetic example; real analyses should
    # use route-specific HCP/atlas-derived cortex-subcortex SC blocks.
    w_tha = make_observation_matrix(rng, max(4, nc // 5), nc)
    w_bg = make_observation_matrix(rng, max(5, nc // 4), nc)
    w_cb = make_observation_matrix(rng, max(6, nc // 3), nc)

    result = simulate_hierarchical_virtual_crrp(
        T_tr=args.t_tr,
        TR=args.tr,
        dt=args.dt,
        P_emp=p_emp,
        L=lap,
        W_c2s_tha=w_tha,
        W_c2s_bg=w_bg,
        W_c2s_cb=w_cb,
        g_lr=args.g_lr,
        warmup_tr=args.warmup_tr,
        seed=args.seed,
    )

    parcel_labels = [f"parcel_{i:03d}" for i in range(nc)]
    global_row = result_to_global_row(result, args.subject_id)
    parcel_df = result_to_parcel_df(result, args.subject_id, parcel_labels=parcel_labels)
    network_df = aggregate_network_df(parcel_df, make_network_mapping(parcel_labels))

    pd.DataFrame([global_row]).to_csv(outdir / "example_subject_global_biomarkers.csv", index=False)
    parcel_df.to_csv(outdir / "example_subject_parcel_biomarkers.csv", index=False)
    network_df.to_csv(outdir / "example_subject_network_biomarkers.csv", index=False)

    np.savez_compressed(
        outdir / "example_model_arrays.npz",
        P=result["P"],
        Pg_tha=result["Pg_tha"],
        Pg_bg=result["Pg_bg"],
        Pg_cb=result["Pg_cb"],
        Pg_fus=result["Pg_fus"],
        g=result["g"],
        e_fus=result["e_fus"],
        r=result["r"],
        K_star=result["K_star"],
    )

    print("Example CRRP subject completed.")
    print(f"  Output directory: {outdir.resolve()}")
    print("  Key global biomarkers:")
    for key in [
        "CRRP_Allocation_global_mean",
        "CRRP_Flexibility_global_mean",
        "CRRP_Residual_global_mean",
        "CRRP_Contrast_global_mean",
        "CRRP_Switching_global_mean",
        "E_Pobs_tha_mean",
        "E_Pobs_bg_mean",
        "E_Pobs_cb_mean",
    ]:
        if key in global_row:
            print(f"    {key}: {global_row[key]:.6g}")


if __name__ == "__main__":
    main()
