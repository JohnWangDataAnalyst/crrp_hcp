#!/usr/bin/env python3
"""
CRRP HCP CSV analysis pipeline
==============================

This script collects the analysis used for the CRRP HCP manuscript from CSV
outputs. It is designed to run from the directory containing these files:

    crrp_subject_parcel_biomarkers.csv
    crrp_subject_network_biomarkers.csv
    crrp_subject_global_biomarkers.csv
    crrp_shuffle_parcel_biomarkers.csv
    crrp_shuffle_network_biomarkers.csv
    crrp_shuffle_global_biomarkers.csv
    crrp_failed_subjects.csv

It is robust to missing files: analyses that do not have the required CSVs are
skipped with a warning. Results are written to:

    crrp_hcp_analysis_outputs/tables
    crrp_hcp_analysis_outputs/figures

Main analyses included
----------------------
1. Data inventory and failed-subject summary.
2. Global biomarker summary.
3. Network-level biomarker summary and rank-frequency tables.
4. Parcel-level Allocation reproducibility across subjects:
       - all pairwise subject-subject correlations
       - subject-to-leave-one-out-template correlations
5. Intact-vs-shuffle paired tests for global biomarkers.
6. Intact-vs-shuffle network-profile correlations and gradient-range reduction.
7. Route hierarchy tests if THA/BG/CB observer-energy columns are available.
8. Manuscript-ready summary text file.

Author: Zheng Wang / ChatGPT analysis draft
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CSV_NAMES = {
    "subject_parcel": "crrp_subject_parcel_biomarkers.csv",
    "subject_network": "crrp_subject_network_biomarkers.csv",
    "subject_global": "crrp_subject_global_biomarkers.csv",
    "shuffle_parcel": "crrp_shuffle_parcel_biomarkers.csv",
    "shuffle_network": "crrp_shuffle_network_biomarkers.csv",
    "shuffle_global": "crrp_shuffle_global_biomarkers.csv",
    "failed_subjects": "crrp_failed_subjects.csv",
}

# Expected biomarker columns. The code also auto-detects any CRRP_* columns.
PRIMARY_METRICS = [
    "CRRP_Allocation",
    "CRRP_Flexibility",
    "CRRP_Residual",
    "CRRP_Contrast",
    "CRRP_Switching",
]

NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        warnings.warn(f"Missing file: {path.name}; skipping related analyses.")
        return None
    return pd.read_csv(path)


def standardize_subject_id(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    df = df.copy()
    for c in ["subject", "Subject", "sub", "subj", "subjectID", "SubjectID", "id"]:
        if c in df.columns and "subject_id" not in df.columns:
            df = df.rename(columns={c: "subject_id"})
    if "subject_id" in df.columns:
        df["subject_id"] = df["subject_id"].astype(str)
    return df


def metric_columns(df: pd.DataFrame | None) -> list[str]:
    if df is None:
        return []
    cols = []
    for c in df.columns:
        if c.startswith("CRRP_"):
            cols.append(c)
    # Include common energy columns if present.
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["energy", "eobs", "observer"]):
            if pd.api.types.is_numeric_dtype(df[c]) and c not in cols:
                cols.append(c)
    return cols


def available_primary_metrics(df: pd.DataFrame | None) -> list[str]:
    if df is None:
        return []
    metrics = [m for m in PRIMARY_METRICS if m in df.columns]
    for c in metric_columns(df):
        if c not in metrics:
            metrics.append(c)
    return metrics


def mean_sd(x: pd.Series | np.ndarray) -> str:
    arr = pd.Series(x).dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return "NA"
    return f"{np.mean(arr):.6g} ± {np.std(arr, ddof=1):.6g}"


def cohen_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return np.nan
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return np.nan
    return float(np.mean(diff) / sd)


def bh_fdr(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q_valid = np.empty_like(q_ranked)
    q_valid[order] = q_ranked
    q[valid] = q_valid
    return q


def paired_tests(intact: np.ndarray, shuffle: np.ndarray) -> dict:
    intact = np.asarray(intact, dtype=float)
    shuffle = np.asarray(shuffle, dtype=float)
    mask = np.isfinite(intact) & np.isfinite(shuffle)
    intact = intact[mask]
    shuffle = shuffle[mask]
    diff = intact - shuffle
    out = {
        "n": len(diff),
        "intact_mean": np.nanmean(intact) if len(diff) else np.nan,
        "intact_sd": np.nanstd(intact, ddof=1) if len(diff) > 1 else np.nan,
        "shuffle_mean": np.nanmean(shuffle) if len(diff) else np.nan,
        "shuffle_sd": np.nanstd(shuffle, ddof=1) if len(diff) > 1 else np.nan,
        "diff_mean": np.nanmean(diff) if len(diff) else np.nan,
        "dz": cohen_dz(diff),
        "t": np.nan,
        "p_t": np.nan,
        "wilcoxon_stat": np.nan,
        "p_wilcoxon": np.nan,
    }
    if stats is not None and len(diff) > 1:
        tt = stats.ttest_rel(intact, shuffle, nan_policy="omit")
        out["t"] = float(tt.statistic)
        out["p_t"] = float(tt.pvalue)
        try:
            wt = stats.wilcoxon(intact, shuffle, zero_method="wilcox", alternative="two-sided")
            out["wilcoxon_stat"] = float(wt.statistic)
            out["p_wilcoxon"] = float(wt.pvalue)
        except Exception:
            pass
    return out


def find_route_energy_columns(df: pd.DataFrame | None) -> dict[str, str]:
    """Find one energy-like column for each THA/BG/CB route if available."""
    if df is None:
        return {}
    candidates = {}
    for route in ["THA", "BG", "CB"]:
        route_patterns = [route.lower()]
        if route == "THA":
            route_patterns += ["thalam"]
        elif route == "BG":
            route_patterns += ["basal", "ganglia", "stri"]
        elif route == "CB":
            route_patterns += ["cereb", "cb"]
        matches = []
        for c in df.columns:
            lc = c.lower()
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            if any(rp in lc for rp in route_patterns) and any(k in lc for k in ["energy", "eobs", "observer", "lcr", "lc", "gain"]):
                matches.append(c)
        if matches:
            # Prefer observer energy over generic gain if available.
            matches = sorted(matches, key=lambda x: ("observer" not in x.lower(), "energy" not in x.lower(), len(x)))
            candidates[route] = matches[0]
    return candidates


def reorder_networks(df: pd.DataFrame, network_col: str = "network") -> pd.DataFrame:
    if network_col not in df.columns:
        return df
    df = df.copy()
    present = list(df[network_col].dropna().unique())
    order = [n for n in NETWORK_ORDER if n in present] + [n for n in present if n not in NETWORK_ORDER]
    df[network_col] = pd.Categorical(df[network_col], categories=order, ordered=True)
    return df.sort_values(network_col)


# -----------------------------------------------------------------------------
# Loading and derived tables
# -----------------------------------------------------------------------------

def load_all(input_dir: Path) -> dict[str, pd.DataFrame | None]:
    data = {}
    for key, fname in CSV_NAMES.items():
        data[key] = standardize_subject_id(read_csv_if_exists(input_dir / fname))
    return data


def derive_global_from_parcel(parcel_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if parcel_df is None or "subject_id" not in parcel_df.columns:
        return None
    metrics = available_primary_metrics(parcel_df)
    if not metrics:
        return None
    return parcel_df.groupby("subject_id", as_index=False)[metrics].mean()


def derive_network_from_parcel(parcel_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if parcel_df is None or "subject_id" not in parcel_df.columns or "network" not in parcel_df.columns:
        return None
    metrics = available_primary_metrics(parcel_df)
    if not metrics:
        return None
    return parcel_df.groupby(["subject_id", "network"], as_index=False, observed=True)[metrics].mean()


def collapse_shuffle(df: pd.DataFrame | None, group_cols: list[str], metrics: list[str]) -> pd.DataFrame | None:
    """Average shuffle randomizations within subject/network/region if needed."""
    if df is None:
        return None
    if not all(c in df.columns for c in group_cols):
        return None
    present_metrics = [m for m in metrics if m in df.columns]
    if not present_metrics:
        return None
    return df.groupby(group_cols, as_index=False, observed=True)[present_metrics].mean()


# -----------------------------------------------------------------------------
# Analyses
# -----------------------------------------------------------------------------

def data_inventory(data: dict[str, pd.DataFrame | None], tables_dir: Path) -> pd.DataFrame:
    rows = []
    for key, df in data.items():
        if df is None:
            rows.append({"dataset": key, "present": False, "rows": 0, "columns": 0, "subjects": np.nan, "metrics": ""})
        else:
            subj_count = df["subject_id"].nunique() if "subject_id" in df.columns else np.nan
            rows.append({
                "dataset": key,
                "present": True,
                "rows": len(df),
                "columns": len(df.columns),
                "subjects": subj_count,
                "metrics": "; ".join(available_primary_metrics(df)),
            })
    inv = pd.DataFrame(rows)
    inv.to_csv(tables_dir / "data_inventory.csv", index=False)
    return inv


def summarize_global(global_df: pd.DataFrame | None, tables_dir: Path) -> pd.DataFrame | None:
    if global_df is None:
        return None
    metrics = available_primary_metrics(global_df)
    if not metrics:
        return None
    rows = []
    for m in metrics:
        x = global_df[m].dropna().to_numpy(dtype=float)
        rows.append({
            "metric": m,
            "n": len(x),
            "mean": np.mean(x) if len(x) else np.nan,
            "sd": np.std(x, ddof=1) if len(x) > 1 else np.nan,
            "median": np.median(x) if len(x) else np.nan,
            "min": np.min(x) if len(x) else np.nan,
            "max": np.max(x) if len(x) else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "global_biomarker_summary.csv", index=False)
    return out


def summarize_network(network_df: pd.DataFrame | None, tables_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if network_df is None or "network" not in network_df.columns:
        return None, None
    metrics = available_primary_metrics(network_df)
    if not metrics:
        return None, None
    rows = []
    for net, g in network_df.groupby("network", observed=True):
        for m in metrics:
            x = g[m].dropna().to_numpy(dtype=float)
            rows.append({
                "network": net,
                "metric": m,
                "n_subjects": g["subject_id"].nunique() if "subject_id" in g.columns else len(g),
                "mean": np.mean(x) if len(x) else np.nan,
                "sd": np.std(x, ddof=1) if len(x) > 1 else np.nan,
                "median": np.median(x) if len(x) else np.nan,
            })
    summary = reorder_networks(pd.DataFrame(rows))
    summary.to_csv(tables_dir / "network_biomarker_summary.csv", index=False)

    # Rank frequency: for each metric, count which network ranks #1 per subject.
    rank_rows = []
    if "subject_id" in network_df.columns:
        for m in metrics:
            sub = network_df[["subject_id", "network", m]].dropna()
            if sub.empty:
                continue
            idx = sub.groupby("subject_id")[m].idxmax()
            winners = sub.loc[idx, "network"].value_counts(normalize=True).rename("rank1_fraction")
            counts = sub.loc[idx, "network"].value_counts().rename("rank1_count")
            for net in sorted(winners.index.astype(str)):
                rank_rows.append({
                    "metric": m,
                    "network": net,
                    "rank1_count": int(counts.loc[net]),
                    "rank1_fraction": float(winners.loc[net]),
                })
    rank_df = reorder_networks(pd.DataFrame(rank_rows)) if rank_rows else None
    if rank_df is not None:
        rank_df.to_csv(tables_dir / "network_rank1_frequency.csv", index=False)
    return summary, rank_df


def parcel_reliability(parcel_df: pd.DataFrame | None, tables_dir: Path, figures_dir: Path, metric: str = "CRRP_Allocation") -> pd.DataFrame | None:
    if parcel_df is None or metric not in parcel_df.columns:
        return None
    if not {"subject_id", "region"}.issubset(parcel_df.columns):
        return None
    wide = parcel_df.pivot(index="subject_id", columns="region", values=metric)
    wide = wide.dropna(axis=1, how="any")
    X = wide.to_numpy(dtype=float)
    if X.shape[0] < 3 or X.shape[1] < 2:
        return None

    C = np.corrcoef(X)
    pairwise = C[np.triu_indices_from(C, k=1)]

    loo = []
    subjects = wide.index.to_list()
    for i, sid in enumerate(subjects):
        template = np.delete(X, i, axis=0).mean(axis=0)
        r = np.corrcoef(X[i], template)[0, 1]
        loo.append({"subject_id": sid, "subject_to_loo_template_r": r})
    loo_df = pd.DataFrame(loo)
    loo_df.to_csv(tables_dir / f"{metric}_subject_to_template_correlations.csv", index=False)

    summary = pd.DataFrame([
        {
            "metric": metric,
            "reliability_type": "pairwise_subject_subject",
            "n": len(pairwise),
            "mean_r": np.nanmean(pairwise),
            "sd_r": np.nanstd(pairwise, ddof=1),
            "median_r": np.nanmedian(pairwise),
            "min_r": np.nanmin(pairwise),
            "max_r": np.nanmax(pairwise),
        },
        {
            "metric": metric,
            "reliability_type": "subject_to_leave_one_out_template",
            "n": len(loo_df),
            "mean_r": loo_df["subject_to_loo_template_r"].mean(),
            "sd_r": loo_df["subject_to_loo_template_r"].std(ddof=1),
            "median_r": loo_df["subject_to_loo_template_r"].median(),
            "min_r": loo_df["subject_to_loo_template_r"].min(),
            "max_r": loo_df["subject_to_loo_template_r"].max(),
        },
    ])
    summary.to_csv(tables_dir / f"{metric}_parcel_map_reliability_summary.csv", index=False)

    # Histogram of subject-to-template correlations.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(loo_df["subject_to_loo_template_r"].dropna(), bins=25, alpha=0.85)
    ax.set_xlabel(f"{metric}: subject-to-LOO-template r")
    ax.set_ylabel("Number of subjects")
    ax.set_title("Parcel-level map reproducibility")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{metric}_subject_to_template_hist.png", dpi=220)
    plt.close(fig)

    return summary


def intact_vs_shuffle_global(
    subject_global: pd.DataFrame | None,
    shuffle_global: pd.DataFrame | None,
    tables_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame | None:
    if subject_global is None or shuffle_global is None:
        return None
    if "subject_id" not in subject_global.columns or "subject_id" not in shuffle_global.columns:
        return None
    metrics = [m for m in available_primary_metrics(subject_global) if m in shuffle_global.columns]
    if not metrics:
        return None

    shuf = collapse_shuffle(shuffle_global, ["subject_id"], metrics)
    if shuf is None:
        return None
    rows = []
    for m in metrics:
        merged = subject_global[["subject_id", m]].merge(shuf[["subject_id", m]], on="subject_id", suffixes=("_intact", "_shuffle"))
        if merged.empty:
            continue
        res = paired_tests(merged[f"{m}_intact"].to_numpy(), merged[f"{m}_shuffle"].to_numpy())
        res["metric"] = m
        rows.append(res)
    out = pd.DataFrame(rows)
    if out.empty:
        return None
    out["q_t"] = bh_fdr(out["p_t"])
    out["q_wilcoxon"] = bh_fdr(out["p_wilcoxon"])
    out = out[["metric"] + [c for c in out.columns if c != "metric"]]
    out.to_csv(tables_dir / "intact_vs_shuffle_global_paired_tests.csv", index=False)

    # Bar plot of intact vs shuffle means for primary metrics.
    plot_metrics = [m for m in PRIMARY_METRICS if m in metrics]
    if plot_metrics:
        means = []
        labels = []
        for m in plot_metrics:
            row = out[out["metric"] == m].iloc[0]
            means.extend([row["intact_mean"], row["shuffle_mean"]])
            labels.extend([m.replace("CRRP_", "") + "\nintact", m.replace("CRRP_", "") + "\nshuffle"])
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.65), 4.5))
        ax.bar(range(len(means)), means, alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Mean value")
        ax.set_title("Global intact-vs-shuffle biomarkers")
        fig.tight_layout()
        fig.savefig(figures_dir / "intact_vs_shuffle_global_bars.png", dpi=220)
        plt.close(fig)
    return out


def intact_vs_shuffle_network(
    subject_network: pd.DataFrame | None,
    shuffle_network: pd.DataFrame | None,
    tables_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame | None:
    if subject_network is None or shuffle_network is None:
        return None
    if "network" not in subject_network.columns or "network" not in shuffle_network.columns:
        return None
    metrics = [m for m in available_primary_metrics(subject_network) if m in shuffle_network.columns]
    if not metrics:
        return None

    # Average within subject/network first; then across subjects to network profile.
    group_cols = ["subject_id", "network"] if "subject_id" in shuffle_network.columns else ["network"]
    shuf_collapsed = collapse_shuffle(shuffle_network, group_cols, metrics)
    intact_collapsed = collapse_shuffle(subject_network, ["subject_id", "network"] if "subject_id" in subject_network.columns else ["network"], metrics)
    if shuf_collapsed is None or intact_collapsed is None:
        return None

    intact_profile = intact_collapsed.groupby("network", as_index=False, observed=True)[metrics].mean()
    shuffle_profile = shuf_collapsed.groupby("network", as_index=False, observed=True)[metrics].mean()

    rows = []
    for m in metrics:
        merged = intact_profile[["network", m]].merge(shuffle_profile[["network", m]], on="network", suffixes=("_intact", "_shuffle"))
        if len(merged) < 3:
            continue
        r = np.corrcoef(merged[f"{m}_intact"], merged[f"{m}_shuffle"])[0, 1]
        range_intact = merged[f"{m}_intact"].max() - merged[f"{m}_intact"].min()
        range_shuffle = merged[f"{m}_shuffle"].max() - merged[f"{m}_shuffle"].min()
        rows.append({
            "metric": m,
            "n_networks": len(merged),
            "intact_shuffle_network_profile_r": r,
            "intact_range": range_intact,
            "shuffle_range": range_shuffle,
            "range_reduction_factor": range_intact / range_shuffle if range_shuffle != 0 else np.inf,
        })

        # Scatter plot per metric.
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        ax.scatter(merged[f"{m}_intact"], merged[f"{m}_shuffle"], s=60, alpha=0.85)
        for _, row in merged.iterrows():
            ax.annotate(str(row["network"]), (row[f"{m}_intact"], row[f"{m}_shuffle"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
        mn = min(merged[f"{m}_intact"].min(), merged[f"{m}_shuffle"].min())
        mx = max(merged[f"{m}_intact"].max(), merged[f"{m}_shuffle"].max())
        ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
        ax.set_xlabel("Intact network mean")
        ax.set_ylabel("Shuffle network mean")
        ax.set_title(f"Network profile: {m}\nr = {r:.3f}")
        fig.tight_layout()
        fig.savefig(figures_dir / f"network_intact_vs_shuffle_{m}.png", dpi=220)
        plt.close(fig)

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(tables_dir / "intact_vs_shuffle_network_profile_correlations.csv", index=False)
    return out if not out.empty else None


def route_hierarchy(global_df: pd.DataFrame | None, tables_dir: Path) -> pd.DataFrame | None:
    route_cols = find_route_energy_columns(global_df)
    if set(route_cols.keys()) != {"THA", "BG", "CB"}:
        return None
    df = global_df[["subject_id", route_cols["THA"], route_cols["BG"], route_cols["CB"]]].dropna().copy()
    df = df.rename(columns={route_cols["THA"]: "THA", route_cols["BG"]: "BG", route_cols["CB"]: "CB"})
    if len(df) < 3:
        return None

    rows = []
    for route in ["THA", "BG", "CB"]:
        rows.append({"test": "route_mean", "comparison": route, "statistic": "mean_sd", "value": mean_sd(df[route])})

    if stats is not None:
        fr = stats.friedmanchisquare(df["THA"], df["BG"], df["CB"])
        rows.append({"test": "friedman", "comparison": "THA_BG_CB", "statistic": "chi2", "value": float(fr.statistic)})
        rows.append({"test": "friedman", "comparison": "THA_BG_CB", "statistic": "p", "value": float(fr.pvalue)})
        pairs = [("BG", "THA"), ("CB", "BG"), ("CB", "THA")]
        pvals = []
        pair_rows = []
        for a, b in pairs:
            wt = stats.wilcoxon(df[a], df[b], zero_method="wilcox", alternative="greater")
            diff = df[a].to_numpy() - df[b].to_numpy()
            pvals.append(float(wt.pvalue))
            pair_rows.append({
                "test": "paired_wilcoxon_greater",
                "comparison": f"{a}>{b}",
                "statistic": "W",
                "value": float(wt.statistic),
                "p": float(wt.pvalue),
                "dz": cohen_dz(diff),
            })
        qvals = bh_fdr(pvals)
        for pr, q in zip(pair_rows, qvals):
            pr["q"] = q
            rows.append(pr)

    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "route_hierarchy_statistics.csv", index=False)
    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_network_bars(network_summary: pd.DataFrame | None, figures_dir: Path) -> None:
    if network_summary is None or network_summary.empty:
        return
    primary = [m for m in PRIMARY_METRICS if m in network_summary["metric"].unique()]
    for m in primary:
        sub = network_summary[network_summary["metric"] == m].copy()
        sub = reorder_networks(sub)
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        x = np.arange(len(sub))
        ax.bar(x, sub["mean"], yerr=sub["sd"], capsize=3, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["network"].astype(str), rotation=35, ha="right")
        ax.set_ylabel(m)
        ax.set_title(f"Network-level {m}")
        fig.tight_layout()
        fig.savefig(figures_dir / f"network_bar_{m}.png", dpi=220)
        plt.close(fig)


def plot_subject_distribution(global_df: pd.DataFrame | None, figures_dir: Path) -> None:
    if global_df is None:
        return
    metrics = [m for m in PRIMARY_METRICS if m in global_df.columns]
    for m in metrics:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.hist(global_df[m].dropna(), bins=25, alpha=0.85)
        ax.set_xlabel(m)
        ax.set_ylabel("Number of subjects")
        ax.set_title(f"Subject-level distribution: {m}")
        fig.tight_layout()
        fig.savefig(figures_dir / f"global_hist_{m}.png", dpi=220)
        plt.close(fig)


def write_manuscript_numbers(
    tables_dir: Path,
    inv: pd.DataFrame,
    global_summary: pd.DataFrame | None,
    alloc_rel: pd.DataFrame | None,
    net_shuffle: pd.DataFrame | None,
    global_shuffle: pd.DataFrame | None,
) -> None:
    lines = []
    lines.append("CRRP HCP analysis manuscript-ready numbers")
    lines.append("=" * 52)
    lines.append("")

    if inv is not None:
        lines.append("Data inventory:")
        for _, row in inv.iterrows():
            if row["present"]:
                lines.append(f"- {row['dataset']}: {int(row['rows'])} rows, {row['subjects']} subjects")
        lines.append("")

    if global_summary is not None:
        lines.append("Global biomarkers:")
        for _, row in global_summary.iterrows():
            lines.append(f"- {row['metric']}: {row['mean']:.6g} ± {row['sd']:.6g}, median={row['median']:.6g}")
        lines.append("")

    if alloc_rel is not None:
        lines.append("Parcel-level Allocation reproducibility:")
        for _, row in alloc_rel.iterrows():
            lines.append(
                f"- {row['reliability_type']}: mean r={row['mean_r']:.3f} ± {row['sd_r']:.3f}, "
                f"median r={row['median_r']:.3f}, range={row['min_r']:.3f}–{row['max_r']:.3f}"
            )
        lines.append("")

    if net_shuffle is not None and not net_shuffle.empty:
        lines.append("Intact-vs-shuffle network-profile comparisons:")
        for _, row in net_shuffle.iterrows():
            lines.append(
                f"- {row['metric']}: r={row['intact_shuffle_network_profile_r']:.3f}, "
                f"range reduction={row['range_reduction_factor']:.2f}× "
                f"(intact={row['intact_range']:.6g}, shuffle={row['shuffle_range']:.6g})"
            )
        lines.append("")

    if global_shuffle is not None and not global_shuffle.empty:
        lines.append("Intact-vs-shuffle paired global tests:")
        for _, row in global_shuffle.iterrows():
            lines.append(
                f"- {row['metric']}: intact={row['intact_mean']:.6g} ± {row['intact_sd']:.6g}, "
                f"shuffle={row['shuffle_mean']:.6g} ± {row['shuffle_sd']:.6g}, "
                f"q={row['q_t']:.3g}, dz={row['dz']:.3f}"
            )
        lines.append("")

    (tables_dir / "manuscript_ready_numbers.txt").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def run_pipeline(input_dir: Path, output_dir: Path) -> None:
    tables_dir = ensure_dir(output_dir / "tables")
    figures_dir = ensure_dir(output_dir / "figures")

    data = load_all(input_dir)

    # Derive missing global/network summaries from parcel files where possible.
    if data["subject_global"] is None:
        data["subject_global"] = derive_global_from_parcel(data["subject_parcel"])
        if data["subject_global"] is not None:
            data["subject_global"].to_csv(tables_dir / "derived_subject_global_from_parcel.csv", index=False)

    if data["subject_network"] is None:
        data["subject_network"] = derive_network_from_parcel(data["subject_parcel"])
        if data["subject_network"] is not None:
            data["subject_network"].to_csv(tables_dir / "derived_subject_network_from_parcel.csv", index=False)

    if data["shuffle_global"] is None:
        data["shuffle_global"] = derive_global_from_parcel(data["shuffle_parcel"])
        if data["shuffle_global"] is not None:
            data["shuffle_global"].to_csv(tables_dir / "derived_shuffle_global_from_parcel.csv", index=False)

    if data["shuffle_network"] is None:
        data["shuffle_network"] = derive_network_from_parcel(data["shuffle_parcel"])
        if data["shuffle_network"] is not None:
            data["shuffle_network"].to_csv(tables_dir / "derived_shuffle_network_from_parcel.csv", index=False)

    inv = data_inventory(data, tables_dir)
    global_summary = summarize_global(data["subject_global"], tables_dir)
    network_summary, _rank_df = summarize_network(data["subject_network"], tables_dir)
    alloc_rel = parcel_reliability(data["subject_parcel"], tables_dir, figures_dir, metric="CRRP_Allocation")
    global_shuffle = intact_vs_shuffle_global(data["subject_global"], data["shuffle_global"], tables_dir, figures_dir)
    net_shuffle = intact_vs_shuffle_network(data["subject_network"], data["shuffle_network"], tables_dir, figures_dir)
    _route_stats = route_hierarchy(data["subject_global"], tables_dir)

    plot_network_bars(network_summary, figures_dir)
    plot_subject_distribution(data["subject_global"], figures_dir)

    # Failed subject summary.
    if data["failed_subjects"] is not None:
        data["failed_subjects"].to_csv(tables_dir / "failed_subjects_copy.csv", index=False)

    write_manuscript_numbers(tables_dir, inv, global_summary, alloc_rel, net_shuffle, global_shuffle)

    print("\nCRRP HCP CSV analysis complete.")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tables:           {tables_dir}")
    print(f"Figures:          {figures_dir}")
    print("Key summary:      tables/manuscript_ready_numbers.txt")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CRRP HCP analysis from CSV biomarker files.")
    parser.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing CRRP CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path("crrp_hcp_analysis_outputs"), help="Output directory.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(args.input_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
