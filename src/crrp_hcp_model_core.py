#!/usr/bin/env python3
"""
CRRP HCP model core: three-observer LOCF + cerebellar-referenced route plasticity.

This module contains the simulation/fitting function used to generate the subject-level
CRRP biomarker CSV files analyzed by crrp_hcp_csv_analysis_pipeline.py.

Core model
----------
- Cortical plant P(t) is constrained by the cortical Laplacian L.
- THA, BG, and CB are route-specific observers with observation matrices W_c2s_*.
- CB is treated as a reference observer.
- A convex THA/BG mixture Pg_fus = g * Pg_tha + (1-g) * Pg_bg is adapted to match CB.
- g is updated by gradient descent on 1/2 ||Pg_fus - Pg_cb||^2.

Expected input convention
-------------------------
P_emp should be a cortical time-series matrix with shape (T, Nc), not a covariance
matrix. The function computes empirical FC as corrcoef(P_emp.T). If you already have
an FC matrix, use build_Astar_alphaK_from_fc() directly or adapt build_Astar_alphaK().

Author: Zheng Wang / ChatGPT cleanup draft
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.linalg import eigvals
from scipy.linalg import solve_continuous_are


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def stability_margin_compute(A: np.ndarray) -> float:
    """Return the dominant real eigenvalue of A."""
    return float(np.max(np.real(eigvals(A))))


def demean(X: np.ndarray) -> np.ndarray:
    """Column-demean a time-by-region matrix."""
    return X - X.mean(axis=0, keepdims=True)


def compute_K_star(A_star: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Feedback/control split used in the current LOCF implementation."""
    return A_star - L


def lqe_P_virtual(
    A_P: np.ndarray,
    W_c2s: np.ndarray,
    q: float = 1e-2,
    r: float = 1e-2,
) -> np.ndarray:
    """Continuous-time LQE/Kalman-style observer gain for virtual subcortical measurements."""
    Nc = A_P.shape[0]
    Ns = W_c2s.shape[0]
    Q = q * np.eye(Nc)
    R = r * np.eye(Ns)
    Pi = solve_continuous_are(A_P.T, W_c2s.T, Q, R)
    return Pi @ W_c2s.T @ np.linalg.inv(R)


def build_Astar_alphaK(
    P_emp: np.ndarray,
    L: np.ndarray,
    q: float = 0.01,
    r: float = 500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build A_star, alpha_p, and K_star from empirical cortical time series.

    Parameters
    ----------
    P_emp : array, shape (T, Nc)
        Cortical parcel time series. The function computes fc = corrcoef(P_emp.T).
    L : array, shape (Nc, Nc)
        Cortical Laplacian / structural backbone used for the LOCF split.
    q, r : float
        CARE weights.
    """
    Xemp = demean(np.asarray(P_emp, dtype=float))
    if Xemp.ndim != 2:
        raise ValueError("P_emp must be a 2D time-by-region matrix.")
    Nc = Xemp.shape[1]

    fc = np.corrcoef(Xemp.T)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    fc_norm = np.linalg.norm(fc)
    if fc_norm < 1e-12:
        raise ValueError("FC norm too small.")

    A = -fc / fc_norm
    B = np.eye(Nc)
    Q = q * np.eye(Nc)
    R = r * np.eye(Nc)

    X = solve_continuous_are(A, B, Q, R)
    A_star = -X

    K_star_s = compute_K_star(A_star, L)
    alpha_p = -np.diag(np.diag(K_star_s))
    K_star = (K_star_s - np.diag(np.diag(K_star_s))).copy()
    return A_star, alpha_p, K_star


def project_rhos(rho_tha: np.ndarray, rho_bg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project THA/BG route weights to nonnegative values whose sum is <= 1; CB is the residual."""
    rho_tha = np.maximum(rho_tha, 0.0)
    rho_bg = np.maximum(rho_bg, 0.0)
    s = rho_tha + rho_bg
    mask = s > 1.0
    if np.any(mask):
        rho_tha[mask] = rho_tha[mask] / s[mask]
        rho_bg[mask] = rho_bg[mask] / s[mask]
    rho_cb = 1.0 - rho_tha - rho_bg
    return rho_tha, rho_bg, rho_cb


# -----------------------------------------------------------------------------
# Simulation config
# -----------------------------------------------------------------------------

@dataclass
class CRRPSimConfig:
    T_tr: int
    TR: float = 0.72
    dt: float = 0.01

    qP_obs_tha: float = 0.2
    rS_obs_tha: float = 1.0
    qP_obs_bg: float = 0.2
    rS_obs_bg: float = 1.0
    qP_obs_cb: float = 0.2
    rS_obs_cb: float = 1.0

    proc_noise: float = 0.001
    sigma_s_tha: float = 0.001
    sigma_s_bg: float = 0.001
    sigma_s_cb: float = 0.001

    g_init: float = 0.5
    g_lr: float = 50.0
    g_clip: bool = True
    lambda_g: float = 0.0
    g_prior: float = 0.5
    lambda_smooth: float = 0.0
    warmup_tr: int = 0
    seed: int = 0


# -----------------------------------------------------------------------------
# Main three-observer CRRP simulator
# -----------------------------------------------------------------------------

def simulate_hierarchical_virtual_crrp(
    T_tr: int,
    TR: float,
    dt: float,
    P_emp: np.ndarray,
    L: np.ndarray,
    W_c2s_tha: np.ndarray,
    W_c2s_bg: np.ndarray,
    W_c2s_cb: np.ndarray,
    qP_obs_tha: float = 0.2,
    rS_obs_tha: float = 1.0,
    qP_obs_bg: float = 0.2,
    rS_obs_bg: float = 1.0,
    qP_obs_cb: float = 0.2,
    rS_obs_cb: float = 1.0,
    proc_noise: float = 0.001,
    sigma_s_tha: float = 0.001,
    sigma_s_bg: float = 0.001,
    sigma_s_cb: float = 0.001,
    g_init: float = 0.5,
    g_lr: float = 50.0,
    g_clip: bool = True,
    lambda_g: float = 0.0,
    g_prior: float = 0.5,
    lambda_smooth: float = 0.0,
    smooth_mat: np.ndarray | None = None,
    warmup_tr: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Three-observer virtual CRRP simulation with post-warmup biomarker recording.

    g update:
        Ph_fus = g * Ph_tha + (1-g) * Ph_bg
        loss = 1/2 ||Ph_fus - Ph_cb||^2
        grad_g = (Ph_fus - Ph_cb) * (Ph_tha - Ph_bg)
        g <- g - dt * g_lr * grad_g
    """
    rng = np.random.default_rng(seed)

    Nc = W_c2s_tha.shape[1]
    if W_c2s_bg.shape[1] != Nc or W_c2s_cb.shape[1] != Nc:
        raise ValueError("All W_c2s matrices must have the same cortical dimension Nc.")
    if L.shape != (Nc, Nc):
        raise ValueError(f"L shape {L.shape} does not match Nc={Nc}.")
    if warmup_tr < 0:
        raise ValueError("warmup_tr must be >= 0.")

    steps_per_TR = int(round(TR / dt))
    if abs(steps_per_TR * dt - TR) > 1e-9:
        raise ValueError("Choose dt so TR/dt is an integer.")

    total_tr = warmup_tr + T_tr

    A_star, alpha_p, K_star = build_Astar_alphaK(P_emp, L)

    Ls_tha = lqe_P_virtual(A_star, W_c2s_tha, q=qP_obs_tha, r=rS_obs_tha)
    Ls_bg = lqe_P_virtual(A_star, W_c2s_bg, q=qP_obs_bg, r=rS_obs_bg)
    Ls_cb = lqe_P_virtual(A_star, W_c2s_cb, q=qP_obs_cb, r=rS_obs_cb)

    P_store = np.zeros((T_tr, Nc))
    Ph_tha_store = np.zeros((T_tr, Nc))
    Ph_bg_store = np.zeros((T_tr, Nc))
    Ph_cb_store = np.zeros((T_tr, Nc))
    Ph_fus_store = np.zeros((T_tr, Nc))
    g_store = np.zeros((T_tr, Nc))
    e_fus_store = np.zeros((T_tr, Nc))
    r_store = np.zeros((T_tr, Nc))
    contrast_store = np.zeros((T_tr, Nc))

    E_Pobs_tha = 0.0
    E_Pobs_bg = 0.0
    E_Pobs_cb = 0.0
    E_cort = 0.0
    C_reg = 0.0
    E_g_update = 0.0

    P = 0.1 * rng.standard_normal(Nc)
    Ph_tha = 0.1 * rng.standard_normal(Nc)
    Ph_bg = 0.1 * rng.standard_normal(Nc)
    Ph_cb = 0.1 * rng.standard_normal(Nc)

    g = np.full(Nc, float(g_init)) + 0.1 * rng.standard_normal(Nc)
    if g_clip:
        g = np.clip(g, 0.0, 1.0)

    sqrt_dt = np.sqrt(dt)

    for k_tr in range(total_tr):
        record_idx = k_tr - warmup_tr
        record_now = k_tr >= warmup_tr

        if record_now:
            P_store[record_idx] = P
            Ph_tha_store[record_idx] = Ph_tha
            Ph_bg_store[record_idx] = Ph_bg
            Ph_cb_store[record_idx] = Ph_cb

            Ph_fus = g * Ph_tha + (1.0 - g) * Ph_bg
            e_fus = Ph_fus - Ph_cb
            r = Ph_cb - Ph_fus
            contrast = (Ph_tha - Ph_bg) ** 2

            Ph_fus_store[record_idx] = Ph_fus
            g_store[record_idx] = g
            e_fus_store[record_idx] = e_fus
            r_store[record_idx] = r
            contrast_store[record_idx] = contrast

        for _ in range(steps_per_TR):
            y_s_tha = W_c2s_tha @ P + sigma_s_tha * rng.standard_normal(W_c2s_tha.shape[0])
            y_s_bg = W_c2s_bg @ P + sigma_s_bg * rng.standard_normal(W_c2s_bg.shape[0])
            y_s_cb = W_c2s_cb @ P + sigma_s_cb * rng.standard_normal(W_c2s_cb.shape[0])

            innov_tha = y_s_tha - W_c2s_tha @ Ph_tha
            innov_bg = y_s_bg - W_c2s_bg @ Ph_bg
            innov_cb = y_s_cb - W_c2s_cb @ Ph_cb

            corr_tha = Ls_tha @ innov_tha
            corr_bg = Ls_bg @ innov_bg
            corr_cb = Ls_cb @ innov_cb

            corr_K_true = -alpha_p @ P + K_star @ P
            Ph_fus = g * Ph_tha + (1.0 - g) * Ph_bg
            corr_K_est = K_star @ Ph_fus

            if record_now:
                E_Pobs_tha += float(corr_tha @ corr_tha)
                E_Pobs_bg += float(corr_bg @ corr_bg)
                E_Pobs_cb += float(corr_cb @ corr_cb)
                C_reg += float(corr_K_true @ corr_K_true)
                E_cort += float(corr_K_est @ corr_K_est)

            Pdot = -(alpha_p @ P) + L @ P + K_star @ Ph_fus
            P = P + dt * Pdot + sqrt_dt * proc_noise * rng.standard_normal(Nc)

            Ph_tha = Ph_tha + dt * (A_star @ Ph_tha + corr_tha)
            Ph_bg = Ph_bg + dt * (A_star @ Ph_bg + corr_bg)
            Ph_cb = Ph_cb + dt * (A_star @ Ph_cb + corr_cb)

            Ph_fus = g * Ph_tha + (1.0 - g) * Ph_bg
            e_fus = Ph_fus - Ph_cb
            delta_route = Ph_tha - Ph_bg
            g_grad = e_fus * delta_route

            if lambda_g > 0.0:
                g_grad = g_grad + lambda_g * (g - g_prior)
            if lambda_smooth > 0.0:
                if smooth_mat is None:
                    raise ValueError("lambda_smooth > 0 but smooth_mat is None.")
                g_grad = g_grad + lambda_smooth * (smooth_mat @ g)

            g = g - dt * g_lr * g_grad
            if g_clip:
                g = np.clip(g, 0.0, 1.0)

            if record_now:
                E_g_update += float(g_grad @ g_grad)

    crrp_allocation = np.mean(g_store, axis=0)
    crrp_flexibility = np.var(g_store, axis=0)
    crrp_residual = np.mean(r_store ** 2, axis=0)
    crrp_contrast = np.mean(contrast_store, axis=0)

    if T_tr > 1:
        g_diff = np.diff(g_store, axis=0) / TR
        crrp_switching = np.mean(np.abs(g_diff), axis=0)
    else:
        crrp_switching = np.zeros(Nc)

    biomarkers_global = {
        "CRRP_Allocation_global_mean": float(np.mean(crrp_allocation)),
        "CRRP_Flexibility_global_mean": float(np.mean(crrp_flexibility)),
        "CRRP_Residual_global_mean": float(np.mean(crrp_residual)),
        "CRRP_Contrast_global_mean": float(np.mean(crrp_contrast)),
        "CRRP_Switching_global_mean": float(np.mean(crrp_switching)),
    }

    n_recorded_steps = max(T_tr * steps_per_TR, 1)
    energies = {
        "E_Pobs_tha": E_Pobs_tha,
        "E_Pobs_bg": E_Pobs_bg,
        "E_Pobs_cb": E_Pobs_cb,
        "E_cort": E_cort,
        "C_reg": C_reg,
        "E_g_update": E_g_update,
        "E_Pobs_tha_mean": E_Pobs_tha / n_recorded_steps,
        "E_Pobs_bg_mean": E_Pobs_bg / n_recorded_steps,
        "E_Pobs_cb_mean": E_Pobs_cb / n_recorded_steps,
        "E_cort_mean": E_cort / n_recorded_steps,
        "C_reg_mean": C_reg / n_recorded_steps,
        "E_g_update_mean": E_g_update / n_recorded_steps,
    }

    Aobs_tha = A_star - Ls_tha @ W_c2s_tha
    Aobs_bg = A_star - Ls_bg @ W_c2s_bg
    Aobs_cb = A_star - Ls_cb @ W_c2s_cb

    return {
        "alpha_p": np.diag(alpha_p),
        "K_star": K_star,
        "P": P_store,
        "Pg_tha": Ph_tha_store,
        "Pg_bg": Ph_bg_store,
        "Pg_cb": Ph_cb_store,
        "Pg_fus": Ph_fus_store,
        "g": g_store,
        "e_fus": e_fus_store,
        "r": r_store,
        "contrast_ts": contrast_store,
        "biomarkers": {
            "CRRP_Allocation": crrp_allocation,
            "CRRP_Flexibility": crrp_flexibility,
            "CRRP_Residual": crrp_residual,
            "CRRP_Contrast": crrp_contrast,
            "CRRP_Switching": crrp_switching,
            **biomarkers_global,
        },
        "energies": energies,
        "dt": dt,
        "TR": TR,
        "steps_per_TR": steps_per_TR,
        "warmup_tr": warmup_tr,
        "total_tr": total_tr,
        "Ls_tha": Ls_tha,
        "Ls_bg": Ls_bg,
        "Ls_cb": Ls_cb,
        "Ksub_tha": Ls_tha @ W_c2s_tha,
        "Ksub_bg": Ls_bg @ W_c2s_bg,
        "Ksub_cb": Ls_cb @ W_c2s_cb,
        "stability_margin": stability_margin_compute(A_star),
        "stability_margin_obs_tha": stability_margin_compute(Aobs_tha),
        "stability_margin_obs_bg": stability_margin_compute(Aobs_bg),
        "stability_margin_obs_cb": stability_margin_compute(Aobs_cb),
    }


# -----------------------------------------------------------------------------
# Convenience export helpers for making the manuscript CSVs
# -----------------------------------------------------------------------------

def result_to_global_row(result: dict[str, Any], subject_id: str) -> dict[str, Any]:
    """Convert one model result to one row for crrp_subject_global_biomarkers.csv."""
    row: dict[str, Any] = {"subject_id": str(subject_id)}
    row.update({k: v for k, v in result["biomarkers"].items() if np.isscalar(v)})
    row.update(result["energies"])
    for k in [
        "stability_margin",
        "stability_margin_obs_tha",
        "stability_margin_obs_bg",
        "stability_margin_obs_cb",
    ]:
        row[k] = result[k]
    return row


def result_to_parcel_df(
    result: dict[str, Any],
    subject_id: str,
    parcel_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Convert one model result to parcel-level CRRP biomarker rows."""
    alloc = result["biomarkers"]["CRRP_Allocation"]
    Nc = len(alloc)
    if parcel_labels is None:
        parcel_labels = [f"parcel_{i:03d}" for i in range(Nc)]
    if len(parcel_labels) != Nc:
        raise ValueError("parcel_labels length does not match number of cortical parcels.")

    df = pd.DataFrame({
        "subject_id": str(subject_id),
        "region": parcel_labels,
        "parcel_index": np.arange(Nc),
        "CRRP_Allocation": alloc,
        "CRRP_Flexibility": result["biomarkers"]["CRRP_Flexibility"],
        "CRRP_Residual": result["biomarkers"]["CRRP_Residual"],
        "CRRP_Contrast": result["biomarkers"]["CRRP_Contrast"],
        "CRRP_Switching": result["biomarkers"]["CRRP_Switching"],
    })
    return df


def aggregate_network_df(
    parcel_df: pd.DataFrame,
    parcel_to_network: dict[str, str] | pd.Series,
) -> pd.DataFrame:
    """Aggregate parcel-level biomarker rows to Yeo/network-level rows."""
    df = parcel_df.copy()
    if isinstance(parcel_to_network, pd.Series):
        mapping = parcel_to_network.to_dict()
    else:
        mapping = parcel_to_network
    df["network"] = df["region"].map(mapping)
    if df["network"].isna().any():
        missing = df.loc[df["network"].isna(), "region"].unique()[:5]
        raise ValueError(f"Missing network labels for parcels like: {missing}")
    metrics = [c for c in df.columns if c.startswith("CRRP_")]
    return df.groupby(["subject_id", "network"], as_index=False)[metrics].mean()


if __name__ == "__main__":
    print("This module defines the CRRP HCP model core. Import it from your batch script.")
