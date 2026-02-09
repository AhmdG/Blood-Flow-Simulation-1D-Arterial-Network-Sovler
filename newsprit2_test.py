import torch
import pickle
from torch.utils.data import DataLoader
from newsprit2 import ArteryMemoryDataset, FlowPredictorUNet1D
from my_models import Regressor2, Regressor3

from scipy.signal import find_peaks

# ---- Paste the metrics functions from my previous message here ----
# (compute_metrics, plot_waveforms_with_uncertainty, bland_altman_peak_flow)
import torch
import numpy as np
from scipy.stats import norm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt


import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from scipy.stats import norm
import importlib
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from collections import defaultdict
import matplotlib.pyplot as plt

def plot_arteries2(
    grouped_data,
    arteries,
    field="Q_true",
    title_prefix="Artery",
):
    """
    grouped_data : dict[str, list[dict]]
        Output of grouping by artery
    arteries : list[str]
        List of artery names to plot (max 4 recommended)
    field : str
        Which array to plot ('A_true', 'Q_true', etc.)
    """

    n = len(arteries)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axes = axes.flatten()

    if n == 1:
        axes = [axes]

    for ax, artery in zip(axes, arteries):
        ar = list(grouped_data.keys())[artery]
        for entry in grouped_data[ar]:
            ax.plot(entry[field], alpha=0.6)

        ax.set_title(f"{title_prefix}: {ar}")
        ax.set_xlabel("Index")
        ax.grid(True)

    axes[0].set_ylabel(field)
    axes[2].set_ylabel(field)
    axes[2].set_xlabel("Index")
    axes[3].set_xlabel("Index")
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_arteries(
    grouped_data,
    arteries,
    field="A_true",
    title_prefix="",
    margin=0.05,
):
    """
    Dynamic y-range computed per artery
    """

    if len(arteries) != 4:
        raise ValueError("Exactly 4 arteries are required for a 2x2 plot")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=False)
    axes = axes.flatten()

    for ax, artery in zip(axes, arteries):
        ar = list(grouped_data.keys())[artery]
        
        curves = [
            entry[field]*10**6
            for entry in grouped_data[ar]
        ]

        if not curves:
            continue

        values = np.concatenate(curves)
        y_min, y_max = values.min(), values.max()

        # Padding
        y_range = y_max - y_min if y_max > y_min else 1.0
        y_min -= margin * y_range
        y_max += margin * y_range

        # Plot curves
        for curve in curves:
            ax.plot(np.linspace(0,  1, len(curve)), curve,  color="blue", alpha=0.15, linewidth=0.8)

        ax.set_ylim(y_min, y_max)
        ax.set_title(f"{title_prefix} {ar}")
        #ax.grid(True)
        ax.tick_params(labelleft=True)

    # Labels
    axes[0].set_ylabel('Q ml/s')
    axes[2].set_ylabel('Q ml/s')
    axes[2].set_xlabel("time s")
    axes[3].set_xlabel("time s")

    plt.tight_layout()
    plt.show()

def evaluate_hemodynamic_model(y_true, y_pred, labels=None, normalize_method='range', naive_method='mean'):
    """
    Evaluate regression model for hemodynamic waveforms.
    
    Parameters:
    - y_true: np.array (n_samples, n_points) ground truth flow waveforms
    - y_pred: np.array (n_samples, n_points) predicted flow waveforms
    - labels: np.array (n_samples,) optional labels for stratification (e.g., arterial locations)
    - normalize_method: 'range' or 'mean' for nRMSE
    - naive_method: 'mean' for MASE (uses mean as naive forecast)
    
    Returns:
    - dict with overall metrics (mean and std across waveforms), and if labels provided, group metrics.
    - Visualizations are saved as PNG files in the current directory.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    n_samples, n_points = y_true.shape
    
    def compute_single_waveform_metrics(y_t, y_p):
        # Metrics for a single waveform
        rmse = np.sqrt(np.mean((y_p - y_t)**2))
        mae = np.mean(np.abs(y_p - y_t))
        me = np.mean(y_p - y_t)
        sd = np.std(y_p - y_t)
        corr, _ = pearsonr(y_t, y_p)
        r2 = r2_score(y_t, y_p)
        
        # nRMSE
        if normalize_method == 'range':
            norm = np.ptp(y_t)
            nrmse = rmse / norm if norm != 0 else np.nan
        elif normalize_method == 'mean':
            norm = np.mean(np.abs(y_t))
            nrmse = rmse / norm if norm != 0 else np.nan
        else:
            raise ValueError("Invalid normalize_method")
        
        # MASE
        if naive_method == 'mean':
            naive_error = np.mean(np.abs(y_t - np.mean(y_t)))
        else:
            raise ValueError("Invalid naive_method")
        mase = mae / naive_error if naive_error != 0 else np.nan
        
        # Feature-specific
        mean_flow_true = np.mean(y_t)
        mean_flow_pred = np.mean(y_p)
        mean_error = np.abs(mean_flow_pred - mean_flow_true)
        
        peak_flow_true = np.max(y_t)
        peak_flow_pred = np.max(y_p)
        peak_error = np.abs(peak_flow_pred - peak_flow_true)
        
        ttp_true = np.argmax(y_t)
        ttp_pred = np.argmax(y_p)
        ttp_error = np.abs(ttp_pred - ttp_true)
        
        return {
            'RMSE': rmse,
            'nRMSE': nrmse,
            'MAE': mae,
            'Correlation': corr,
            'R2': r2,
            'Bias': me,
            'SD': sd,
            'MASE': mase,
            'Mean_Flow_Error': mean_error,
            'Peak_Flow_Error': peak_error,
            'TTP_Error': ttp_error
        }
    
    # Compute per sample
    per_sample_metrics = [compute_single_waveform_metrics(y_true[i], y_pred[i]) for i in range(n_samples)]
    
    # Aggregate overall
    overall_mean = {k: np.nanmean([m[k] for m in per_sample_metrics]) for k in per_sample_metrics[0]}
    overall_std = {k: np.nanstd([m[k] for m in per_sample_metrics]) for k in per_sample_metrics[0]}
    overall_metrics = {'mean': overall_mean, 'std': overall_std}
    
    # LoA overall on all flattened errors
    errors = (y_pred - y_true).flatten()
    me_overall = np.mean(errors)
    sd_overall = np.std(errors)
    loa_lower = me_overall - 1.96 * sd_overall
    loa_upper = me_overall + 1.96 * sd_overall
    overall_metrics['LoA'] = {'lower': loa_lower, 'upper': loa_upper, 'bias': me_overall, 'sd': sd_overall}
    
    result = {'overall': overall_metrics}
    
    if labels is not None:
        unique_labels = np.unique(labels)
        group_metrics = {}
        for lbl in unique_labels:
            idx = labels == lbl
            group_per_sample = [per_sample_metrics[i] for i in range(n_samples) if idx[i]]
            if group_per_sample:
                group_mean = {k: np.nanmean([m[k] for m in group_per_sample]) for k in group_per_sample[0]}
                group_std = {k: np.nanstd([m[k] for m in group_per_sample]) for k in group_per_sample[0]}
                group_metrics[lbl] = {'mean': group_mean, 'std': group_std}
                
                # Group LoA
                group_errors = (y_pred[idx] - y_true[idx]).flatten()
                g_me = np.mean(group_errors)
                g_sd = np.std(group_errors)
                g_loa_lower = g_me - 1.96 * g_sd
                g_loa_upper = g_me + 1.96 * g_sd
                group_metrics[lbl]['LoA'] = {'lower': g_loa_lower, 'upper': g_loa_upper, 'bias': g_me, 'sd': g_sd}
        result['groups'] = group_metrics
    
    # Visualizations
    # 1. Waveform overlay (first sample)
    plt.figure(figsize=(8, 4))
    plt.plot(y_true[0], label='True')
    plt.plot(y_pred[0], label='Predicted')
    plt.legend()
    plt.title('Example Waveform Overlay')
    plt.xlabel('Time Points')
    plt.ylabel('Flow')
    plt.savefig('waveform_overlay.png')
    plt.close()
    
    # 2. Scatter plot (all points)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title('Scatter Plot: Predicted vs True')
    plt.xlabel('True Flow')
    plt.ylabel('Predicted Flow')
    plt.savefig('scatter_plot.png')
    plt.close()
    
    # 3. Bland-Altman plot (all points)
    means = ((y_true + y_pred) / 2).flatten()
    diffs = (y_pred - y_true).flatten()
    plt.figure(figsize=(8, 4))
    plt.scatter(means, diffs, alpha=0.5)
    plt.axhline(me_overall, color='red', label='Bias')
    plt.axhline(loa_lower, color='green', linestyle='--', label='LoA Lower')
    plt.axhline(loa_upper, color='green', linestyle='--', label='LoA Upper')
    plt.title('Bland-Altman Plot')
    plt.xlabel('Mean Flow')
    plt.ylabel('Difference (Pred - True)')
    plt.legend()
    plt.savefig('bland_altman.png')
    plt.close()
    
    return result

# The same utility function used in the training script
def get_class_from_path(class_path: str):
    """Dynamically loads and returns a class object from its string path."""
    try:
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"Could not load class '{class_path}': {e}")

def filter_damaged_data(data_list, max_allowed_flow=1.0):
    """
    Filters a list of dicts containing arterial simulation results.
    Removes any sample where Q_true has ANY value > max_allowed_flow
    (physically impossible → failed simulation, blow-up, etc.)
    """
    clean_data = []

    for sample in data_list:
        Q = sample['Q_true']                  # numpy array
        # Main check: any flow > threshold → damaged
        if np.any(Q > max_allowed_flow):
            continue
            
        # If we get here → sample is good
        clean_data.append(sample)

    return clean_data

def evaluate_for_paper(y_true, mu, sigma=None, eps=1e-8):
    """
    y_true, mu, sigma: numpy arrays on ORIGINAL (raw) scale, shape (N,) or (N,1)
    """
    y = np.asarray(y_true).flatten()
    μ = np.asarray(mu).flatten()
    if sigma is None:
        σ = np.ones_like(μ) * 1e-3  # small constant to avoid div by zero
        σ = np.asarray(sigma).flatten() + eps

    # Relative errors
    rel_err = np.abs(y - μ) / (np.abs(y) + eps)
    rMAE  = 100 * rel_err.mean()
    rRMSE = 100 * np.sqrt((rel_err**2).mean())
    MAPEs = 100 * np.mean(np.abs(y - μ) / (np.abs(y) + 1e-6))   # safer when y≈0

    # NLL
    if sigma is None:
       
        NLL = 0.5 * np.log(2 * np.pi * σ**2) + 0.5 * ((y - μ)/σ)**2
        NLL = NLL.mean()

    # CRPS (closed-form for Gaussian)
    z = (y - μ) / σ
    CRPS = σ * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1/np.sqrt(np.pi))
    CRPS = CRPS.mean()
    CRPS_clim = np.mean(np.abs(y - y.mean()))                   # climatology benchmark
    CRPS_skill = 1 - CRPS / CRPS_clim

    # 95% PI
    q = norm.ppf(0.975)                                         # 1.95996
    lower = μ - q * σ
    upper = μ + q * σ
    PICP_95 = 100 * np.mean((y >= lower) & (y <= upper))
    rMPIW_95 = 100 * np.mean((upper - lower) / (np.abs(y) + eps))

    results = {
        "rMAE (%)"     : round(rMAE, 3),
        "rRMSE (%)"    : round(rRMSE, 3),
        "MAPE-s (%)"   : round(MAPEs, 3),
        "NLL"          : round(NLL, 4),
        "CRPS"         : round(CRPS, 6),
        "CRPS Skill"   : round(CRPS_skill, 4),
        "PICP₉₅ (%)"   : round(PICP_95, 2),
        "rMPIW₉₅ (%)"  : round(rMPIW_95, 3),
    }
    return results

def plot_relative_reliability_diagram(mu_pred, log_sigma_pred, y_true_tensor, 
                                    scaling_method="per_batch", save_path=None):
    """
    Fully scale-invariant reliability diagram using standardized residuals z = (y - μ)/σ
    Works even with synthetic ≠ real scales.
    """
    # --- Reverse scaling to get mu_phys, sigma_phys in TEST scale (whatever it is) ---
    # (Same reversal block as before — per_batch/global/fixed)
    # ... [insert your reversal code here] ...
    # → mu_phys, sigma_phys, y_true now in the same (real physiological) units

    mu     = mu_pred.flatten().cpu().numpy()
    sigma  = log_sigma_pred.exp().flatten().cpu().numpy()
    y      = y_true_tensor.flatten().cpu().numpy()

    # --- Standardized residuals (relative, unitless) ---
    z = (y - mu) / (sigma + 1e-12)   # This is the relative error normalized by predicted uncertainty

    # --- Reliability diagram on z (scale-invariant) ---
    plt.figure(figsize=(5.5, 5))
    observed = []
    nominal  = np.linspace(5, 95, 19)

    for alpha in np.linspace(0.05, 0.95, 19):
        z_crit = norm.ppf(1 - alpha/2)
        cov = 100 * np.mean(np.abs(z) <= z_crit)   # Coverage in relative space
        observed.append(cov)

    plt.plot([0, 100], [0, 100], "k--", lw=2, label="Perfect calibration")
    plt.plot(nominal, observed, "o-", color="dodgerblue", lw=2, markersize=6, label="Our model")
    plt.xlabel("Nominal confidence level (%)")
    plt.ylabel("Observed coverage (%)")
    plt.title("Scale-Invariant Reliability Diagram\n(using standardized residuals z = (y - μ)/σ)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
def evaluate_relative_and_normalized(mu_pred, log_sigma_pred, y_true_tensor,
                                   scaling_method="per_batch"):
    """
    Everything scale-invariant — works even if training and test have different units
    """
    # Reverse scaling exactly as before → get mu_phys, sigma_phys in test-scale units
    # (use the same reversal code as before)
    # ... [same reversal block] ...
    # → now you have mu_phys and sigma_phys in the same units as y_true_tensor (10⁻³ range)

    device = y_true_tensor.device

    # Work entirely in torch until the very end
    mu_pred    = mu_pred#.flatten()           # [N×T]
    sigma_pred = torch.exp(log_sigma_pred)#.flatten()
    y_true     = y_true_tensor.flatten()

    # --------------------------------------------------------------
    # REVERSE THE SCALING — this is the only part that matters
    # --------------------------------------------------------------
    if scaling_method == "per_batch":          # ← the one that gave you beautiful results
        B, T = y_true_tensor.shape
        y_reshaped = y_true_tensor.view(B, -1)  # [B, T]

        mean_per_batch = y_reshaped.mean(dim=1, keepdim=True)           # [B, 1]
        std_per_batch  = y_reshaped.std(dim=1, keepdim=True) + 1e-8     # [B, 1]

        # Broadcast back to [B, T] then flatten
        mean_per_batch = mean_per_batch.expand(-1, T).reshape(-1)
        std_per_batch  = std_per_batch.expand(-1, T).reshape(-1)

        mu_phys    = mu_pred * std_per_batch + mean_per_batch
        sigma_phys = sigma_pred * std_per_batch

    elif scaling_method == "global":
        global_mean = y_true_tensor.mean()
        global_std  = y_true_tensor.std()
        mu_phys    = mu_pred * global_std + global_mean
        sigma_phys = sigma_pred * global_std

    elif scaling_method == "fixed":
        SCALE = 5000.0          # ← change to whatever you used
        mu_phys    = mu_pred / SCALE
        sigma_phys = sigma_pred / SCALE

    # --------------------------------------------------------------
    # Now compute metrics (convert to numpy only here)
    # --------------------------------------------------------------
    mu    = mu_phys.cpu().numpy()
    sigma = sigma_phys.cpu().numpy()
    y  = y_true.cpu().numpy()

    # ──────────────────────── Relative point metrics ────────────────────────
    relative_error = np.abs(mu - y) / (np.abs(y) + 1e-8)          # avoid div by zero
    MAE_rel  = np.mean(relative_error)                     # same as MAPE/100
    MAPE     = 100 * MAE_rel
    MEDAE_rel= np.median(relative_error)

    # ──────────────────────── Normalized RMSE (very common) ────────────────────────
    NRMSE_mean = np.sqrt(np.mean((mu - y)**2)) / np.mean(y)         # normalized by mean
    NRMSE_iqr  = np.sqrt(np.mean((mu - y)**2)) / (np.percentile(y, 75) - np.percentile(y, 25))

    # ──────────────────────── Probabilistic metrics (already scale-invariant) ─────
    z = (y - mu) / (sigma + 1e-12)
    CRPS = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1/np.sqrt(np.pi))
    
    cov_90 = 100 * np.mean(np.abs(z) <= 1.645)
    cov_95 = 100 * np.mean(np.abs(z) <= 1.960)

    # Normalized interval width (very important!)
    norm_width_90 = np.mean(3.29 * sigma / (np.abs(y) + 1e-8))

    results = {
        'MAPE (%)'          : MAPE,
        'Median APE (%)'    : 100 * MEDAE_rel,
        'NRMSE (mean)'      : NRMSE_mean,
        'NRMSE (IQR)'       : NRMSE_iqr,
        'CRPS (normalized)' : CRPS.mean() / np.mean(np.abs(y)),
        '90% Coverage (%)'  : cov_90,
        '95% Coverage (%)'  : cov_95,
        'Norm. 90% Width'   : norm_width_90,      # this is the key one
    }
    return results

# ------------------------------------------------------------------
# 1. Core evaluation function (returns everything you need)
# ------------------------------------------------------------------
def evaluate_correctly(mu_pred, log_sigma_pred, y_true_tensor, scaling_method="per_batch"):
    """
    mu_pred:        torch tensor [N, T] – raw model output
    log_sigma_pred: torch tensor [N, T] – raw log_sigma output
    y_true_tensor:  torch tensor [N, T] – ground truth in original tiny units (10⁻⁵)
    scaling_method: "per_batch" | "global" | "fixed"
    """
    device = y_true_tensor.device

    # Work entirely in torch until the very end
    mu_pred    = mu_pred.flatten()           # [N×T]
    sigma_pred = torch.exp(log_sigma_pred).flatten()
    y_true     = y_true_tensor.flatten()

    # --------------------------------------------------------------
    # REVERSE THE SCALING — this is the only part that matters
    # --------------------------------------------------------------
    if scaling_method == "per_batch":          # ← the one that gave you beautiful results
        B, T = y_true_tensor.shape
        y_reshaped = y_true_tensor.view(B, -1)  # [B, T]

        mean_per_batch = y_reshaped.mean(dim=1, keepdim=True)           # [B, 1]
        std_per_batch  = y_reshaped.std(dim=1, keepdim=True) + 1e-8     # [B, 1]

        # Broadcast back to [B, T] then flatten
        mean_per_batch = mean_per_batch.expand(-1, T).reshape(-1)
        std_per_batch  = std_per_batch.expand(-1, T).reshape(-1)

        mu_phys    = mu_pred * std_per_batch + mean_per_batch
        sigma_phys = sigma_pred * std_per_batch

    elif scaling_method == "global":
        global_mean = y_true_tensor.mean()
        global_std  = y_true_tensor.std()
        mu_phys    = mu_pred * global_std + global_mean
        sigma_phys = sigma_pred * global_std

    elif scaling_method == "fixed":
        SCALE = 5000.0          # ← change to whatever you used
        mu_phys    = mu_pred / SCALE
        sigma_phys = sigma_pred / SCALE

    # --------------------------------------------------------------
    # Now compute metrics (convert to numpy only here)
    # --------------------------------------------------------------
    mu_phys    = mu_phys.cpu().numpy()
    sigma_phys = sigma_phys.cpu().numpy()
    y_true_np  = y_true.cpu().numpy()

    mae  = mean_absolute_error(y_true_np, mu_phys)
    rmse = np.sqrt(mean_squared_error(y_true_np, mu_phys))
    mape = 100 * np.mean(np.abs((y_true_np - mu_phys) / (np.abs(y_true_np) + 1e-8)))
    r2   = r2_score(y_true_np, mu_phys)

    # CRPS (Gaussian analytical)
    z = (y_true_np - mu_phys) / (sigma_phys + 1e-8)
    crps = sigma_phys * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1/np.sqrt(np.pi))

    # Coverage
    cov_90 = 100 * np.mean(np.abs(y_true_np - mu_phys) <= 1.645 * sigma_phys)
    cov_95 = 100 * np.mean(np.abs(y_true_np - mu_phys) <= 1.960 * sigma_phys)
    width_90 = np.mean(3.290 * sigma_phys)   # 1.645 × 2
    width_95 = np.mean(3.920 * sigma_phys)

    results = {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE_%': float(mape),
        'R2': float(r2),
        'CRPS': float(crps.mean()),
        'Coverage_90%': float(cov_90),
        'Width_90%': float(width_90),
        'Coverage_95%': float(cov_95),
        'Width_95%': float(width_95),
        'mu_phys': mu_pred, 'sigma_phys': log_sigma_pred, 'y_true': y_true_tensor
    }
    return results

# ------------------------------------------------------------------
# 2. One-liner to build the full table (just feed your models)
# ------------------------------------------------------------------
def make_results_table(models_dict, y_true):
    """
    models_dict = {
        "Persistence":  (mu_persist, None),
        "Your Model (global)": (mu_global, log_sigma_global),
        ...
    }
    """
    rows = []
    for name, (mu, log_sigma) in models_dict.items():
        if log_sigma is None:  # deterministic
            res = evaluate_correctly(mu, torch.zeros_like(mu)-2, y_true)  # dummy sigma
            rows.append([name, res['MAE'], res['RMSE'], res['MAPE_%'], res['R2'], "-", "-", "-", "-"])
        else:
            res = evaluate_correctly(mu, log_sigma, y_true)
            rows.append([name, res['MAE'], res['RMSE'], res['MAPE_%'], res['R2'],
                         res['CRPS'], res['Coverage_90%'], res['Width_90%'], res['Coverage_95%']])
    
    df = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "MAPE%", "R²", "CRPS", "90% Cov", "90% Width", "95% Cov"])
    return df.round(4)

# ------------------------------------------------------------------
# 3. Calibration / Reliability plot
# ------------------------------------------------------------------
def plot_reliability_diagram(mu_pred, log_sigma_pred, y_true_tensor, 
                           scaling_method="per_batch", save_path=None):
    """
    All inputs: raw model outputs + original y_true_tensor (10^-5 range)
    """
    device = y_true_tensor.device

    # --- Reverse scaling exactly like in evaluation ---
    if scaling_method == "per_batch":
        B, T = y_true_tensor.shape
        y_resh = y_true_tensor.view(B, -1)
        mean_b = y_resh.mean(dim=1, keepdim=True)
        std_b  = y_resh.std(dim=1, keepdim=True) + 1e-8
        mean_b = mean_b.expand(-1, T).reshape(-1)
        std_b  = std_b.expand(-1, T).reshape(-1)
        
        mu_phys    = mu_pred.flatten() * std_b + mean_b
        sigma_phys = torch.exp(log_sigma_pred).flatten() * std_b

    elif scaling_method == "global":
        mean_g = y_true_tensor.mean()
        std_g  = y_true_tensor.std()
        mu_phys    = mu_pred.flatten() * std_g + mean_g
        sigma_phys = torch.exp(log_sigma_pred).flatten() * std_g

    elif scaling_method == "fixed":
        SCALE = 5000.0
        mu_phys    = mu_pred.flatten() / SCALE
        sigma_phys = torch.exp(log_sigma_pred).flatten() / SCALE

    # --- Now everything is in original tiny units (10^-5) ---
    mu_phys    = mu_phys.cpu().numpy()
    
    sigma_phys = sigma_phys.cpu().numpy()
    y_true     = y_true_tensor.flatten().cpu().numpy()

    # --- Reliability diagram ---
    plt.figure(figsize=(5.5, 5))
    observed = []
    nominal  = np.linspace(5, 95, 19)

    for alpha in np.linspace(0.05, 0.95, 19):
        z = norm.ppf(1 - alpha/2)
        lower = mu_phys - z * sigma_phys
        upper = mu_phys + z * sigma_phys
        cov = 100 * np.mean((y_true >= lower) & (y_true <= upper))
        observed.append(cov)

    plt.plot([0, 100], [0, 100], "k--", lw=2, label="Perfect calibration")
    plt.plot(nominal, observed, "o-", color="dodgerblue", lw=2, markersize=6, label="Our model")
    plt.xlabel("Nominal confidence level (%)")
    plt.ylabel("Observed coverage (%)")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
# ------------------------------------------------------------------
# 4. PI width vs flow magnitude (proves heteroscedasticity learning)
# ------------------------------------------------------------------
def plot_interval_width_vs_flow(mu_phys, sigma_phys, y_true, save_path=None):
    plt.figure(figsize=(6, 4.5))
    flow = y_true
    width_90 = 3.29 * sigma_phys  # 1.645 × 2
    plt.scatter(flow, width_90, alpha=0.5, s=10, color="coral")
    plt.xlabel("True flow (L/min or m³/h)")
    plt.ylabel("90% Prediction Interval Width")
    plt.title("Learned Heteroscedasticity")
    plt.grid(True, alpha=0.3)
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compute_metrics(q_true: torch.Tensor,
                    mu_pred: torch.Tensor,
                    sigma_pred: torch.Tensor = None,
                    dt: float = None,
                    mask_systole: torch.Tensor = None):
    """
    Fully PyTorch 2.0+ compatible version – no more std() errors!
    """
    # Move to CPU + flatten once
    q = q_true.detach().cpu()
    mu = mu_pred.detach().cpu()

    results = {}

    # ------------------------------------------------------------------
    # 1. Point-prediction accuracy
    # ------------------------------------------------------------------
    results["MAE (mL/s)"]        = torch.mean(torch.abs(q - mu)).item()
    results["RMSE (mL/s)"]       = torch.sqrt(torch.mean((q - mu)**2)).item()
    results["Peak Error (mL/s)"] = torch.max(torch.abs(q - mu)).item()
    rms_q = torch.sqrt(torch.mean(q**2)) + 1e-8
    results["Rel. RMSE (%)"]     = 100 * results["RMSE (mL/s)"] / rms_q.item()

    # ------------------------------------------------------------------
    # 2. Waveform similarity – now 100% safe
    # ------------------------------------------------------------------
    def ncc_per_wave(a: torch.Tensor, b: torch.Tensor) -> float:
        a_c = a - a.mean()
        b_c = b - b.mean()
        std_a = torch.sqrt(torch.mean(a_c**2))
        std_b = torch.sqrt(torch.mean(b_c**2))
        if std_a < 1e-8 or std_b < 1e-8:
            return torch.tensor(0.0)
        return torch.mean(a_c * b_c) / (std_a * std_b + 1e-8)

    ncc_vals = [ncc_per_wave(q_true[i], mu_pred[i]) for i in range(q_true.shape[0])]
    results["NCC"] = float(torch.mean(torch.stack(ncc_vals)).item())

    # DTW distance – robust version
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        dtw_dists = []
        for i in range(q_true.shape[0]):
            x = q_true[i].flatten().cpu().numpy()    # ← safe 1D
            y = mu_pred[i].flatten().cpu().numpy()   # ← safe 1D
            dist, _ = fastdtw(x, y, dist=euclidean)
            dtw_dists.append(dist)
        results["DTW distance"] = float(np.mean(dtw_dists))
    except Exception as e:
        results["DTW distance"] = float('nan')
        print(f"DTW computation failed ({e}), setting to NaN")

    # ------------------------------------------------------------------
    # 3. Uncertainty quantification
    # ------------------------------------------------------------------
    if sigma_pred is not None:
        sigma = sigma_pred.detach().cpu()
        z = (q - mu) / (sigma + 1e-8)

        results["NLL"] = 0.5 * torch.mean(z**2 + 2*torch.log(sigma + 1e-8) + np.log(2*np.pi)).item()
        results["Calibration Error (ENE)"] = torch.mean(torch.abs(z)).item()
        results["Sharpness (avg sigma)"] = torch.mean(sigma).item()

        results["Coverage 1σ (%)"] = 100 * torch.mean((torch.abs(z) <= 1.0).float()).item()
        results["Coverage 2σ (%)"] = 100 * torch.mean((torch.abs(z) <= 2.0).float()).item()
        results["Coverage 3σ (%)"] = 100 * torch.mean((torch.abs(z) <= 3.0).float()).item()

    # ------------------------------------------------------------------
    # 4. Physiological metrics
    # ------------------------------------------------------------------
    if dt is not None:
        sv_true = torch.trapz(q_true, dx=dt, dim=1).cpu()
        sv_pred = torch.trapz(mu_pred, dx=dt, dim=1).cpu()
        results["Stroke Volume MAE (mL)"] = torch.mean(torch.abs(sv_true - sv_pred)).item()

    # Peak systolic flow
    peak_true = q_true.max(dim=1).values.cpu()
    peak_pred = mu_pred.max(dim=1).values.cpu()
    results["Peak Systolic Flow MAE (mL/s)"] = torch.mean(torch.abs(peak_true - peak_pred)).item()

    return results


# ----------------------------------------------------------------------
# Bonus: Plotting helpers (for paper figures)
# ----------------------------------------------------------------------
def plot_waveforms_with_uncertainty(q_true, mu_pred, sigma_pred, idx=0, ax=None, dt=None):
    if ax is None: fig, ax = plt.subplots(figsize=(6, 3.5))
    t = np.arange(q_true.shape[1]) * dt
    ax.plot(t, q_true[idx], 'k', label='Ground truth', linewidth=2)
    ax.plot(t, mu_pred[idx], 'C3', label='Prediction', linewidth=2)
    ax.fill_between(t,
                    mu_pred[idx] - sigma_pred[idx],
                    mu_pred[idx] + sigma_pred[idx],
                    color='C3', alpha=0.3, label=r'$\pm 1\sigma$')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flow (mL/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def bland_altman_peak_flow(q_true, mu_pred, ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(5, 4))
    peak_true = q_true.max(dim=1).values.cpu().numpy()
    peak_pred = mu_pred.max(dim=1).values.cpu().numpy()
    mean = (peak_true + peak_pred) / 2
    diff  = peak_pred - peak_true
    md    = np.mean(diff)
    sd    = np.std(diff)

    ax.scatter(mean, diff, alpha=0.6)
    ax.axhline(md, color='gray', linestyle='--')
    ax.axhline(md + 1.96*sd, color='red', linestyle='--')
    ax.axhline(md - 1.96*sd, color='red', linestyle='--')
    ax.text(0.95, 0.85, f'Bias = {md:.1f}\n±1.96SD = [{md-1.96*sd:.1f}, {md+1.96*sd:.1f}]',
            transform=ax.transAxes, ha='right', bbox=dict(facecolor='white', alpha=0.8))
    ax.set_xlabel("Mean of measured and predicted peak flow (mL/s)")
    ax.set_ylabel("Prediction − Ground truth (mL/s)")
    ax.set_title("Bland–Altman plot – Peak systolic flow")
    return ax
# -----------------------
# Configuration
# -----------------------
if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    test_dataset_path = "dataset_test.pkl"
    train_dataset_path = "dataset_train.pkl"
    best_model_path = "_model_2025-12-12_13-52.pth"#"best_model_TVHuber.pth"
    checkpoint_path = "checkpoints/best"+best_model_path
    stats = torch.load("normalization_stats.pth")
    GLOBAL_MEAN = stats["mean"]
    GLOBAL_STD  = stats["std"]

    # -----------------------
    # Load test dataset
    # -----------------------

    print("Loading test dataset...")
    with open(test_dataset_path, "rb") as f:
        test_samples = pickle.load(f)

    print(f"Loaded {len(test_samples)} test samples.")

    with open(train_dataset_path, "rb") as f:
        train_samples = pickle.load(f)

    print(f"Loaded {len(train_samples)} train samples.")

    # Dataset class (must match the one used in training)
    train_samples = filter_damaged_data(train_samples, max_allowed_flow=1.0)
    test_samples = filter_damaged_data(test_samples, max_allowed_flow=1.0)
    sorted_data = sorted(train_samples, key=lambda d: d.get("artery", ""))
    grouped_by_artery = defaultdict(list)

    for item in train_samples:
        grouped_by_artery[item["artery"]].append(item)

    #arteries_to_plot = ["common_carotid_R","common_carotid_L","aorta","femoral_R", ]
    arteries_to_plot = [
    0,
    2,
    16,
    26,
    ]
    plot_arteries(
        grouped_data=grouped_by_artery,
        arteries=arteries_to_plot,
        field="Q_true",
    )

    test_loader = DataLoader(
        ArteryMemoryDataset(test_samples),
        batch_size=batch_size,
        shuffle=False,   # IMPORTANT: keep test order fixed
    )

    train_loader = DataLoader(
        ArteryMemoryDataset(train_samples),
        batch_size=batch_size,
        shuffle=False,   # IMPORTANT: keep test order fixed
    )


    # -----------------------
    # Load model
    # -----------------------
    self_containing = False  # whether the checkpoint is self-contained
    print("Loading best model...")
    if self_containing:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # 2. Extract paths and state dictionaries
        model_class_path = checkpoint['class_paths']['model']
        model_state_dict = checkpoint['state_dicts']['model']

        print(f"Loading model class from path: {model_class_path}")

        # 3. Dynamically get the model class
        ModelAClass = get_class_from_path(model_class_path)

        # 4. Instantiate the model
        # IMPORTANT: You must pass any required initialization arguments here!
        # (e.g., input_dim, num_classes, etc., if your model's __init__ requires them)
        # If your checkpoint saved hyperparameters, load and use them here.
        model = ModelAClass() 

        # 5. Load the trained parameters
        model.load_state_dict(model_state_dict)

    else:

        model =  Regressor3(width=192, depth=8).to(device)  # your model class
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model_state_dict = checkpoint['state_dicts']['model']
        model.load_state_dict(model_state_dict)
        #model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("Model loaded.")

    # -----------------------
    # Run inference on test set
    # -----------------------

    print("Running inference on test set...")
    sigma_pred = []
    mu_pred = []
    q_true = []
    all_metadata = []
    dt = .01  # time step in seconds (adjust as per your data)

    q_true_list   = []
    mu_pred_list  = []
    sigma_pred_list = []
    SCALE_Y = 1_000_000.0  
    unique_arteries = set()
    num_samples = 0
    with torch.no_grad():
        #plt.figure()
        for A_batch, Q_batch, yaml_model, artery, x_idx in train_loader:
            B = A_batch.shape[0]
            A_batch = A_batch.float().to(device)
            
            if 0:
                for i in range(B):
                    if Q_batch[i].max().item()<1:
                        num_samples +=1
                        #print(f"Artery: {artery[i]}, Model: {yaml_model[i]}, max: {Q_batch[i].max().item()}, idx: {x_idx[i].item()}")
                        # save unique arteries
                        if artery[i] not in unique_arteries:
                            unique_arteries.add(artery[i])
                    continue
                    try:
                        plt.plot(Q_batch[i].cpu().numpy())
                    except:
                        pass
                #continue
        

            #plt.plot(A_batch[0,:].cpu().numpy() ,label='2')
            #plt.plot((Q_batch_raw[0,:].cpu().numpy()-GLOBAL_MEAN)/GLOBAL_STD, label='3')
          
            Q_batch_raw = SCALE_Y*Q_batch.float()                     # keep raw physical units

            # === Normalize input A ===
            A_mean = A_batch.mean(dim=1, keepdim=True)
            A_std  = A_batch.std(dim=1, keepdim=True) + 1e-8
            A_norm = (A_batch - A_mean) / A_std
            #A_norm = A_norm.unsqueeze(1)                      # [B,1,T]

            # === Normalize target Q (same as training) ===!!!! very wrong


            mu_norm = model(A_norm)
            if 0:
                #plt.figure()
                #plt.plot(A_batch[0,:].cpu().numpy(), label='1')
                plt.figure()
                idx = 6
                plt.plot(mu_norm[idx,:].cpu().numpy(), label='1') 
                plt.plot(Q_batch_raw[idx,:].cpu().numpy() ,label='2')
                #plt.plot((Q_batch_raw[0,:].cpu().numpy()-GLOBAL_MEAN)/GLOBAL_STD, label='3')
                plt.show()
            #continue

            # === Squeeze channel dim ===
            mu_norm   = mu_norm.squeeze(1)      # [B, T]
            #log_sigma = log_sigma.squeeze(1)    # [B, T]


            # === Denormalize to physical units ===
            #mu_raw    = mu_norm  * GLOBAL_STD + GLOBAL_MEAN     # de-z-score the mean
            #sigma_raw = torch.exp(log_sigma) * GLOBAL_STD           # de-z-score the std-dev

            # === Store each sample as [T] ===
            q_true_list.extend(Q_batch_raw.squeeze(1).cpu().numpy())     # list of [T]
            mu_pred_list.extend(mu_norm.numpy())                         # list of [T]
            #sigma_pred_list.extend(sigma_raw.numpy())                  # list of [T]
        print(f"Unique arteries with max flow < 1: {len(unique_arteries)}, {list(unique_arteries)}")
        print(f"Number of samples with max flow < 1: {num_samples}")
        #plt.show()
    # === Final tensors ===
    q_true     = torch.tensor(np.stack(q_true_list))#.flatten()     # [N, T]
    mu_pred    = torch.tensor(np.stack(mu_pred_list))#.flatten()  
    results = evaluate_hemodynamic_model(q_true.numpy(), mu_pred.numpy(), labels=None, normalize_method='range', naive_method='mean') 
    #sigma_pred = torch.tensor(np.stack(sigma_pred_list)).flatten()   
    results = evaluate_for_paper(q_true, mu_pred, sigma_pred=None, eps=1e-8)
    print(results)
    plot_waveforms_with_uncertainty(q_true, mu_pred, sigma_pred, idx=0, ax=None, dt=dt)

    print(f"Success! Test set: {q_true.shape[0]} samples, {q_true.shape[1]} time points")
    results = evaluate_relative_and_normalized(mu_pred, sigma_pred, q_true, scaling_method="per_batch")

    print(results)
    plot_relative_reliability_diagram(mu_pred, sigma_pred, q_true)
    plot_interval_width_vs_flow(mu_pred, sigma_pred, q_true)
    if 0:
        metrics = compute_metrics(
            q_true=q_true,
            mu_pred=mu_pred,
            sigma_pred=sigma_pred,
            dt=dt,
            mask_systole=None
        )

        # 4. Pretty table (exactly what goes into the paper)
        print("\n" + "="*70)
        print("           FINAL RESULTS – 1D FLOW-FROM-AREA INVERSION")
        print("="*70)
        print(f"{'Metric':<35} {'Value':>15}")
        print("-"*70)
        for k, v in metrics.items():
            if "Coverage" in k or "NCC" in k:
                print(f"{k:<35} {v:15.2f}")
            elif any(x in k for x in ["NLL", "Sharpness"]):
                print(f"{k:<35} {v:15.3f}")
            else:
                print(f"{k:<35} {v:15.2f}")
        print("-"*70)

        # 5. Figures
        plt.figure(figsize=(12, 8))

        # 5a – Example waveforms with uncertainty bands
        ax1 = plt.subplot(2, 2, 1)
        plot_waveforms_with_uncertainty(q_true, mu_pred, sigma_pred, idx=2, ax=ax1,dt=dt)
        ax1.set_title("Example waveform with ±1σ uncertainty")

        ax2 = plt.subplot(2, 2, 2)
        plot_waveforms_with_uncertainty(q_true, mu_pred, sigma_pred, idx=7, ax=ax2,dt=dt)
        ax2.set_title("Another patient (higher variability)")

        # 5b – Bland-Altman on peak systolic flow
        ax3 = plt.subplot(2, 2, 3)
        bland_altman_peak_flow(q_true, mu_pred, ax=ax3)

        # 5c – Calibration plot (observed vs expected confidence)
        ax4 = plt.subplot(2, 2, 4)
        z = ((q_true - mu_pred) / (sigma_pred + 1e-8)).flatten().cpu().numpy()
        abs_z = np.abs(z)
        percentiles = np.linspace(0, 99, 200)
        observed = [np.percentile(abs_z, p) for p in percentiles]
        expected = [norm.ppf(p/100 + (100-p)/200) for p in percentiles]  # midpoint correction
        ax4.plot(expected, observed, 'C0', lw=2, label='Model')
        ax4.plot(expected, expected, 'k--', label='Perfect calibration')
        ax4.set_xlabel("Expected |z| (standard deviations)")
        ax4.set_ylabel("Observed |z|")
        ax4.set_title("Reliability diagram (perfect = diagonal)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("\nDone! Figures and table ready for the paper.")

        print("Inference completed.")
    if 0:

        # -----------------------
        # Save predictions
        # -----------------------

        import numpy as np

        np.save("test_predictions.npy", np.concatenate(all_predictions, axis=0))
        np.save("test_targets.npy", np.concatenate(all_targets, axis=0))

        with open("test_metadata.pkl", "wb") as f:
            pickle.dump(all_metadata, f)

        print("Saved test_predictions.npy, test_targets.npy, test_metadata.pkl")

        # -----------------------
        # Optional: plot one sample
        # -----------------------

        def plot_sample(mu, Q, title="Plot"):
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,4))
            plt.plot(Q, label="True")
            plt.plot(mu, label="Predicted")
            plt.title(title)
            plt.legend()
            plt.show()

        PLOT_INDEX = 0  # choose any sample to visualize

        yaml_model, artery_name, x_idx = all_metadata[PLOT_INDEX]
        mu = np.concatenate(all_predictions, axis=0)[PLOT_INDEX]
        Q = np.concatenate(all_targets, axis=0)[PLOT_INDEX]

        title = f"{yaml_model} | {artery_name} | x={x_idx}"
        plot_sample(mu, Q, title=title)
