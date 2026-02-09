# ===================================================================
#   FULL DEMO: Dummy 1D Hemodynamic Data + All Metrics + Figures
# ===================================================================
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ---- Paste the metrics functions from my previous message here ----
# (compute_metrics, plot_waveforms_with_uncertainty, bland_altman_peak_flow)
import torch
import numpy as np
from scipy.stats import norm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import torch.nn as nn
from newsprit3 import Regressor3Hetero

def heteroscedastic_nll(y_true, mu, log_var):
    return torch.mean(
        0.5 * torch.exp(-log_var) * (y_true - mu) ** 2
        + 0.5 * log_var
    )

def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_inference(
    model,
    dataloader,
    n_mc=20,
    device="cpu",
    SCALE_Y = 1_000_000.0,
):
    model.to(device)

    all_means = []
    all_stds = []
    q_true_list = []

    with torch.no_grad():
        for A_batch, Q_batch, yaml_model, artery, x_idx in dataloader:
            B = A_batch.shape[0]
            A_batch = A_batch.float().to(device)
            Q_batch_raw = SCALE_Y*Q_batch.float()                     # keep raw physical units
            q_true_list.extend(Q_batch_raw.squeeze(1).cpu().numpy())  

            # === Normalize input A ===
            A_mean = A_batch.mean(dim=1, keepdim=True)
            A_std  = A_batch.std(dim=1, keepdim=True) + 1e-8
            A_norm = (A_batch - A_mean) / A_std
            #x_batch = x_batch.to(device)

            mc_preds = []

            for _ in range(n_mc):
                y_hat = model(A_norm)  # (B, 100)
                mc_preds.append(y_hat.unsqueeze(0))

            mc_preds = torch.cat(mc_preds, dim=0)  # (n_mc, B, 100)

            mean = mc_preds.mean(dim=0)
            std = mc_preds.std(dim=0)
            print(std.max())

            all_means.append(mean.cpu())
            all_stds.append(std.cpu())

    return (
        torch.cat(all_means, dim=0),#.numpy(),   # (8000, 100)
        torch.cat(all_stds, dim=0),#.numpy(),
        torch.tensor(np.stack(q_true_list))#.flatten()     # [N, T]
    )

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
            return 0.0
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

            
            dist, _ = fastdtw(x, y)#, dist=euclidean)
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

# ===================================================================
#   MAIN – Run everything and show results
# ===================================================================
def main():
    import pickle
    import torch
    from torch.utils.data import DataLoader
    from newsprit2_test import ArteryMemoryDataset, filter_damaged_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    test_dataset_path = "dataset_test.pkl"
    train_dataset_path = "dataset_train.pkl"
    best_model_path = "_model_2025-12-15_14-08.pth"#"best_model_TVHuber.pth"
    #best_model_path = "_model_2025-12-12_13-52.pth"#"best_model_TVHuber.pth"
    #best_model_path = "_model_2025-12-16_11-02.pth"
    checkpoint_path = "checkpoints/best"+best_model_path

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

    print("Loading best model...")
    has_sigma = True

    from my_models import Regressor3  # your model class
    if has_sigma:
        model =  Regressor3Hetero(width=192, depth=8).to(device)#Regressor3(width=192, depth=8).to(device)  # your model class
    else:
        model =  Regressor3(width=192, depth=8).to(device)  # your model class
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_state_dict = checkpoint['state_dicts']['model']
    model.load_state_dict(model_state_dict)
    has_dropout = False
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            print("Found Dropout:", m)
            has_dropout = True

    print("Has dropout:", has_dropout)
    #model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if 0:
        enable_mc_dropout(model)

        q_pred, sigma_pred, q_true = mc_dropout_inference(model, test_loader)
        print("Model loaded.")

    # -----------------------
    # Run inference on test set
    # -----------------------
    else:

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
                #B = A_batch.shape[0]
                A_batch = A_batch.float().to(device)
                Q_batch_raw = SCALE_Y*Q_batch.float()                     # keep raw physical units

                # === Normalize input A ===
                A_mean = A_batch.mean(dim=1, keepdim=True)
                A_std  = A_batch.std(dim=1, keepdim=True) + 1e-8
                A_norm = (A_batch - A_mean) / A_std
                #A_norm = A_norm.unsqueeze(1)                      # [B,1,T]

                # === Normalize target Q (same as training) ===!!!! very wrong
      
                if has_sigma:
                    mu_norm , sigma= model(A_norm)
                          # [B, T]
                    sigma     = torch.exp(0.5*sigma).squeeze(1)  # [B, T]
                else:
                    mu_norm   = model(A_norm)        # [B, T]
                    sigma     = torch.zeros_like(mu_norm) + 1e-3
                mu_norm   = mu_norm.squeeze(1)
                # === Store each sample as [T] ===
                q_true_list.extend(Q_batch_raw.squeeze(1).cpu().numpy())     # list of [T]
                mu_pred_list.extend(mu_norm.numpy())                         # list of [T]
                sigma_pred_list.extend(sigma.numpy())   
                #sigma_pred_list.extend(sigma_raw.numpy())                  # list of [T]
            print(f"Unique arteries with max flow < 1: {len(unique_arteries)}, {list(unique_arteries)}")
            print(f"Number of samples with max flow < 1: {num_samples}")
            #plt.show()
        # === Final tensors ===
        q_true     = torch.tensor(np.stack(q_true_list))#.flatten()     # [N, T]
        q_pred    = torch.tensor(np.stack(mu_pred_list))#.flatten() 
        sigma_pred = torch.tensor(np.stack(sigma_pred_list))
        
        # example: RMSE per sample
        rmse = torch.sqrt(torch.mean((q_true - q_pred)**2, axis=1))
        norm_ = torch.sqrt(torch.mean(q_true**2, axis=1))
        rmse /= (norm_ + 1e-8)

        ranking = torch.argsort(rmse)   # ascending
        n = int(1*len(ranking))
        q_true, q_pred, sigma_pred = q_true[ranking], q_pred[ranking], sigma_pred[ranking]
        #q_true, q_pred, sigma_pred = q_true[:n], q_pred[:n], sigma_pred[:n]
        #q_true, q_pred, sigma_pred = q_true.numpy(), q_pred.numpy(), sigma_pred.numpy()
        if 0:
        
            for q, mu, s in zip(q_true[ranking][:], q_pred[ranking][:], sigma_pred[ranking][:]):
                continue
                plt.figure(figsize=(6, 3.5))
                t = np.arange(q.shape[0]) * dt
                plt.plot(t, q.numpy(), 'k', label='Ground truth', linewidth=2)
                plt.plot(t, mu.numpy(), 'C3', label='Prediction', linewidth=2)
                plt.fill_between(t,
                                mu.numpy() - s.numpy(),
                                mu.numpy() + s.numpy(),
                                color='C3', alpha=0.3, label=r'$\pm 1\sigma$')
                plt.xlabel("Time (s)")
                plt.ylabel("Flow (mL/s)")
                plt.title(f"Example of best predictions (RMSE = {torch.sqrt(torch.mean((q - mu)**2)).item():.2f} mL/s)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()
        # 1. Generate dummy data


    # 3. Compute ALL metrics
    dt = 0.01  # time step in seconds
    metrics = compute_metrics(
        q_true=q_true,
        mu_pred=q_pred,
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
    ax1 = plt.subplot(2, 1, 1)
    plot_waveforms_with_uncertainty(q_true, q_pred, sigma_pred, idx=0, ax=ax1,dt=dt)
    ax1.set_title("Example waveform with ±1σ uncertainty")

    ax2 = plt.subplot(2, 1, 2)
    plot_waveforms_with_uncertainty(q_true, q_pred, sigma_pred, idx=5000, ax=ax2,dt=dt)
    ax2.set_title("Another patient (higher variability)")

    # 5b – Bland-Altman on peak systolic flow
    plt.figure(figsize=(12, 8))
    ax3 = plt.subplot(1,1,1)
    bland_altman_peak_flow(q_true[:5000], q_pred[:5000], ax=ax3)

    # 5c – Calibration plot (observed vs expected confidence)
    plt.figure(figsize=(12, 8))
    ax4 = plt.subplot(1,1,1)
    z = ((q_true - q_pred) / (sigma_pred + 1e-8)).flatten().cpu().numpy()
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


if __name__ == "__main__":
    # Make sure you have fastdtw installed:  pip install fastdtw
    main()