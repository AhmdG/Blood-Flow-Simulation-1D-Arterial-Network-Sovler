import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml
import numpy as np
import os
import matplotlib.pyplot as plt 
from my_models import FlowPredictorUNet1D, FlowPredictor, calculate_loss ,ProbabilisticFlowNet
from os import listdir
from os.path import isfile, join
import random
from sklearn.model_selection import train_test_split


def save_checkpoint(state, is_best, checkpoint_dir="checkpoints",
                    last_filename=None,
                    best_filename=None):
    """
    Saves training state. Writes both the last checkpoint and,
    if `is_best` is True, also saves a separate best-model file.

    Args:
        state (dict): Contains model_state, optim_state, epoch, etc.
        is_best (bool): True if current state is the best so far.
        checkpoint_dir (str): Directory to save checkpoints.
        last_filename (str): Filename for the most recent checkpoint.
        best_filename (str): Filename for the best checkpoint.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the most recent checkpoint
    last_path = os.path.join(checkpoint_dir, last_filename)
    torch.save(state, last_path)

    # If best model, also save a separate file
    if is_best:
        best_path = os.path.join(checkpoint_dir, best_filename)
        torch.save(state, best_path)
        #print(f"✔ Saved new BEST model → {best_path}")

    #print(f"💾 Saved LAST checkpoint → {last_path}")

seed = 12345

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if 0:
    warmup_samples  = []
    for epoch in range(warmup_epochs):
        sample = random.choice(all_samples)
        warmup_samples.append(sample)

        A_true = torch.tensor(sample["A_true"], dtype=torch.float32, device=device)
        Q_true = torch.tensor(sample["Q_true"], dtype=torch.float32, device=device)

        if standarize_output:
            A_true = (A_true - A_true.mean()) / (A_true.std() + 1e-8)
            Q_true = (Q_true - Q_true.mean()) / (Q_true.std() + 1e-8)

        model.train()

        mu, log_sigma = model(A_true[None, None, :])
        target_c = crop_target_to_match(mu, Q_true[None, None, :])
        loss = F.mse_loss(mu, target_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("epoch", epoch, "loss", loss.item())

from torch.utils.data import Dataset, DataLoader

def compute_global_mean_std(train_loader, device='cpu'):
    """
    Computes mean and std of the input tensor A across the whole training dataset.
    Assumes train_loader yields: (A_batch, Q_batch, ...)
    A_batch shape: [B, T]  (or [B, C, T] if already channeled)
    """
    mean_sum = 0.0
    std_sum = 0.0
    n_samples = 0

    print("Computing global flow mean and std over training set...")
    
    for _, Q_batch, _, _, _ in train_loader:   # adjust if you have different loader structure
        Q_batch = Q_batch.float().to(device)    # [B, T] or [B, C, T]
        
        # Flatten to treat all time steps and channels equally
        Q_flat = Q_batch.view(Q_batch.size(0), -1)  # [B, T*C]
        
        batch_mean = Q_flat.mean(dim=1)   # [B]
        batch_std  = Q_flat.std(dim=1)    # [B]
        
        # Accumulate for global mean
        mean_sum += batch_mean.sum().item()
        
        # For variance: E[X^2] - (E[X])^2
        std_sum += (batch_std**2 * Q_flat.size(1)).sum().item()  # sum of variances * n_elements_per_sample
        
        n_samples += Q_batch.size(0)
    
    # Global mean (per sample, then average over samples)
    global_mean = mean_sum / n_samples
    
    # Global variance = total sum of variances / total elements
    total_elements = n_samples * Q_flat.size(1)  # B * (T*C)
    global_variance = std_sum / total_elements
    global_std = torch.sqrt(torch.tensor(global_variance))
    
    print(f"Global mean: {global_mean:.6f}")
    print(f"Global std:  {global_std.item():.6f}")
    
    return global_mean, global_std.item()

# Usage example
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ArteryMemoryDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        A = torch.tensor(s["A_true"])              # shape [T]
        Q = torch.tensor(s["Q_true"])              # shape [T]
        return A, Q, s["yaml_model"], s["artery"], s["x_idx"]

def plot_prediction(model, inputs, targets, device='cpu', sample_idx=0, figsize=(10,6), title=""):
    model.eval()
    model.to(device)

    # ---- 1. Move data to device ----
    
    inputs = inputs.to(device)      # shape: [B, 1, T]
    targets = targets.to(device)    # shape: [B, 1, T?]

    # ---- 2. Run forward pass ----
    mu, log_sigma = model(inputs)
    sigma = torch.exp(log_sigma)

    # ---- 3. Crop target to match mu size ----
    # mu: [B, 1, Tpred]
    B, C, Tpred = mu.shape
    _, _, Tt = targets.shape
    
    start = (Tt - Tpred) // 2
    end = start + Tpred
    target_c = targets[:, :, start:end]    # shape: [B, 1, Tpred]

    # ---- 4. Choose one sample from the batch ----
    mu_np      = mu[sample_idx,0].detach().numpy()
    sigma_np   = sigma[sample_idx,0].detach().numpy()
    target_np  = target_c[sample_idx,0].detach().numpy()

    time = np.arange(len(mu_np))

    # ---- 5. Plot ----
    plt.figure(figsize=figsize)
    plt.plot(time, target_np, label='True Flow', linewidth=2)
    plt.plot(time, mu_np, label='Predicted μ', linewidth=2)
    plt.fill_between(time, mu_np - sigma_np, mu_np + sigma_np,
                     alpha=0.3, label='μ ± σ (Uncertainty)')

    plt.xlabel("Time index")
    plt.ylabel("Normalized flow")
    plt.title(title)#"Flow Prediction with Uncertainty")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
# -------------------------------
# 1D U-Net-style network
# -------------------------------

# -------------------------------
# Gaussian NLL loss
# -------------------------------

def crop_target_to_match(output, target):
    """
    output: (B, 1, Lout)
    target: (B, 1, Ltarget) — usually larger
    Return: cropped target with length = Lout
    """
    _, _, Lout = output.shape
    _, _, Ltarget = target.shape

    if Lout == Ltarget:
        return target

    diff = Ltarget - Lout
    start = diff // 2
    end = start + Lout
    return target[:, :, start:end]

def gaussian_nll_loss(mu, log_sigma, target, eps=1e-6):
    target = crop_target_to_match(mu, target)
    sigma2 = torch.exp(2 * log_sigma) + eps
    nll = 0.5 * ((target - mu)**2 / sigma2) + 0.5 * torch.log(2 * torch.pi * sigma2)
    return nll.mean()


# -------------------------------
# Training loop
# -------------------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowPredictorUNet1D().to(device)#ProbabilisticFlowNet().to(device) ##FlowPredictor().to(device)#
    yaml_files = [f for f in listdir('yamls') if isfile(join('yamls', f))]
    warmup_epochs = 200
    main_epochs = 1000
    every_epochs = 20
    #device = next(model.parameters()).device
    standarize_output = True
    best_model_path = "best_model_Global_Q.pt"
    start_epoch = 0  # update if loading checkpoint

    # check for existing dataset files
    if os.path.exists("dataset_train.pkl") and os.path.exists("dataset_test.pkl"):
        print("Loading existing dataset files...")
        import pickle  
        with open("dataset_train.pkl", "rb") as f:
            train_samples = pickle.load(f)
        with open("dataset_test.pkl", "rb") as f:
            test_samples = pickle.load(f)
        print(f"Loaded {len(train_samples)} training samples and {len(test_samples)} test samples.")
    else:
        print("Building dataset from scratch...")
             
        #----build the data
        all_samples = []

        for yaml_config in yaml_files:
            model_name = yaml_config.replace(".yaml", "")
            yaml_path = os.path.join("yamls", yaml_config)

            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f)

            data_dir = os.path.join("results", f"{model_name}_results")

            P_files = [f for f in os.listdir(data_dir) if f.endswith("_P.last")]

            for P_file in P_files:
                artery_name = P_file.replace("_P.last", "")

                P = np.loadtxt(os.path.join(data_dir, f"{artery_name}_P.last")).T
                Q = np.loadtxt(os.path.join(data_dir, f"{artery_name}_Q.last")).T

                n_x = P.shape[0]  # number of spatial positions
                for x_idx in range(n_x):
                    all_samples.append({
                        "A_true": P[x_idx].copy(),
                        "Q_true": Q[x_idx].copy(),
                        "yaml_model": model_name,
                        "artery": artery_name,
                        "x_idx": x_idx,
                    })

        print("Preloaded", len(all_samples), "samples")
        #############
        train_samples, test_samples = train_test_split(
            all_samples, test_size=0.2, shuffle=True, random_state=42
        )

        print(f"Train: {len(train_samples)} | Test: {len(test_samples)}")

        import pickle

        with open("dataset_train.pkl", "wb") as f:
            pickle.dump(train_samples, f)

        with open("dataset_test.pkl", "wb") as f:
            pickle.dump(test_samples, f)

        print("Saved dataset to dataset_train.pkl and dataset_test.pkl")


    batch_size = 64

    train_loader = DataLoader(ArteryMemoryDataset(train_samples),
                          batch_size=batch_size,
                          shuffle=True)
    if os.path.exists("normalization_stats.pth"):
        print("Loading existing normalization stats...")
        stats = torch.load("normalization_stats.pth")
        GLOBAL_MEAN = stats["mean"]
        GLOBAL_STD  = stats["std"]
        print(f"Loaded GLOBAL_MEAN: {GLOBAL_MEAN:.6f}, GLOBAL_STD: {GLOBAL_STD:.6f}")
    else:
        global_mean, global_std = compute_global_mean_std(train_loader, device=device)
        # Save them for later use (inference, validation, etc.)
        torch.save({"mean": global_mean, "std": global_std}, "normalization_stats.pth")#
        # Load once at the beginning
        stats = torch.load("normalization_stats.pth")
        GLOBAL_MEAN = stats["mean"]
        GLOBAL_STD  = stats["std"]

    test_loader = DataLoader(ArteryMemoryDataset(test_samples),
                            batch_size=batch_size,
                            shuffle=False)
    



    # zero sigma bias init (if exists)
    if hasattr(model, "out_log_sigma"):
        with torch.no_grad():
            try:
                model.out_log_sigma.bias.fill_(0.0)
            except Exception:
                pass

    # OPTIONAL: get sigma head params to freeze
    sigma_params = []
    if hasattr(model, "out_log_sigma"):
        sigma_params = list(model.out_log_sigma.parameters())

    # Freeze sigma head
    for p in sigma_params:
        p.requires_grad = False

    # Warmup with MSE
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    ####
    checkpoint_path = os.path.join("checkpoints", "last_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print("Checkpoint found. Loading and resuming training...")
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        start_epoch = checkpoint["epoch"] + 1

    else:
        print("No checkpoint found. Starting training from scratch.")

    train_losses = []
    val_losses = []
    criterion = torch.nn.HuberLoss(delta=1.0)
    for epoch in range(start_epoch, warmup_epochs):
        model.train()
        running_loss = 0.0

        for A_batch, Q_batch, _, _, _ in train_loader:
            # A_batch: [B, T]
            # Q_batch: [B, T]
            A_batch = A_batch.float().to(device)
            Q_batch = Q_batch.float().to(device)

            if standarize_output:
                #A_batch = (A_batch - GLOBAL_MEAN) / (GLOBAL_STD + 1e-8)
                A_batch = (A_batch - A_batch.mean(dim=1, keepdim=True)) / (A_batch.std(dim=1, keepdim=True) + 1e-8)
                #Q_batch = (Q_batch - Q_batch.mean(dim=1, keepdim=True)) / (Q_batch.std(dim=1, keepdim=True) + 1e-8)
                Q_batch = (10**4)*Q_batch# - GLOBAL_MEAN) / (GLOBAL_STD + 1e-8)

            # Add channel & batch dims: [B, 1, T]
            A_batch = A_batch.unsqueeze(1)
            Q_batch = Q_batch.unsqueeze(1)

            mu, log_sigma = model(A_batch)
            target_c = crop_target_to_match(mu, Q_batch)

            #loss = torch.nn.functional.mse_loss(mu, target_c)
            loss = criterion(mu, target_c)
            

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

        # Average train loss
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        if epoch % 5 == 0:
            save_checkpoint({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "val_loss": 0,
    }, is_best=False)

        # ----------------------
        # Validation
        # ----------------------
        if 0:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for A_batch, Q_batch, _, _, _ in test_loader:
                    A_batch = A_batch.float().to(device)
                    Q_batch = Q_batch.float().to(device)

                    if standarize_output:
                        A_batch = (A_batch - A_batch.mean(dim=1, keepdim=True)) / (A_batch.std(dim=1, keepdim=True) + 1e-8)
                        #Q_batch = (Q_batch - Q_batch.mean(dim=1, keepdim=True)) / (Q_batch.std(dim=1, keepdim=True) + 1e-8)
                        Q_batch = (Q_batch - GLOBAL_MEAN) / (GLOBAL_STD + 1e-8)

                    A_batch = A_batch.unsqueeze(1)
                    Q_batch = Q_batch.unsqueeze(1)

                    mu, log_sigma = model(A_batch)
                    target_c = crop_target_to_match(mu, Q_batch)
                    val_loss += torch.nn.functional.mse_loss(mu, target_c).item()

            epoch_val_loss = val_loss / len(test_loader)
            val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{warmup_epochs} | "
            f"Train Loss: {epoch_train_loss:.6f}  ")
            #f"Val Loss: {epoch_val_loss:.6f}")

    # Unfreeze sigma params
    for p in sigma_params:
        p.requires_grad = True

    # Recreate optimizer (include all params) for NLL phase
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Main NLL training with clamped log_sigma
    total_loss = 0.0
    best_val_loss = float("inf")
    
    for epoch in range(start_epoch, main_epochs):
        model.train()
        running_loss = 0.0

        for A_batch, Q_batch, _, _, _ in train_loader:
            # A_batch: [B, T]
            # Q_batch: [B, T]
            A_batch = A_batch.float().to(device)
            Q_batch = Q_batch.float().to(device)

            if standarize_output:
                #A_batch = (A_batch - GLOBAL_MEAN) / (GLOBAL_STD + 1e-8)
                A_batch = (A_batch - A_batch.mean(dim=1, keepdim=True)) / (A_batch.std(dim=1, keepdim=True) + 1e-8)
                #Q_batch = (Q_batch - Q_batch.mean(dim=1, keepdim=True)) / (Q_batch.std(dim=1, keepdim=True) + 1e-8)
                Q_batch = (Q_batch - GLOBAL_MEAN) / (GLOBAL_STD + 1e-8)

            # Add channel & batch dims: [B, 1, T]
            A_batch = A_batch.unsqueeze(1)
            Q_batch = Q_batch.unsqueeze(1)

            mu, log_sigma = model(A_batch)
            target_c = crop_target_to_match(mu, Q_batch)

            loss = gaussian_nll_loss(mu, log_sigma, target_c)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

        # Average train loss
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # ----------------------
        # Validation
        # ----------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for A_batch, Q_batch, _, _, _ in test_loader:
                A_batch = A_batch.float().to(device)
                Q_batch = Q_batch.float().to(device)

                if standarize_output:
                    
                    A_batch = (A_batch - A_batch.mean(dim=1, keepdim=True)) / (A_batch.std(dim=1, keepdim=True) + 1e-8)
                    #Q_batch = (Q_batch - Q_batch.mean(dim=1, keepdim=True)) / (Q_batch.std(dim=1, keepdim=True) + 1e-8)
                    Q_batch = (Q_batch - GLOBAL_MEAN) / (GLOBAL_STD + 1e-8)
                    

                A_batch = A_batch.unsqueeze(1)
                Q_batch = Q_batch.unsqueeze(1)

                mu, log_sigma = model(A_batch)
                target_c = crop_target_to_match(mu, Q_batch)
                val_loss += gaussian_nll_loss(mu, log_sigma, target_c)#torch.nn.functional.mse_loss(mu, target_c).item()

        epoch_val_loss = val_loss / len(test_loader)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{main_epochs} | "
            f"Train Loss: {epoch_train_loss:.6f} | "
            f"Val Loss: {epoch_val_loss:.6f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Saved best model (val={epoch_val_loss:.6f})")
            save_checkpoint({
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optim_state": optimizer.state_dict(),
                            "val_loss": val_loss,
                        }, is_best=True)
        if epoch % 5 == 0:
            save_checkpoint({
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "val_loss": val_loss,
                    }, is_best=False)


        if epoch % every_epochs == 0 and 1:
            #plot_sample_idx = 0  # you choose
            plot_sample = random.choice(test_samples)#test_samples[plot_sample_idx]

            plot_yaml  = plot_sample["yaml_model"]
            plot_artery = plot_sample["artery"]
            plot_xidx = plot_sample["x_idx"]
            plot_title = f"{plot_yaml} | {plot_artery} | x={plot_xidx}"

            #plot_prediction(model, A_true[None, None, :], Q_true[None, None, :], device='cpu', sample_idx=0, figsize=(10,6))
            plot_prediction(model, A_batch, Q_batch, device='cpu', sample_idx=0, figsize=(10,6), title=plot_title)
    ##########################
    