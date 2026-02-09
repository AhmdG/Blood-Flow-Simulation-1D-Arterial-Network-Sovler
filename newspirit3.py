from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import torch.nn as nn
from my_models import FlowPredictorUNet1D
from os import listdir
from os.path import isfile, join
import random
from sklearn.model_selection import train_test_split
import os
import numpy as np

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

# 2. Create datasets that z-score targets on-the-fly using FIXED stats
class ZScoredFlowDataset(TensorDataset):
    def __init__(self, X_raw, y_raw, mean_flow, std_flow,
                 x_mean=None, x_std=None):
        super().__init__(X_raw, y_raw)
        self.mean_flow = mean_flow
        self.std_flow  = std_flow
        self.x_mean = x_mean
        self.x_std  = x_std

    def __getitem__(self, idx):
        x, y_raw = super().__getitem__(idx)
        # z-score inputs if you have stats for them
        if self.x_mean is not None:
            x = (x - self.x_mean) / (self.x_std + 1e-8)
        # z-score target with FIXED training statistics
        y_z = (y_raw - self.mean_flow) / (self.std_flow + 1e-8)
        return x, y_z
    

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

if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowPredictorUNet1D().to(device)#FlowPredictor().to(device)#
    yaml_files = [f for f in listdir('yamls') if isfile(join('yamls', f))]
    warmup_epochs = 20
    main_epochs = 200
    every_epochs = 1
    #device = next(model.parameters()).device
    standarize_output = True

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


    batch_size = 32

    train_loader = DataLoader(ArteryMemoryDataset(train_samples),
                          batch_size=batch_size,
                          shuffle=True)
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
    if 0:

        # 1. Compute statistics ONLY on training targets (raw values!)
        train_flow_raw = train_dataset[:][1]          # or however you extract targets
        mean_flow = train_flow_raw.mean()
        std_flow  = train_flow_raw.std()              # population std, not sample



        # Build datasets
        train_ds = ZScoredFlowDataset(X_train_raw, y_train_raw, mean_flow, std_flow,
                                    x_mean=X_train_raw.mean(0), x_std=X_train_raw.std(0))
        val_ds   = ZScoredFlowDataset(X_val_raw,   y_val_raw,   mean_flow, std_flow,
                                    x_mean=X_train_raw.mean(0), x_std=X_train_raw.std(0))

        train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=2048, shuffle=False)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        model.train()
        for A_batch, Q_batch, _, _, _ in train_loader:
            A_batch = (A_batch - A_batch.mean(dim=1, keepdim=True)) / (A_batch.std(dim=1, keepdim=True) + 1e-8)

            Q_batch = (Q_batch - GLOBAL_MEAN) / (GLOBAL_STD + 1e-8)
            optimizer.zero_grad()

            mu_z, log_sigma_z = model(x_batch)
            loss = gaussian_nll_loss(mu_z, log_sigma_z, y_z_batch)
            loss.backward()
            optimizer.step()

        # Validation (optional, but use the same fixed scaling)
        model.eval()
        with torch.no_grad():
            val_losses = []
            for x_batch, y_z_batch in val_loader:
                mu_z, log_sigma_z = model(x_batch)
                val_losses.append(gaussian_nll_loss(mu_z, log_sigma_z, y_z_batch))
            print(f"Epoch {epoch} — val NLL: {torch.stack(val_losses).mean():.4f}")