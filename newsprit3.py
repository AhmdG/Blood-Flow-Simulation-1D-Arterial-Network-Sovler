import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from newsprit2 import ArteryMemoryDataset, compute_global_mean_std, save_checkpoint

import os
import yaml
from os import listdir
from os.path import isfile, join 


class Regressor3Hetero(nn.Module):
    def __init__(self, width=128, depth=8, skip=True):
        super().__init__()
        layers = []

        # Initial convolution: 1 → width
        layers.append(nn.Conv1d(1, width, kernel_size=7, padding=3))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(depth):
            if i % 2 == 0:
                layers.append(nn.Conv1d(width, width, kernel_size=5, padding=2))
            else:
                layers.append(nn.Conv1d(width, width, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())

        # Final 1x1 convolution: width → 2 (μ and log σ²)
        layers.append(nn.Conv1d(width, 2, kernel_size=1, padding=0))

        self.net = nn.Sequential(*layers)
        self.skip = skip

        self._initialize_smooth_weights()

    def _initialize_smooth_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv1d):
                n = layer.kernel_size[0] * layer.in_channels
                layer.weight.data.normal_(0, torch.sqrt(torch.tensor(2. / n)))
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, x):
        """
        x: (B, L)
        returns:
            mu:      (B, L)
            log_var: (B, L)
        """
        input_for_skip = x
        x = x.unsqueeze(1)            # (B, 1, L)

        out = self.net(x)             # (B, 2, L)
        mu = out[:, 0, :]             # (B, L)
        log_var = out[:, 1, :]        # (B, L)

        if self.skip:
            mu = mu + 0.1 * input_for_skip

        return mu, log_var
  
class Regressor(nn.Module):

    def __init__(self, width=128, depth=6, skip=True):

        super().__init__()
        layers = []
        layers.append(nn.Conv1d(1, width, kernel_size=3, padding=1))
        #layers.append(nn.ReLU())
        layers.append(nn.Tanh())
        
        for i in range(depth):
            if i % 2 == 0:  # Every other layer uses larger kernel
                layers.append(nn.Conv1d(width, width, kernel_size=5, padding=2))
            else:
                layers.append(nn.Conv1d(width, width, kernel_size=3, padding=1))
            #layers.append(nn.Conv1d(width, width, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            
        # Final layer: back to 1 channel
        layers.append(nn.Conv1d(width, 1, kernel_size=1))   # 1×1 conv
        
        self.net = nn.Sequential(*layers)
        
        # Optional: residual connection (very powerful here)
        self.skip = skip

        #
        self._initialize_smooth_weights()


    def _initialize_smooth_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv1d):
                # Use Gaussian initialization for smoother filters
                n = layer.kernel_size[0] * layer.in_channels
                layer.weight.data.normal_(0, torch.sqrt(torch.tensor(2. / n)))
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, x):
        # x shape: (batch, 100) → (batch, 1, 100)
        x = x.unsqueeze(1)
        out = self.net(x)
        out = out.squeeze(1)
        
        if self.skip:
            out = out + x.squeeze(1) * 0.1   # weak residual (because output ≈ small correction)
        return out
    
class HuberWithTV(nn.Module):
    def __init__(self, delta=1):#, tv_weight=0.4):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        #self.tv_weight = tv_weight
        self.delta = delta
        
    def forward(self, pred, target):
        huber = self.huber(pred, target)
        
        # Total Variation regularization (promotes piecewise smoothness)
        #tv = torch.sum(torch.abs(pred[:, 1:] - pred[:, :-1]), dim=1).mean()
        
        return huber #+ self.tv_weight * tv
    def get_params(self):
        return {'delta': self.delta}#, 'tv_weight': self.tv_weight}


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


def heteroscedastic_nll(y_true, mu, log_var):
    return torch.mean(
        0.5 * torch.exp(-log_var) * (y_true - mu) ** 2
        + 0.5 * log_var
    )

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    yaml_files = [f for f in listdir('yamls') if isfile(join('yamls', f))]
    warmup_epochs = 200
    main_epochs = 1000
    every_epochs = 20
    #device = next(model.parameters()).device
    standarize_output = True
    from datetime import datetime

    # Auto unique name:  best_2025-12-04_15-27.pth   (or add seconds: _15-27-08)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")      # change to %H-%M-%S if you run very often
    best_model_path = f"best_model_{timestamp}.pth"
    # best_model_path = "best_model_delta=1.0, tv_weight=0.03.pth"
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
    train_samples = filter_damaged_data(train_samples, max_allowed_flow=1.0)
    test_samples = filter_damaged_data(test_samples, max_allowed_flow=1.0)
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

    if 0:

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

    ####


    train_losses = []
    val_losses = []
    #criterion = torch.nn.HuberLoss(delta=1.0)
    criterion = HuberWithTV(delta=400, tv_weight=0.5)
    # =============================================
    # 1. Your data (replace with your real X, y)
    # =============================================


    # =============================================
    # 2. Critical preprocessing for tiny targets
    # =============================================
    SCALE_Y = 1_000_000.0          # 1e6 → brings y to roughly [-20, +300]


    model = Regressor3Hetero(width=192, depth=8).to(device)#Regressor(width=192, depth=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15)

    checkpoint_path = os.path.join("checkpoints", "last_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print("Checkpoint found. Loading and resuming training...")
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        start_epoch = checkpoint["epoch"] + 1

    else:
        print("No checkpoint found. Starting training from scratch.")

    # =============================================
    # 5. Training loop
    # =============================================
    epochs = 500
    best_val_loss = np.inf
    patience = 50
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for A_batch, Q_batch, _, _ , _ in train_loader:
            A_batch = (A_batch - A_batch.mean(dim=1, keepdim=True)) / (A_batch.std(dim=1, keepdim=True) + 1e-8).to(device)
            Q_batch = SCALE_Y*Q_batch.to(device)
                
            #pred = model(A_batch.float())
            #loss = criterion(pred.squeeze(), Q_batch.float())
            mu, log_var = model(A_batch.float())
            loss = heteroscedastic_nll(pred.squeeze(), mu, log_var)
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * A_batch.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for A_batch, Q_batch, _ , _ , _ in test_loader:
                A_batch = (A_batch - A_batch.mean(dim=1, keepdim=True)) / (A_batch.std(dim=1, keepdim=True) + 1e-8)
                Q_batch = SCALE_Y*Q_batch

                pred = model(A_batch.float())
                val_loss += criterion(pred.squeeze(), Q_batch.float()).item() * A_batch.size(0)
        
        train_loss /= len(train_loader)
        val_loss   /= len(test_loader)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss - 1/(10*SCALE_Y):   # small tolerance
            best_val_loss = val_loss
            print(f"New best model found at epoch {epoch}, val loss: {val_loss:.4f}")
            save_checkpoint({
                                "epoch": epoch,
                                "model_state": model.state_dict(),
                                "optim_state": optimizer.state_dict(),
                                "val_loss": val_loss,
                                'criterion_kwargs': criterion.get_params(),
                            }, is_best=True,
                            last_filename="last" + best_model_path,
                            best_filename="best" + best_model_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        if (epoch+1) % 20 == 0 or epoch < 10:
            # Show error in original units
            mae_original = val_loss / SCALE_Y          # because Huber ≈ MAE here
            print(f"Epoch {epoch+1:3d} | Val loss (scaled): {val_loss:.4f} | "
                f"≈ MAE original: {mae_original:.2e}")
            save_checkpoint({
                                "epoch": epoch,
                                "model_state": model.state_dict(),
                                "optim_state": optimizer.state_dict(),
                                "val_loss": val_loss,
                                'criterion_kwargs': criterion.get_params(),
                            }, is_best=False,
                            last_filename="last"+best_model_path,
                            best_filename="best"+best_model_path)

    print("Training finished!")
    #model.load_state_dict(torch.load("best_model.pth"))