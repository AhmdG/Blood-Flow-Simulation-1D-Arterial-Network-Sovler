import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml
import numpy as np
from vessel import Vessel, Blood
from train_pinn2 import Heart, Network, get_inlet_file, build_graph_topology, process_data, build_extrapol
from scipy.interpolate import interp1d
import torch.optim as optim
import os
from test_batch_builder import build_batch2, process_batch, pressure, newtone, build_batch
from pinn_library import build_index_map
import matplotlib.pyplot as plt 
from my_models import FlowPredictorUNet1D, FlowPredictor, calculate_loss 
from os import listdir
from os.path import isfile, join
def plot_prediction(model, inputs, targets, device='cpu', sample_idx=0, figsize=(10,6)):
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
    plt.title("Flow Prediction with Uncertainty")
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
# Example dummy dataset
# -------------------------------
class DummyPressureFlowDataset(Dataset):
    def __init__(self, n_samples=100, seq_len=512):
        self.pressure = torch.randn(n_samples, 1, seq_len)
        # Generate correlated target waveforms (smoothed pressure as pseudo-flow)
        self.flow = F.avg_pool1d(self.pressure, 5, stride=1, padding=2)
    def __len__(self): return len(self.pressure)
    def __getitem__(self, idx): return self.pressure[idx], self.flow[idx]


# -------------------------------
# Training loop
# -------------------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowPredictorUNet1D().to(device)#FlowPredictor().to(device)#
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    yaml_files = [f for f in listdir('yamls') if isfile(join('yamls', f))]
    batch_size = 128

    model_dir = "pinn_model_unsupervised_only3"
    config_once = True


    total_loss = 0.0
    all_losses = []
    new_loss_f = True

    warmup_epochs = 2000
    main_epochs = 20000
    every_epochs = 200
    device = next(model.parameters()).device
    T = 1
    standarize_output = True
    #############

    # hyperparams


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


    for epoch in range(warmup_epochs):

        yaml_config = yaml_files[np.random.randint(0,len(yaml_files))]#"adan56.yaml""adan3.yaml"
        with open('yamls/'+yaml_config, 'r') as f:
            config = yaml.safe_load(f)
        blood = Blood(config["blood"])
        heart = Heart(get_inlet_file(config))


        #f_interp = interp1d(heart.input_data[:, 0], heart.input_data[:, 1], kind='cubic')
        tosave = config.get("write_results", [])
        network = Network(config["network"], blood, heart,
                        config["solver"]["Ccfl"], config["solver"]["jump"],
                        tosave, verbose=True)
        #print("Network initialized with arteries:", len(network.vessels))
        graph_topology = build_graph_topology(network)
        data_path = os.path.join('results', yaml_config[:-5]+'_results/')#results/14-02-11-00-46_results/#'network_state_4/'#'network_state_5/' 5 is for only 3 arteries
        data_files = [f for f in listdir(data_path) if 'last' in f]
        artery_name = data_files[np.random.randint(0,len(data_files))][:-7]

        A_true_all = np.loadtxt(data_path+artery_name+'_P.last').T #x, t dims
        Q_true_all = np.loadtxt(data_path+artery_name+'_Q.last').T
        x_idx = np.random.randint(0, A_true_all.shape[0]-1)
        A_true =  A_true_all[x_idx, :]
        Q_true =  Q_true_all[x_idx, :]
        

    #for epoch in range(warmup_epochs):
        model.train()
        A_true = torch.from_numpy(A_true).float()
        Q_true = torch.from_numpy(Q_true).float()
        if standarize_output and 1:
            A_true = (A_true - A_true.mean())/A_true.std()
            Q_true = (Q_true - Q_true.mean())/Q_true.std()
        inputs = A_true
        targets = Q_true
        inputs = inputs.to(device)
        targets = targets.to(device)
        mu, log_sigma = model(inputs.unsqueeze(0).unsqueeze(0))
        target_c = crop_target_to_match(mu, targets.unsqueeze(0).unsqueeze(0))
        loss = F.mse_loss(mu, target_c)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"Warmup Epoch {epoch+1}/{warmup_epochs}, mse loss={loss.item():.6f}")

    # Unfreeze sigma params
    for p in sigma_params:
        p.requires_grad = True

    # Recreate optimizer (include all params) for NLL phase
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Main NLL training with clamped log_sigma
    total_loss = 0.0
    for epoch in range(main_epochs):
        model.train()
        
        art = np.random.randint(0,len(network.vessels))
        A_true, Q_true = build_batch2(network, graph_topology, T, data_dict, batch_size=512, xs=None, idx0s=[art])
        #sample, A_true, Q_true = build_batch(network, graph_topology, T, data_dict, batch_size=512, xs=None, idx0s=[idx0s])
        A_true = torch.from_numpy(A_true).float()
        Q_true = torch.from_numpy(Q_true).float()
        if standarize_output and 1:
            A_true = (A_true - A_true.mean())/A_true.std()
            Q_true = (Q_true - Q_true.mean())/Q_true.std()
        inputs = A_true
        targets = Q_true
        inputs = inputs.to(device)
        targets = targets.to(device)
        mu, log_sigma = model(inputs.unsqueeze(0).unsqueeze(0))

        # clamp sigma strongly
        log_sigma = torch.clamp(log_sigma, min=-3.0, max=1.0)

        target_c = crop_target_to_match(mu, targets.unsqueeze(0).unsqueeze(0))
        loss = gaussian_nll_loss(mu, log_sigma, target_c)
        #loss_phy, _ = calculate_loss(A_true, mu, A_true, Q_true, x, t, sample, extrapolators, f_interp, epoch=epoch,
                        #rho = model.rho, gamma_profile=model.gamma_profile, mu=model.mu)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        avg = total_loss #/ len(loader)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss = {avg:.4f}")
            plot_prediction(model, A_true.unsqueeze(0).unsqueeze(0), Q_true.unsqueeze(0).unsqueeze(0), device='cpu', sample_idx=0, figsize=(10,6))
    ##########################
    