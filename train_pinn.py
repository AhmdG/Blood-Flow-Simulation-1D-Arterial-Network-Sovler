import torch
import torch.nn as nn
import torch.autograd as autograd
#from vessel import Vessel
from torch.utils.data import Dataset
import os
import numpy as np
import json

import os
import json
import numpy as np

from torch.utils.data import DataLoader
import torch.optim as optim

class ArteryDatasetOnDemand(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.arteries = []#{} # List of (artery_path, t_prev_file, t_curr_file)
        self.cycles = len(next(os.walk(root_dir))[1])
        self.time_files = []
        self.valid_indices = {}
        self.valid_indices_all = []
        self.cycle_lengths = []
        self.len = 0
        
        for cycle in range(self.cycles):
            first_artery_path = os.path.join(root_dir+f"/{cycle}/", os.listdir(root_dir+f"/{cycle}")[0])
            self.time_files.append(sorted([
                    f for f in os.listdir(first_artery_path)
                    if f.endswith(".npz")
                ]))
            self.cycle_lengths.append(len(self.time_files[cycle]))
            self.valid_indices[cycle] = self.time_files
            self.len += self.cycle_lengths[cycle]
        cycle = 0
        for artery_folder in sorted(os.listdir(root_dir+f"/{cycle}")):

            artery_path = os.path.join(root_dir+f"/{cycle}", artery_folder)
            self.meta = json.load(open(os.path.join(artery_path, "meta.json")))

            #x = torch.linspace(0, meta['L'], int(meta['L']*1000))
            if not os.path.isdir(artery_path):
                continue

            #for i, j in zip(time_files[:-1],time_files[1:]) :
                #self.samples.append((artery_path, i, j))
            self.arteries.append(artery_folder)#{'x':x,'meta':meta}#self.samples.get(artery_path, [])+[[i,j]]
            pass
        self.structure =[
                    f for f in os.listdir(first_artery_path)
                    if f.endswith(".npy")]
        pass
        for cycle in range(self.cycles):
            #self.valid_indices_all[cycle] = []
            for i, artery_folder in enumerate(self.arteries):
                for j, f in enumerate(self.time_files[cycle][1:]):
                    self.valid_indices_all.append((cycle, i, j+1)) # +1 to skip the first time step

                    #self.valid_indices_all.append([cycle, artery_folder, f ] for f in self.time_files[cycle])
        pass


    def __len__(self):
        return len(self.valid_indices_all)
    
    def find_indices(self, idx, cycles, arteries, lengths):
        offset = 0
        for c in range(cycles):
            total = lengths[c] * arteries
            if idx < offset + total:
                local = idx - offset
                artery = local // lengths[c]
                sample = local % lengths[c]
                return c, artery, sample
            offset += total
        raise IndexError("Index out of bounds")

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            cycle, i, artery_folder = idx
        else:
            cycle, artery_index, i = self.valid_indices_all[idx]#self.find_indices( idx, self.cycles, len(self.arteries), self.cycle_lengths)

            artery_folder = self.arteries[artery_index] 

        structure = {}
        f_curr = self.time_files[cycle][i]
        f_prev = self.time_files[cycle][i-1]
        #f_curr = self.samples[i]
        #prev = self.samples[i-1]
        artery_path = os.path.join(self.root_dir+f"/{cycle}", artery_folder)

        # Load meta
        if not hasattr(self, 'cached_meta') or self.cached_meta_path != artery_path:
            meta = json.load(open(os.path.join(artery_path, "meta.json")))
            for key in self.structure:
                structure[key[:-4]] = np.load(os.path.join(artery_path, key))

            #self.cached_meta = (meta, A0)
            #self.cached_meta_path = artery_path
        else:
            meta, A0 = self.cached_meta

        # Load time step data
        prev = np.load(os.path.join(artery_path, f_prev), allow_pickle=False)
        curr = np.load(os.path.join(artery_path, f_curr), allow_pickle=False)

        #x =  torch.linspace(0, meta['L'], meta['M']).unsqueeze(-1)#self.arteries[artery_folder] #torch.tensor(prev['x'], dtype=torch.float32).unsqueeze(-1)
        #Q_prev = torch.tensor(prev['Q'], dtype=torch.float32).unsqueeze(-1)
        #A_prev = torch.tensor(prev['A'], dtype=torch.float32).unsqueeze(-1)
        Q_true = torch.tensor(curr['Q'], dtype=torch.float32)
        A_true = torch.tensor(curr['A'], dtype=torch.float32)
        Q_in = torch.tensor(prev['Q_in'], dtype=torch.float32)# not current['Q_in']
        P_out = torch.tensor(prev['P_out'], dtype=torch.float32)
        Q_in_next = torch.tensor(curr['Q_in'], dtype=torch.float32)# not current['Q_in']
        P_out_next = torch.tensor(curr['P_out'], dtype=torch.float32)



        #return x, Q_prev, A_prev, Q_true, A_true,  meta, Q_in, P_out, structure
        return structure['x'], Q_true, A_true,  meta, Q_in, P_out, Q_in_next, P_out_next, structure

class ArteryDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        cycles = len(next(os.walk(root_dir))[1])
        for cycle in range(cycles):
            for artery_dir in os.listdir(root_dir+f"/{cycle}"):
                artery_path = os.path.join(root_dir+f"/{cycle}", artery_dir)
                if not os.path.isdir(artery_path):
                    continue

                meta = json.load(open(os.path.join(artery_path, "meta.json")))
                x = torch.linspace(0, meta['L'], int(meta['L']*1000))
                A0 = np.load(os.path.join(artery_path, "A0.npy"))

                time_files = sorted([
                    f for f in os.listdir(artery_path) if f.startswith("t") and f.endswith(".npz")
                ]) #float(f[1:-4])/1e10

                for i in range(len(time_files) - 1):
                    t_prev = np.load(os.path.join(artery_path, time_files[i]))
                    t_next = np.load(os.path.join(artery_path, time_files[i + 1]))

                    self.samples.append({
                        'x': x,
                        'Q_prev': t_prev['Q'],
                        'A_prev': t_prev['A'],
                        'Q_true': t_next['Q'],
                        'A_true': t_next['A'],
                        #'Q_in': t_next['Q_in'],
                        #'P_out': t_next['P_out'],
                        'A0': A0,
                        'meta': meta,
                        'cycle':cycle
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Convert all to torch tensors
        x = torch.tensor(s['x'], dtype=torch.float32).unsqueeze(-1)
        Q_prev = torch.tensor(s['Q_prev'], dtype=torch.float32).unsqueeze(-1)
        A_prev = torch.tensor(s['A_prev'], dtype=torch.float32).unsqueeze(-1)
        Q_true = torch.tensor(s['Q_true'], dtype=torch.float32)
        A_true = torch.tensor(s['A_true'], dtype=torch.float32)
        #Q_in = torch.tensor([s['Q_in']], dtype=torch.float32)
        #P_out = torch.tensor([s['P_out']], dtype=torch.float32)
        A0 = torch.tensor(s['A0'], dtype=torch.float32).unsqueeze(-1)
        artery_params = torch.tensor([s['meta']['Rd'], s['meta']['Ru'], s['meta']['L']], dtype=torch.float32)

        return x, Q_prev, A_prev, Q_true, A_true,  A0, artery_params#, Q_in, P_out,

# Define neural network architecture
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),  # Inputs: (x, t, params)
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)   # Outputs: (A, Q)
        )
        #self.m  = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)

    #def forward(self, x, Q_prev, A_prev, artery_params, bc):
    def forward(self, x, artery_params, bc):#, Q_true, A_true):
        # x: [B, N, 1]
        # Q_prev, A_prev: [B, N, 1]
        # artery_params: [B, 3] -> Rd, Ru, L
        # bc: [B, 2] -> Q_in, P_out at t_n
        B, N,_ = x.shape
        artery_params = artery_params.unsqueeze(1).repeat(1, N, 1)  # [B, N, 3]
        bc = bc.repeat(1, N, 1)  # [B, N, 2]

        #input_tensor = torch.cat([x, Q_prev, A_prev, artery_params], dim=-1).float()  # [B, N, 6] # bc
        input_tensor = torch.cat([x, artery_params, bc], dim=-1).float()  # [B, N, 6] # bc
        #output_tensor = .0000001*self.net(input_tensor)
        #return output_tensor[..., 0]+self.m(Q_true).unsqueeze(-1).requires_grad_(), output_tensor[..., 1]+self.m(A_true).unsqueeze(-1).requires_grad_()  # [B, N, 2] -> Q_n, A_n
        return self.net(input_tensor)  # [B, N, 2] -> Q_n, A_n


#def compute_residual_loss_full2(x, Q_prev, A_prev, Q_pred, A_pred,
     #A0, beta, dA0dx, dTaudx, dt, nu, rho, Cv, gamma, gamma_profile, Pext=10000,  tapered=None, viscoelastic=None):
def compute_residual_loss_full(x,  Q_pred, A_pred,
     A0, Q_in_next, P_out_next, beta, dA0dx, dTaudx,  nu, rho, Cv, gamma, gamma_profile, Pext=10000,  tapered=None, viscoelastic=None):
    """

    Computes physics residuals of the 1D Navier-Stokes equations.
    
    Args:
        model: PINN model
        x: [N, 1] spatial points
        Q_prev, A_prev: [N, 1] previous state
        artery_params: [B, 3] Rd, Ru, L
        bc: [B, 2] boundary conditions Q_in, P_out
        A0: [N, 1] reference area
        dt: float, time step
        k: float, stiffness (for gamma)
        nu: viscosity
        
    Returns:
        scalar residual loss

    Full residual with tapering and viscoelastic terms.
    """

    # Mass conservation
    #dA_dt = (A_pred - A_prev) / dt
    dQ_dx = torch.autograd.grad(Q_pred, x, grad_outputs=torch.ones_like(Q_pred), create_graph=True)[0]
    #mass_residual = dA_dt + dQ_dx
    mass_residual =  dQ_dx

    # Momentum conservation
    #dQ_dt = (Q_pred - Q_prev) / dt

    #gamma = k / A0
    sqrt_ratio = torch.sqrt(A_pred / A0 + 1e-6)  # A0 from artery parameters
    pressure = Pext + beta * (sqrt_ratio - 1.0)
    #pressure = gamma * A_pred * torch.sqrt(A_pred + 1e-8)
    flux = Q_pred**2 / A_pred + pressure
    dF_dx = torch.autograd.grad(flux, x, grad_outputs=torch.ones_like(flux), create_graph=True)[0]

    # Viscosity source
    visc_coef = 2 * (gamma_profile + 2) * torch.pi * nu  # scalar

    S_visc = -visc_coef * Q_pred / (A_pred + 1e-8)

    # Tapering source
    if tapered:
        S_taper = (
            0.5 * beta * A_pred * torch.sqrt(A_pred + 1e-8) / (A0 * rho) * dA0dx
            - A_pred / rho * (torch.sqrt(A_pred / A0 + 1e-8) - 1.0) * dTaudx
        )
    else:
        S_taper = 0.0

    # Viscoelastic source: Cv * ∂²Q/∂x²
    if viscoelastic:
        d2Q_dx2 = torch.autograd.grad(dQ_dx, x, grad_outputs=torch.ones_like(dQ_dx), create_graph=True)[0]
        S_visco = Cv * d2Q_dx2
    else:
        S_visco = 0.0

    # Final residual
    #momentum_residual = dQ_dt + dF_dx - S_visc + S_taper + S_visco
    momentum_residual =  dF_dx + S_visc + S_taper + S_visco

    # Final loss

    Q0_pred = Q_pred[:, 0:1, :]
    PL_pred = pressure[-1]#gamma[-1] * A_pred[: ,-1:, :] * torch.sqrt(A_pred[:, -1:, :] + 1e-8)

    loss_bc_in = torch.mean((Q0_pred - Q_in_next)**2) # this is not valid as Q_inpred is for the next iteration
    loss_bc_out = torch.mean((PL_pred - P_out_next)**2)

    # === Residual loss ===
    mass_loss = torch.mean(mass_residual**2)
    momentum_loss = torch.mean(momentum_residual**2)

    total_loss = mass_loss + momentum_loss + loss_bc_in + loss_bc_out
    return total_loss

    #return mass_loss + momentum_loss



#def compute_loss( model, x, Q_prev, A_prev, Q_pred, A_pred,
    #Q_true, A_true, A0, beta, dA0dx, dTaudx, dt, nu, rho, Cv, gamma, gamma_profile,
    #lambda_phys=0.1, tapered=None, viscoelastic=None):
def compute_loss( x, Q_pred, A_pred, Q_true, A_true, A0, Q_in, P_out, Q_in_next, P_out_next, beta, dA0dx, dTaudx, dt, nu, rho, Cv, gamma, gamma_profile,
    lambda_phys=0.1, tapered=None, viscoelastic=None, n_params=None):
    #( model, x, Q_pred, A_pred,
                   # Q_true, A_true,  structure['A0'], Q_in, P_out, structure['beta'].unsqueeze(-1), structure['dA0dx'].unsqueeze(-1), structure['dTaudx'].unsqueeze(-1),
                    #meta['dt'],  meta['mu'], meta['rho'], structure['Cv'].unsqueeze(-1), structure['gamma'].unsqueeze(-1), meta['gamma_profile']
                     # , tapered=True, viscoelastic=False)
    N = x.shape[1]
    loss_sup = (1/N)*torch.sum((Q_pred.squeeze() - Q_true)**2 + (A_pred.squeeze() - A_true)**2)

    #loss_phys = compute_residual_loss_full( x, Q_prev, A_prev,Q_pred, A_pred,  A0, beta, dA0dx, dTaudx, dt, nu, rho, Cv, gamma, gamma_profile
                                           #, tapered=tapered, viscoelastic=viscoelastic)
    loss_phys = compute_residual_loss_full( x, Q_pred, A_pred, A0, Q_in_next, P_out_next,  beta, dA0dx, dTaudx, nu, rho, Cv, gamma, gamma_profile
                                           , tapered=tapered, viscoelastic=viscoelastic)
    # x,  Q_pred, A_pred,
    # A0, Q_in, P_out,  beta, dA0dx, dTaudx,  nu, rho, Cv, gamma, gamma_profile, Pext=10000,  tapered=None, viscoelastic=None
    # L2 regularization
    #l2_reg = (1/n_params)*sum(torch.sum(p**2) for p in model.parameters())

    return .01*loss_sup + lambda_phys * loss_phys #+ 1e-6 * l2_reg
    #return  lambda_phys * loss_phys + 1e-6 * l2_reg
# Define PDE and boundary condition loss function

def loss_function_not_used_for_now(model, x, t, Q_prev, A_prev, artery_params, bc, A0, blood ):
    """
    x, t, params: tensors of shape (N, 1) for collocation points
    A0: reference cross-sectional area (tensor of shape (N, 1) or function of x)
    U00A, U00Q: boundary conditions for A and Q at x = x_left
    UM1A, UM1Q: boundary conditions for A and Q at x = x_right
    gamma_profile: dimensionless viscosity parameter
    mu: dynamic viscosity
    rho: density
    beta: stiffness parameter
    dA0dx: derivative of A0 w.r.t. x (tensor or function of x)
    dTaudx: derivative of wall shear stress (tensor or function of x)
    tapered: whether to include tapering effects
    viscoelastic: whether to include viscoelastic effects
    Cv: viscoelastic damping coefficient
    dx: spatial step size (for viscoelastic term)
    x_left, x_right: domain boundaries
    """
    # PDE collocation points
    #x, Q_prev, A_prev,  A0, artery_params
    rd, ru, L, beta, tapered, viscoelastic, dA0dx, dTaudx, gamma_profile, Cv, U00A, U00Q = artery_params

    x = x.requires_grad_(True)
    #t = t.requires_grad_(True)
    #inputs = [L, ru, rd, t_prev]#torch.cat([x, t, params], dim=1)

    AQ = model(x, Q_prev, A_prev, artery_params, bc)
    A, Q = AQ[:, 0:1], AQ[:, 1:2]
    #x = torch.linspace(0, L, dx)
    # Compute derivatives
    # A_x = autograd.grad(A, x, grad_outputs=torch.ones_like(A), create_graph=True)[0]
    Q_x = autograd.grad(Q, x, grad_outputs=torch.ones_like(Q), create_graph=True)[0]
    A_t = autograd.grad(A, t, grad_outputs=torch.ones_like(A), create_graph=True)[0]
    Q_t = autograd.grad(Q, t, grad_outputs=torch.ones_like(Q), create_graph=True)[0] # maybe we should (Q_pred - Q_prev)/dt
    # Flux terms (aligned with MUSCL solver)
    # flux_A = Q
    flux_Q = (Q**2 / A) + (beta / (blood.rho * A0)) * A * torch.sqrt(A)

    # PDE residuals
    residual_continuity = A_t + Q_x
    flux_Q_x = autograd.grad(flux_Q, x, grad_outputs=torch.ones_like(flux_Q), create_graph=True)[0]
    residual_momentum = Q_t + flux_Q_x

    # Source terms (from MUSCL solver)
    source_Q = torch.zeros_like(Q)

    # Viscous dissipation term
    source_Q -= 2 * (gamma_profile + 2) * torch.pi * blood.mu * Q / (A * blood.rho)

    # Tapering effects (if enabled)
    if tapered and (A0 is not None) and (dA0dx is not None) and (dTaudx is not None):
        source_Q += 0.5 * beta * torch.sqrt(A) * A / (A0 * blood.rho) * dA0dx
        source_Q -= (A / blood.rho) * (torch.sqrt(A / A0) - 1.0) * dTaudx

    # Add source term to momentum residual
    residual_momentum -= source_Q

    # Viscoelastic term (if enabled)
    if viscoelastic:
        Q_xx = autograd.grad(Q_x, x, grad_outputs=torch.ones_like(Q_x), create_graph=True)[0]
        residual_momentum -= Cv * Q_xx  # Viscoelastic damping term

    # PDE loss (L2 norm of residuals)
    loss_continuity = torch.mean(residual_continuity**2)
    loss_momentum = torch.mean(residual_momentum**2)

    pde_loss = loss_continuity + loss_momentum

    if 0:
        # Boundary condition loss
        # Left boundary (x = x_left)
        idx_left = (x == x_left).nonzero(as_tuple=True)[0]
        
        if len(idx_left) > 0:
            A_left = A[idx_left]
            Q_left = Q[idx_left]
            loss_bc_left = torch.mean((A_left - U00A[idx_left])**2) + torch.mean((Q_left - U00Q[idx_left])**2)
        else:
            loss_bc_left = torch.tensor(0.0, device=x.device)

        # Right boundary (x = x_right)
        idx_right = (x == x_right).nonzero(as_tuple=True)[0]
        if len(idx_right) > 0:
            A_right = A[idx_right]
            Q_right = Q[idx_right]
            loss_bc_right = torch.mean((A_right - UM1A[idx_right])**2) + torch.mean((Q_right - UM1Q[idx_right])**2)
        else:
            loss_bc_right = torch.tensor(0.0, device=x.device)

    # Total boundary loss
    #loss_bc = loss_bc_left + loss_bc_right

    # Total loss
    total_loss = pde_loss #+ loss_bc

    return total_loss

def train(model, dataset, epochs=100, batch_size=1, lr=1e-4):
    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Set random seed for reproducibility
    print("Training PINN model...")
    n = 100  #  every nth item
    n_samples = int(len(dataset)/n)
    #dataset = loader.dataset  # assuming loader is your original DataLoader
    subset_indices = torch.randint(0, len(dataset)-1, (n_samples,))#list(range(0, len(dataset), n))
    subset = torch.utils.data.Subset(dataset, subset_indices)

    seed = 42
    generator = torch.Generator().manual_seed(seed)
    total_size = len(subset)
    train_ratio = 0.9
    val_ratio = 0.05
    test_ratio =   1- train_ratio - val_ratio
    # Compute sizes
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size  # Make sure it sums to total_size
    # Split the dataset
    from torch.utils.data import random_split
    train_dataset, val_dataset, test_dataset = random_split(
        subset,
        [train_size, val_size, test_size],
        generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    softplus = nn.Softplus()
    #from itertools import islice
    

    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=torch.utils.data.SubsetRandomSampler(subset_indices))
    best_val_loss = torch.inf
    model_dir = "pinn_model"
    model_name = 'pinn_solver_noprev_signals.pth'
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for epoch in range(epochs):
        total_loss = 0.0
        idx = -1
        #for x, Q_prev, A_prev, Q_true, A_true,  meta, Q_in, P_out, structure in train_loader:
        for x,  Q_true, A_true,  meta, Q_in, P_out, Q_in_next, P_out_next, structure in train_loader:
            idx+=1
            if idx==len(train_loader)-1:
                pass
 
            optimizer.zero_grad()
            x.unsqueeze_(-1)  # Ensure x is [B, N, 1]
            x.requires_grad_(True)

            # Expand BCs to match batch format
            bc = torch.cat([Q_in, P_out])#, dim=1)
            
            artery_params = torch.tensor([meta['Ru'], meta['Rd'], meta['L']]).unsqueeze(0)
            Q_pred_A_pred = model(x, artery_params, bc)
            #Q_pred, A_pred = model(x, artery_params, bc, Q_true, A_true)

            Q_pred, A_pred = Q_pred_A_pred[...,0:1], softplus(Q_pred_A_pred[..., 1:2])+ 1e-6
            #loss = compute_loss( model, x, Q_prev, A_prev, Q_pred, A_pred,
                    #Q_true, A_true, structure['A0'], structure['beta'], structure['dA0dx'], structure['dTaudx'],
                    #meta['dt'],  meta['mu'], meta['rho'], structure['Cv'], structure['gamma'], meta['gamma_profile']
                      #, tapered=True, viscoelastic=False)
            loss = compute_loss(x, Q_pred, A_pred,
                    Q_true, A_true,  structure['A0'].unsqueeze(-1), Q_in, P_out, Q_in_next, P_out_next,  structure['beta'].unsqueeze(-1), structure['dA0dx'].unsqueeze(-1), structure['dTaudx'].unsqueeze(-1),
                    meta['dt'],  meta['mu'], meta['rho'], structure['Cv'].unsqueeze(-1), structure['gamma'].unsqueeze(-1), meta['gamma_profile']
                      , tapered=True, viscoelastic=False,n_params=n_params)
            #( model, x, Q_pred, A_pred, Q_true, A_true, A0, Q_in, P_out, beta, dA0dx, dTaudx, dt, nu, rho, Cv, gamma, gamma_profile,
            #lambda_phys=0.1, tapered=None, viscoelastic=None)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 1 == 0:
            print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")
        # Validation step
        if epoch % 4 == 0:
            val_loss = 0.0
            #with torch.no_grad():
            for x,  Q_true, A_true,  meta, Q_in, P_out, Q_in_next, P_out_next, structure in val_loader:
                idx+=1
                if idx==len(train_loader)-1:
                    pass
    
                optimizer.zero_grad()
                x.unsqueeze_(-1)  # Ensure x is [B, N, 1]
                x.requires_grad_(True)

                # Expand BCs to match batch format
                bc = torch.cat([Q_in, P_out])#, dim=1)
                
                artery_params = torch.tensor([meta['Ru'], meta['Rd'], meta['L']]).unsqueeze(0)
                #Q_pred_A_pred = model(x, Q_prev, A_prev, artery_params, bc)
                Q_pred_A_pred = model(x, artery_params, bc)

                Q_pred, A_pred = Q_pred_A_pred[...,0:1], softplus(Q_pred_A_pred[..., 1:2])+ 1e-6
                #loss = compute_loss( model, x, Q_prev, A_prev, Q_pred, A_pred,
                        #Q_true, A_true, structure['A0'], structure['beta'], structure['dA0dx'], structure['dTaudx'],
                        #meta['dt'],  meta['mu'], meta['rho'], structure['Cv'], structure['gamma'], meta['gamma_profile']
                        #, tapered=True, viscoelastic=False)
                loss = compute_loss( x, Q_pred, A_pred,
                        Q_true, A_true,  structure['A0'].unsqueeze(-1), Q_in, P_out, Q_in_next, P_out_next,  structure['beta'].unsqueeze(-1), structure['dA0dx'].unsqueeze(-1), structure['dTaudx'].unsqueeze(-1),
                        meta['dt'],  meta['mu'], meta['rho'], structure['Cv'].unsqueeze(-1), structure['gamma'].unsqueeze(-1), meta['gamma_profile']
                        , tapered=True, viscoelastic=False, n_params=n_params)
                val_loss += loss.item()
            print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
            if val_loss/len(val_loader)<best_val_loss:
                best_val_loss = val_loss/len(val_loader)
                # Save model state
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                print(f"Saving model with best validation loss: {best_val_loss:.4f}")
                torch.save(model.state_dict(), os.path.join(model_dir, model_name))


        # print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model
    model = PINN()
    dataset = ArteryDatasetOnDemand("network_state_2")
    #loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    #x, Q_prev, A_prev, Q_true, A_true,  A0, artery_params = dataset[(3, 1, 'abdominal_aorta_III')]
    if 1:
        model = train(
            model,
            dataset,
            epochs=1000,
            batch_size=1,
            lr=1e-4)