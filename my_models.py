import torch
import torch.autograd as autograd
#from vessel import Vessel
from torch.utils.data import Dataset
import os, glob
# import json
import numpy as np
# from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn, autograd, Tensor
from torch.nn import functional as F
from torch.nn import L1Loss
import networkx as nx
from tqdm.auto import tqdm
from vessel import Vessel, Blood
import yaml
import matplotlib.pyplot as plt #
from test_batch_builder import build_batch, process_batch, pressure, newtone
from scipy.interpolate import interp1d
import math

class Regressor3(nn.Module):
    def __init__(self, width=128, depth=8, skip=True):
        super().__init__()
        layers = []
        
        # Initial convolution: 1 → width
        layers.append(nn.Conv1d(1, width, kernel_size=7, padding=3))  # better: padding=3 for k=7
        # or keep padding=1 if you want, but padding=(k-1)//2 is standard
        #layers.append(nn.Tanh())
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(depth):
            if i % 2 == 0:
                layers.append(nn.Conv1d(width, width, kernel_size=5, padding=2))
            else:
                layers.append(nn.Conv1d(width, width, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
        
        # Final 1x1 convolution: width → 1
        layers.append(nn.Conv1d(width, 1, kernel_size=1, padding=0))
        
        self.net = nn.Sequential(*layers)
        self.skip = skip
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
        # x: (batch_size, seq_len) → (batch_size, 1, seq_len)
        input_for_skip = x
        x = x.unsqueeze(1)                    # Now: (B, 1, L)

        out = self.net(x)                     # → (B, 1, L)
        out = out.squeeze(1)                  # → (B, L)

        if self.skip:
            # This is now guaranteed to work: both are (B, L)
            out = out + input_for_skip * 0.1   # scaled residual

        return out

class Regressor(nn.Module):
    def __init__(self, width=128, depth=6, skip=True):
        super().__init__()
        layers = []
        layers.append(nn.Conv1d(1, width, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        
        for _ in range(depth):
            layers.append(nn.Conv1d(width, width, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            
        # Final layer: back to 1 channel
        layers.append(nn.Conv1d(width, 1, kernel_size=1))   # 1×1 conv
        
        self.net = nn.Sequential(*layers)
        
        # Optional: residual connection (very powerful here)
        self.skip = skip

        
class Regressor2(nn.Module):

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
    

    def forward(self, x):
        # x shape: (batch, 100) → (batch, 1, 100)
        x = x.unsqueeze(1)
        out = self.net(x)
        out = out.squeeze(1)
        
        if self.skip:
            out = out + x.squeeze(1) * 0.1   # weak residual (because output ≈ small correction)
        return out



def loss_fn(A, Q, A_pred, Q_pred, f_continuity, f_momentum, fc_bif, fc_conj, f_inlet,  f_A_leaf, f_u_leaf, epoch, detailed_loss=True):
    """
    A: (b, 1)
    Q: (b, 1)
    """
    loss = L1Loss(reduction='sum')

    if 0:
    
        leaf_A_loss = F.mse_loss(f_A_leaf, torch.zeros_like(f_A_leaf)) if f_A_leaf is not None else torch.tensor(0.0)
        leaf_u_loss = F.mse_loss(f_u_leaf, torch.zeros_like(f_u_leaf)) if f_u_leaf is not None else torch.tensor(0.0)
        inlet_loss = F.mse_loss(f_inlet, torch.zeros_like(f_inlet)) if f_inlet is not None else torch.tensor(0.0)
        A_loss = F.mse_loss(A_pred, A) if A is not None else torch.tensor(0.0)
        Q_loss = F.mse_loss(Q_pred, Q) if Q is not None else torch.tensor(0.0)
        f_continuity_loss = F.mse_loss(f_continuity, torch.zeros_like(f_continuity)) if f_continuity is not None else torch.tensor(0.0)
        f_momentum_loss = F.mse_loss(f_momentum, torch.zeros_like(f_momentum)) if f_momentum is not None else torch.tensor(0.0)
        bif_loss = F.mse_loss(fc_bif, torch.zeros_like(fc_bif)) if fc_bif is not None else torch.tensor(0.0)
        conj_loss = F.mse_loss(fc_conj, torch.zeros_like(fc_conj)) if fc_conj is not None else torch.tensor(0.0)
    else:

        leaf_A_loss = loss(f_A_leaf, torch.zeros_like(f_A_leaf)) if f_A_leaf is not None else torch.tensor(0.0)
        leaf_u_loss = loss(f_u_leaf, torch.zeros_like(f_u_leaf)) if f_u_leaf is not None else torch.tensor(0.0)
        inlet_loss = loss(f_inlet, torch.zeros_like(f_inlet)) if f_inlet is not None else torch.tensor(0.0)
        A_loss = loss(A_pred, A) if A is not None else torch.tensor(0.0)
        Q_loss = loss(Q_pred, Q) if Q is not None else torch.tensor(0.0)
        f_continuity_loss = loss(f_continuity, torch.zeros_like(f_continuity)) if f_continuity is not None else torch.tensor(0.0)
        f_momentum_loss = loss(f_momentum, torch.zeros_like(f_momentum)) if f_momentum is not None else torch.tensor(0.0)
        bif_loss = loss(fc_bif, torch.zeros_like(fc_bif)) if fc_bif is not None else torch.tensor(0.0)
        conj_loss = loss(fc_conj, torch.zeros_like(fc_conj)) if fc_conj is not None else torch.tensor(0.0)
    if epoch is None:
        loss = leaf_A_loss + leaf_u_loss + A_loss + Q_loss + f_continuity_loss + f_momentum_loss + bif_loss + conj_loss+inlet_loss
        if detailed_loss:
            return loss, {'leaf_A_loss':leaf_A_loss,'leaf_u_loss':  leaf_u_loss,
                           'A_loss': A_loss, 'Q_loss': Q_loss, 'f_continuity_loss': f_continuity_loss,
                             'f_momentum_loss': f_momentum_loss, 'bif_loss': bif_loss,
                               'conj_loss': conj_loss, 'inlet_loss': inlet_loss}
        return loss
    a_a, a_q = 1, 1
    #, a_leaf_a, a_leaf_q, a_conti, a_momentum, a_bif, a_conj, a_inlet, , 10**8, 
    if epoch > -1 and 1:
        
        #loss = (10**8)*leaf_A_loss + leaf_u_loss + .1*A_loss + .1*Q_loss + f_continuity_loss + f_momentum_loss + (10**2)*bif_loss + (10**2)*conj_loss+(10**8)*inlet_loss
        loss = leaf_A_loss + leaf_u_loss + A_loss + Q_loss + f_continuity_loss + f_momentum_loss/1000 + bif_loss + conj_loss + inlet_loss
    else:
        # loss = 10000*A_loss + 10000*Q_loss +(10**2)*bif_loss +(10**2)*conj_loss
        
        loss =   (a_a*A_loss +  a_q*Q_loss)/(a_a + a_q)+ inlet_loss #(10**2)*bif_loss + (10**2)*conj_loss#
        if epoch%200==0 and 0:
            print(f"A loss={A_loss},    Q_loss={Q_loss}")
        #loss =    Q_loss #(10**2)*bif_loss + (10**2)*conj_loss#
    return loss, []

def get_U_bif(u1, u2, u3, A1, A2, A3):
    """    return np.array([
        v1.u[-1],
        v2.u[0],
        v3.u[0],
        np.sqrt(np.sqrt(v1.A[-1])),
        np.sqrt(np.sqrt(v2.A[0])),
        np.sqrt(np.sqrt(v3.A[0]))
    ])"""
    return torch.stack((
        u1,
        u2,
        u3,
        torch.sqrt(torch.sqrt(A1)),
        torch.sqrt(torch.sqrt(A2)),
        torch.sqrt(torch.sqrt(A3))
    ), dim=-1)

def bif_fc(U, A0_1, A0_2, A0_3, beta1, beta2, beta3):
    """    return np.array([
        # U[0] + 4 * k[0] * U[3] - W[0],
        # U[1] - 4 * k[1] * U[4] - W[1],
        # U[2] - 4 * k[2] * U[5] - W[2],
        # in case of nn prediction the first three equations are not needed
        U[0] * U[3] ** 4 - U[1] * U[4] ** 4 - U[2] * U[5] ** 4,
        v1.beta[-1] * (U[3] ** 2 / np.sqrt(v1.A0[-1]) - 1) - v2.beta[0] * (U[4] ** 2 / np.sqrt(v2.A0[0]) - 1),
        v1.beta[-1] * (U[3] ** 2 / np.sqrt(v1.A0[-1]) - 1) - v3.beta[0] * (U[5] ** 2 / np.sqrt(v3.A0[0]) - 1),
    ])"""
    return torch.stack((U[:, 0] * U[:, 3] ** 4 - U[:, 1] * U[:, 4] ** 4 - U[:, 2] * U[:, 5] ** 4,
        beta1 * (U[:, 3] ** 2 / torch.sqrt(A0_1) - 1) - beta2 * (U[:, 4] ** 2 / torch.sqrt(A0_2) - 1),
        beta1 * (U[:, 3] ** 2 / torch.sqrt(A0_1) - 1) - beta3 * (U[:, 5] ** 2 / torch.sqrt(A0_3) - 1)), dim=-1)

def get_U_conj(u1, u2, A1, A2):
    """    return np.array([
        v1.u[-1],
        v2.u[0],
        np.sqrt(np.sqrt(v1.A[-1])),
        np.sqrt(np.sqrt(v2.A[0]))
    ])"""
    return torch.stack((
        u1,
        u2,
        torch.sqrt(torch.sqrt(A1)),
        torch.sqrt(torch.sqrt(A2))), dim=-1)

def conjunction_fc(u1, u2, A1, A2, beta1, beta2, A0_1, A0_2, rho):
    """
    Conjunction function for two vessels.


    U = get_U_conj(v1, v2)
    k = (
        np.sqrt(1.5 * v1.gamma[-1]), 
        np.sqrt(1.5 * v2.gamma[0]))
    W = (
        U[0] + 4 * k[0] * U[2],
        U[1] - 4 * k[1] * U[3]
    )
    return np.array([
        U[0] + 4 * k[0] * U[2] - W[0],
        U[1] - 4 * k[1] * U[3] - W[1],
        # in case of nn prediction the first two equations are not needed therefore W and k (and gamma)
        U[0] * U[2]**4 - U[1] * U[3]**4,
        0.5 * rho * U[0]**2 + v1.beta[-1] * (U[2]**2 / np.sqrt(v1.A0[-1]) - 1) -
        (0.5 * rho * U[1]**2 + v2.beta[0] * (U[3]**2 / np.sqrt(v2.A0[0]) - 1))
    ])
        """
    U = get_U_conj(u1, u2, A1, A2)
    #k = (
     #   torch.sqrt(1.5 * gamma1[:, -1]), 
      #  torch.sqrt(1.5 * gamma2[:, 0]))
    return torch.stack((
        U[:, 0] * U[:, 2]**4 - U[:, 1] * U[:, 3]**4,
        0.5 * rho * U[:, 0]**2 + beta1 * (U[:, 2]**2 / torch.sqrt(A0_1) - 1) - 
        0.5 * rho * U[:, 1]**2 - beta2 * (U[:, 3]**2 / torch.sqrt(A0_2) - 1)
        ), dim=-1)

def calc_grad(y, x) -> Tensor:
    return autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        #retain_graph=True,
        #only_inputs=True,
        #allow_unused=True
    )[0]
 #autograd.grad(Q, x, grad_outputs=torch.ones_like(Q), create_graph=True)[0]


def calculate_loss(A_pred, Q_pred, A_true, Q_true, x, t, inputs, extrapolators, f_interp, epoch=None,
        rho = None, gamma_profile=None, mu=None,  index_mapping=None):

        #sample = [x, t, artery_idx, succ1, succ2, L, Rp, Rd, A0[x_idx],
                    #dA0dx[x_idx], dTaudx[x_idx], beta[x_idx], gamma[x_idx]]
        #inputs_non_std = h.clone().detach()
        #x = inputs[:, 0]
        #t = inputs[:, 1]
        A_t = calc_grad(A_pred, t)#[:, 1]  # Gradient of A with respect to t
        # Calculate gradient of Q with respect to x     
        Q_x = calc_grad(Q_pred, x)#[:, 0]  # Gradient of Q with respect to x
        
        # Continuity equation residual
        f_continuity = A_t + Q_x
        
        # Calculate derivatives for momentum equation
        Q_t = calc_grad(Q_pred, t)#[:, 1]  # Gradient of Q with respect to t    
        
        # Term 1: ∂/∂x (Q²/A)
        Q2_over_A = Q_pred**2 / A_pred
        d_Q2_over_A_dx = calc_grad(Q2_over_A, x)#[:, 0]  # Gradient of Q²/A with respect to x
        
        # Term 2: ∂/∂x (β/ρ A^(3/2))
        beta = inputs[:, 11]  # Pressure-area constant
        if A_pred.any()<0:
            A32 = A_pred ** (3/2)
            d_A32_dx = calc_grad(A32, x)#[:, 0]  # Gradient of A^(3/2) with respect to x

            term2 = (beta / rho) * d_A32_dx
        else:
            term2 = 0
        
        # Viscous dissipation term
        viscous_term = -2 * (gamma_profile + 2) * torch.pi * mu * Q_pred / (A_pred * rho+1e-6)
        
        # Tapering (Area) term
        A0 = inputs[:, 8]  # Reference area
        dA0dx = inputs[:, 9]  # Gradient of reference area
        dTaudx = inputs[:, 10]  # Gradient of shear stress
        if not A_pred.any()<0:
            tapering_area_term = (beta / (2 * rho)) * (A32 / A0) * dA0dx
        else:
            tapering_area_term = 0
        
        # Tapering (Shear) term
        if not A_pred.any()<0:
            sqrt_A_over_A0 = torch.sqrt(A_pred / A0)
            tapering_shear_term = -(A_pred / rho) * (sqrt_A_over_A0 - 1) * dTaudx
        else:
            tapering_shear_term = 0
        
        # Viscoelasticity term
        #Q_xx = calc_grad(Q_x, inputs)[:,0]  # Second derivative of Q with respect to x
        # Viscoelastic term: Cv * ∂²Q/∂x²
        # Note: Cv is a learnable parameter, can be adjusted based on the model's
        # viscoelastic properties.
        viscoelastic_term = 0#self.Cv * Q_xx if self.Cv else 0
        
        # Momentum equation residual
        f_momentum = (
            Q_t 
            + d_Q2_over_A_dx 
            + term2 
            - viscous_term 
            - tapering_area_term 
            - tapering_shear_term 
            - viscoelastic_term
        )

        #conj_idx, conj_conns_idx, bif_idx, bif_conn1_idx,  bif_conn2_idx = extract_aligned_conn_data(data=inputs_non_std, n_arteries=self.n_arteries)
        
        rest_idx, leaf_idx, inlet_flow_samples, inlet, conj_idx, conj_conns_idx, bif_idx, bif_conn1_idx,  bif_conn2_idx = process_batch(inputs, f_interp)#, n_arteries=self.n_arteries)
        U_pred = Q_pred / A_pred

        if len(leaf_idx) > 0:
            # gamma = inputs[:, 12]
            # Cc , R1, R2 = [], [], []
            #for l in leaf_idx:
            idxs = inputs[leaf_idx, 2].long()
            A_leaf_true, u_leaf_true = [], []
            for idx, l in zip(idxs, leaf_idx):
                A_interp = extrapolators[idx.item()]['A'] 
                Q_interp = extrapolators[idx.item()]['Q'] 
                A = A_interp(t[l].item()).item()
                Q = Q_interp(t[l].item()).item()
                A_leaf_true.append(A), u_leaf_true.append(Q/A)
            f_A_leaf = torch.tensor(A_leaf_true) - A_pred[leaf_idx]
            f_u_leaf = torch.tensor(u_leaf_true) - U_pred[leaf_idx]
        else:
            f_A_leaf, f_u_leaf = None, None
        
        
        if len(inlet) > 0:
            f_inlet = Q_pred[inlet] - torch.from_numpy(inlet_flow_samples)
        else:
            f_inlet = None
        
        
        if len(bif_idx) > 0:
            
            U = get_U_bif(U_pred[bif_idx], U_pred[bif_conn1_idx], U_pred[bif_conn2_idx],
                        A_pred[bif_idx], A_pred[bif_conn1_idx], A_pred[bif_conn2_idx])#u1, u2, u3, A1, A2, A3)
            fc_bif = bif_fc(U, A0[bif_idx], A0[bif_conn1_idx], A0[bif_conn2_idx], beta[bif_idx], beta[bif_conn1_idx], beta[bif_conn2_idx])
        else:
            fc_bif = None
        #get_U_conj(u1, u2, A1, A2)
        if len(conj_idx) > 0:
            fc_conj = conjunction_fc(U_pred[conj_idx], U_pred[conj_conns_idx], A_pred[conj_idx], A_pred[conj_conns_idx],
                        beta[conj_idx], beta[conj_conns_idx], A0[conj_idx], A0[conj_conns_idx], rho)

        else:
            fc_conj = None
        #A, Q = None, None
        loss, loss_dict = loss_fn(A_true, Q_true, A_pred, Q_pred, f_continuity, f_momentum, fc_bif, fc_conj, f_inlet, f_A_leaf, f_u_leaf, epoch)
        if loss!=loss:
            pass
        return loss, loss_dict


class Sine(nn.Module):
    def __init__(self, omega_0=1.0):
        super().__init__()
        self.omega_0 = omega_0
    def forward(self, x):
        return torch.sin(self.omega_0 * x)

class SIRENLayer(nn.Module):
    def __init__(self, in_dim, out_dim, omega_0=1.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_dim, out_dim)
        # self.init_weights()
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1 / self.linear.in_features
            else:
                bound = math.sqrt(6 / self.linear.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIREN(nn.Module):

    def __init__(self, in_dim=None, hidden_dim=None, out_dim=None, num_layers=None, omega_0=70,**kwargs):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(1060.0))  # Density
        self.mu = nn.Parameter(torch.tensor(0.004))   # Viscosity
        self.gamma_profile = nn.Parameter(torch.tensor(2.0))  # Profile parameter
        self.Cv = nn.Parameter(torch.tensor(0.0))   # Viscoelasticity coefficient

        self.out_dim=out_dim
        self.mu_A = kwargs['mu_A']
        self.sigma_A = kwargs['sigma_A']
        self.mu_Q = kwargs['mu_Q']
        self.sigma_Q = kwargs['sigma_Q']
        self.max_A = kwargs['max_A']
        self.max_Q = kwargs['max_Q']
        # print(omega_0)
        #omega_0=1000
        self.net = []
        self.net.append(SIRENLayer(in_dim, hidden_dim, omega_0, is_first=True))
        for _ in range(num_layers-2):
            self.net.append(SIRENLayer(hidden_dim, hidden_dim, omega_0=1))
        
        
        
        

        if out_dim == 1:
            self.head_A = nn.Linear(hidden_dim, out_dim)  # for log(A) or A
            self.head_Q = nn.Linear(hidden_dim, out_dim)  # for log(Q) or Q
        else:
            self.net.append(nn.Linear(hidden_dim, out_dim))  # output layer is linear
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        hidden_output = self.net(x)
        if self.out_dim==1:

            A_pred = self.head_A(hidden_output).squeeze()
            Q_pred = self.head_Q(hidden_output).squeeze()
        else:
        
            A_pred = hidden_output[:, 0]#.abs()#F.softplus(hidden_output[:, 0], beta=5) #self.softplus(hidden_output[:, 0], beta=5)#+ 1e-8  # Ensure A is positive, add small constant to avoid division by zero
            Q_pred = hidden_output[:, 1]
        return  A_pred, Q_pred

class FfnBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inter_dim = 1 * dim
        self.fc1 = nn.Linear(dim, inter_dim)
        self.fc2 = nn.Linear(inter_dim, dim)
        self.act_fn = torch.sin#nn.Tanh()
        #self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        #x0 = x
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.act_fn(x)
        #x = self.dropout(x)
        return x# + x0

class Pinn(nn.Module):
    """
    `forward`: returns a tensor of shape (D, 2), where D is the number of
    data points, and the 2nd dim. is the predicted values of A and Q.
    """
    def __init__(self, n_arteries, in_dim, rho, **kwargs):#min_x: int, max_x: int, A0: float, dA0dx: float, dTaudx: float):
        super().__init__()

        self.n_arteries = n_arteries
        #self.rho = rho  # Density, can be a learnable parameter if needed
        #self.MIN_X = min_x
        #self.MAX_X = max_x
        
        # Physical parameters
        #self.A0 = A0  # Reference cross-sectional area
        #self.dA0dx = dA0dx  # Gradient of reference area
        #self.dTaudx = dTaudx  # Gradient of shear stress
        
        # Physical constants (can be made learnable if needed)
        #self.beta = nn.Parameter(torch.tensor(1.0))  # Pressure-area constant
        self.rho = nn.Parameter(torch.tensor(1060.0))  # Density
        self.mu = nn.Parameter(torch.tensor(0.004))   # Viscosity
        self.gamma_profile = nn.Parameter(torch.tensor(2.0))  # Profile parameter
        self.Cv = nn.Parameter(torch.tensor(0.0))   # Viscoelasticity coefficient
        ###
        self.mu_A = kwargs['mu_A']
        self.sigma_A = kwargs['sigma_A']
        self.mu_Q = kwargs['mu_Q']
        self.sigma_Q = kwargs['sigma_Q']
        self.max_A = kwargs['max_A']
        self.max_Q = kwargs['max_Q']

        # Build FFN network
        self.hidden_dim = 128
        self.num_blocks = 2
        inputs_dim = in_dim  # x and t
        self.first_map = nn.Linear(inputs_dim, self.hidden_dim)  # Inputs: x, t
        self.activation = nn.Tanh()#torch.sin#
        self.last_map = nn.Linear(self.hidden_dim, 2)   # Outputs: A, Q
        self.ffn_blocks = nn.ModuleList([
            FfnBlock(self.hidden_dim) for _ in range(self.num_blocks)
        ])
        self.softplus = nn.Softplus()

        #self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def ffn(self, inputs: Tensor) -> Tensor:
        x = self.first_map(inputs)
        x = self.activation(x)
        for blk in self.ffn_blocks:
            x = blk(x)
        #x = self.activation(x)
        x = self.last_map(x)
        return x

   
    def forward(
        self,
        inputs: Tensor,
        A_pred: Tensor = None,
        Q_pred: Tensor = None,
        n_arteries=None):
        """
        [x/L, tt/T, artery_index/n_arteries, conn_0/n_arteries, conn_1/n_arteries, L, Rp, Rd, A0, dA0dx, dTaudx]
        All shapes are (b,)

        inputs: x, t
        labels: A, Q
        instead of conn_0 and conn_1 use the distance 
        add the inflow condition
        how to deal with the wk3
        add some data?
        compare results with simulation
        rechek residuals, bcs ...
        important i am deviding the beta by 10**5 check this especially in the loss function
        """
 
        
        x = inputs[:, 0:1]  # Spatial coordinate (must be used!)
    
        t = inputs[:, 1:2]  # Time (must be used!)
        #segment_id = inputs[:, 2:3]  # Segment ID
        # Concatenate and process
        h = torch.cat([x, t, inputs[:, 2:]], dim=1)
        
        #h= h.clone().detach().requires_grad_(True)
        
        #h = scale_input_batch(h, n_arteries=n_arteries, scale_last='linear')
        hidden_output = self.ffn(h)
        
        A_pred = hidden_output[:, 0]#.abs()#F.softplus(hidden_output[:, 0], beta=5) #self.softplus(hidden_output[:, 0], beta=5)#+ 1e-8  # Ensure A is positive, add small constant to avoid division by zero
        Q_pred = hidden_output[:, 1]

        #preds = torch.stack([A_pred, Q_pred], dim=1)
        
        return A_pred, Q_pred

class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, num_features=128, sigma=10.0):
        super().__init__()


        self.B = nn.Parameter(
            sigma * torch.randn((num_features, 2)), requires_grad=False
        )
        self.register_buffer("B", B)  # fixed, not learnable
    def forward(self, x):
        # x shape: [N, in_dim]
        x_proj = 2 * torch.pi * x @ self.B.T  # [N, num_features]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PINN_FourierMLP(nn.Module):
    def __init__(self, in_dim=None, hidden_dim=128, num_layers=6,
                   out_dim=1, groups=(32,16,8), sigmas=(1.0,5.0,10.0), 
                   mu_A=None, sigma_A=None,mu_Q=None, sigma_Q=None, max_Q = None, max_A=None,**kwargs):
        super().__init__()
        
        self.rho = nn.Parameter(torch.tensor(1060.0))  # Density
        self.mu = nn.Parameter(torch.tensor(0.004))   # Viscosity
        self.gamma_profile = nn.Parameter(torch.tensor(2.0))  # Profile parameter
        self.Cv = nn.Parameter(torch.tensor(0.0))   # Viscoelasticity coefficient

        self.mu_A = mu_A
        self.sigma_A = sigma_A
        self.mu_Q =mu_Q 
        self.sigma_Q = sigma_Q 
        self.max_A =max_A
        self.max_Q = max_Q 

        self.ff = MultiScaleFF(in_dim=2, groups=groups, sigmas=sigmas) #FourierFeatures(2, num_features, sigma)
        self.sine = Sine()
        ff_dim = 2*self.ff.B.shape[0]# 2 * num_features
        extra_dim = in_dim-2
        input_dim = ff_dim + extra_dim 
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(self.sine) #nn.Tanh()
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(self.sine) #nn.Tanh()
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)


        # separate heads
        #self.head_A = nn.Linear(hidden_dim, out_dim)  # for log(A) or A
        #self.head_Q = nn.Linear(hidden_dim, out_dim)  # for log(Q) or Q

    def forward(self, x):
        x_ff = self.ff(x[:, 0:2])
        x_all =  torch.cat([x_ff, x[:, 2:]], dim=-1)
        hidden_output = self.net(x_all)
                # raw outputs (normalized space)
        #A_pred = self.head_A(hidden_output)
        #Q_pred = self.head_Q(hidden_output)
        #A_pred = self.sigma_A * A_pred + self.mu_A
        #Q_pred = self.sigma_Q * Q_pred + self.mu_Q
        A_pred = hidden_output[:, 0]##F.softplus(hidden_output[:, 0], beta=5)#.abs()# #self.softplus(hidden_output[:, 0], beta=5)#+ 1e-8  # Ensure A is positive, add small constant to avoid division by zero
        Q_pred = hidden_output[:, 1]
        return  A_pred.squeeze(), Q_pred.squeeze()

class MultiScaleFF(nn.Module):
    def __init__(self, in_dim=2, groups=(128, 64, 32), sigmas=(1.0,5.0,10.0)):
        super().__init__()
        B_parts = []
        #self.groups = groups
        #self.sigmas = sigmas
        for g, s in zip(groups, sigmas):
            B_parts.append(s * torch.randn((g, in_dim)))
        B = torch.cat(B_parts, dim=0)   # [sum(g), in_dim]
        self.register_buffer("B", B) 
        self.B = nn.Parameter(B, requires_grad=False)
    def forward(self, x):
        x_proj = 2*math.pi * x @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
    
class ProbabilisticFlowNet(nn.Module):
    def __init__(self, input_dim=100):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),    # per-batch normalization on features
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mu_head        = nn.Linear(64, 1)
        self.logsigma_head  = nn.Linear(64, 1)   # can be shared or separate

    def forward(self, x):
        h = self.backbone(x)
        return self.mu_head(h).squeeze(-1), self.logsigma_head(h).squeeze(-1)

class FlowPredictorUNet1D(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, **kwargs):
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(base_filters, base_filters, 5, padding=2),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = nn.Sequential(
            nn.Conv1d(base_filters, base_filters*2, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(base_filters*2, base_filters*2, 5, padding=2),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool1d(2)

        # -------------------------
        # Bottleneck
        # -------------------------
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_filters*2, base_filters*4, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(base_filters*4, base_filters*4, 5, padding=2),
            nn.ReLU()
        )

        # -------------------------
        # Decoder
        # -------------------------
        self.up1 = nn.ConvTranspose1d(base_filters*4, base_filters*2, 2, stride=2)

        # Input channels = up1 out (64) + enc2 out (64) = 128
        self.dec1 = nn.Sequential(
            nn.Conv1d(base_filters*4, base_filters*2, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(base_filters*2, base_filters*2, 5, padding=2),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose1d(base_filters*2, base_filters, 2, stride=2)

        # Input channels = up2 out (32) + enc1 out (32) = 64
        self.dec2 = nn.Sequential(
            nn.Conv1d(base_filters*2, base_filters, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(base_filters, base_filters, 5, padding=2),
            nn.ReLU()
        )

        # -------------------------
        # Heads
        # -------------------------
        self.out_mu = nn.Conv1d(base_filters, 1, 1)
        self.out_log_sigma = nn.Conv1d(base_filters, 1, 1)
        
    def center_crop_to_match(self, tensor, target_tensor):
        _, _, L = tensor.shape
        _, _, Lt = target_tensor.shape
        if L == Lt:
            return tensor
        # crop to match center
        diff = L - Lt
        start = diff // 2
        end = start + Lt
        return tensor[:, :, start:end]

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        up1 = self.up1(b)                   # (batch, 64, T/2)
        e2_cropped = self.center_crop_to_match(e2, up1)
        d1 = self.dec1(torch.cat([up1, e2_cropped], dim=1))

        up2 = self.up2(d1)                  # (batch, 32, T)
        e1_cropped = self.center_crop_to_match(e1, up2)
        d2 = self.dec2(torch.cat([up2, e1_cropped], dim=1))

        mu = self.out_mu(d2)
        log_sigma = self.out_log_sigma(d2)
        return mu, log_sigma
    
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)

class FlowPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.blocks = nn.Sequential(
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32)
        )
        self.mu_head = nn.Conv1d(32, 1, kernel_size=1)
        self.sigma_head = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x):
        z = self.inp(x)
        z = self.blocks(z)
        mu = self.mu_head(z)
        log_sigma = self.sigma_head(z)
        return mu, log_sigma
