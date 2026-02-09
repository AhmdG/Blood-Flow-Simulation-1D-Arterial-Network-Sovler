import torch
import torch.nn as nn
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PINN class
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),  # Input: [x, t]
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)  # Output: u(x, t)
        )
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# PDE parameters
c = 0.5  # Advection velocity (m/s)
nu = 4e-6  # Kinematic viscosity (m^2/s)
u0 = 0.1  # Inlet velocity amplitude (m/s)
f = 1.0  # Frequency of pulsatile flow (Hz)
L = 0.1  # Artery length (m)
T = 1.0  # Time duration (s)

# PDE residual
def pde_residual(model, x, t):
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = model(x, t)
    
    # Compute derivatives using automatic differentiation
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    # PDE: du/dt + c * du/dx = nu * d^2u/dx^2
    residual = u_t + c * u_x - nu * u_xx
    return residual

# Loss function
def compute_loss(model, x_pde, t_pde, x_bc_in, t_bc_in, x_bc_out, t_bc_out, x_ic, t_ic):
    # PDE loss
    pde_res = pde_residual(model, x_pde, t_pde)
    pde_loss = torch.mean(pde_res**2)
    
    # Boundary condition at x=0 (pulsatile inlet)
    u_bc_in = model(x_bc_in, t_bc_in)
    u_target_in = u0 * (1 + torch.sin(2 * np.pi * f * t_bc_in))
    bc_in_loss = torch.mean((u_bc_in - u_target_in)**2)
    
    # Boundary condition at x=L (zero-gradient)
    x_bc_out = x_bc_out.requires_grad_(True)
    u_out = model(x_bc_out, t_bc_out)
    u_x_out = torch.autograd.grad(u_out, x_bc_out, grad_outputs=torch.ones_like(u_out), create_graph=True)[0]
    bc_out_loss = torch.mean(u_x_out**2)
    
    # Initial condition at t=0
    u_ic = model(x_ic, t_ic)
    ic_loss = torch.mean(u_ic**2)
    
    # Total loss
    total_loss = pde_loss + bc_in_loss + bc_out_loss + ic_loss
    return total_loss

# Generate training points
N_pde = 10000  # Number of PDE points
N_bc = 1000   # Number of boundary points
N_ic = 1000   # Number of initial condition points

# PDE points (randomly sampled in domain)
x_pde = L * torch.rand(N_pde, 1, device=device)
t_pde = T * torch.rand(N_pde, 1, device=device)

# Boundary points (x=0 and x=L)
x_bc_in = torch.zeros(N_bc, 1, device=device)
t_bc_in = T * torch.rand(N_bc, 1, device=device)
x_bc_out = L * torch.ones(N_bc, 1, device=device)
t_bc_out = T * torch.rand(N_bc, 1, device=device)

# Initial condition points (t=0)
x_ic = L * torch.rand(N_ic, 1, device=device)
t_ic = torch.zeros(N_ic, 1, device=device)

# Initialize model and optimizer
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 10000
for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = compute_loss(model, x_pde, t_pde, x_bc_in, t_bc_in, x_bc_out, t_bc_out, x_ic, t_ic)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.6f}')

# Save model (optional)
torch.save(model.state_dict(), 'pinn_blood_flow.pth')

# Example evaluation
x_test = torch.linspace(0, L, 100, device=device).reshape(-1, 1)
t_test = torch.ones_like(x_test, device=device) * 0.5  # Evaluate at t=0.5s
u_pred = model(x_test, t_test).detach().cpu().numpy()