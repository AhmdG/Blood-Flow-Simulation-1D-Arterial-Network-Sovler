import torch
import torch.autograd 

import numpy as np

from torch.nn import functional as F

import matplotlib.pyplot as plt #
from test_batch_builder import  process_batch, test_batch
import os
import numpy as np

def build_index_map(ct):
    # Build index map
    mapping = {}
    start_idx = 0
    for name, trans, cols in ct.transformers_:
        if name == 'categories':
            ohe = trans
            categories = ohe.categories_
            for orig_col, cats in zip(cols, categories):
                end_idx = start_idx + len(cats)
                mapping[orig_col] = list(range(start_idx, end_idx))
                start_idx = end_idx
        else:
            end_idx = start_idx + len(cols)
            for orig_col in cols:
                mapping[orig_col] = [start_idx]
                start_idx += 1
    return mapping

    
def access_data(pth, artery_idx, t, x, M, cycle, config=None):
    
    Q_true, A_true = [], []

    #artery_idx = 0
    artery_name = config["network"][artery_idx]['label']
    artery_path = f"{pth}/{cycle}/{artery_name}/" #6_3_wk
    #if not torch.is_tensor(x):
        #x = torch.tensor(x) 

    
    times_files = [f for f in os.listdir(artery_path) if f.endswith(".npz")]
    
    times = [float(f[1:-4])/1e10  for f in times_files]
    times = np.array(times)
    times_index_arg = np.argsort(times) 
    times = times[times_index_arg]-int(cycle)
    times_files = [times_files[i] for i in times_index_arg]
    t_idx = np.linspace(0, len(times_files)-1, len(t)).astype(int)
    times = times[t_idx]
    times_files = [times_files[i] for i in t_idx]
    distances = np.abs(times[None, :] - t[:, None])  # shape = (2, 4)
    closest_indices = np.argmin(distances, axis=1)
    
    x_idxs = np.floor(x * (M - 1)).astype(int)#int(torch.floor((x * (artery.M - 1)) / artery.L))
    # override artery_name
    # artery_name = "vertebral_R"
    #x_idx = -1
    for x_idx, t_idx in zip(x_idxs, closest_indices):
        t_file = times_files[t_idx]
        curr = np.load(os.path.join(artery_path, t_file), allow_pickle=False)
        # Q_true.append(torch.tensor(curr['Q'][x_idx], dtype=torch.float32))
        Q_true.append(curr['Q'][x_idx])
        A_true.append(curr['A'][x_idx])
    return torch.tensor(A_true), torch.tensor(Q_true)

def plot_results(artery_idx, config, batch_size):

    artery_name = config["network"][artery_idx]['label']
    # override artery_name
    # artery_name = "vertebral_R"
    cycle = 7
    x_idx = -1
    artery_path = f"network_state_5/{cycle}/{artery_name}/" #6_3_wk
    times_files = np.array([f for f in os.listdir(artery_path) if f.endswith(".npz")])
    times = np.array([float(f[1:-4])/1e10  for f in times_files])
    times_index_arg = np.argsort(times) 
    times = times[times_index_arg]-cycle
    #times = np.clip(times, a_min=0, a_max=1)
    times_files = times_files[times_index_arg]
    t_idx = np.linspace(0, len(times_files)-1, batch_size).astype(int)
    times_files = times_files[t_idx]
    times = times[t_idx]
    A, Q = [], []
    for t in times_files:
        curr = np.load(os.path.join(artery_path, t), allow_pickle=False)
        Q_true = torch.tensor(curr['Q'][x_idx], dtype=torch.float32)
        A_true = torch.tensor(curr['A'][x_idx], dtype=torch.float32)
        #file = artery_path+t+'.npz'
        A.append(A_true.item()) 
        Q.append(Q_true.item())
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    fig.suptitle(f"{artery_name, x_idx}")
    #ax[0].subplot(1, 2, 1)
    ax[0].plot( A)
    # Force only major ticks and turn off minor ticks
    ax[0].yaxis.set_minor_locator(plt.NullLocator())
    ax[0].set_title("A prediction")
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("A")

    #ax[1].subplot(1, 2, 2)
    ax[1].plot( Q) # np.round(times,4),
    ax[1].set_title("Q prediction")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("Q")
    plt.show()

def mse(A=None, B=None):
    if A is None:
        return 0
    if torch.is_tensor(A):
        return F.mse_loss(A, torch.zeros_like(A)) if B is None else F.mse_loss(A, B)
    else: # A is array 
        return (np.square(A)).mean(axis=0) if B is None else (np.square(A - B)).mean(axis=0)# check 0 or 1

def loss_fn(A_pred, Q_pred, A_true, Q_true, f_continuity, f_momentum, fc_bif, fc_conj, f_inlet,  f_A_leaf, f_u_leaf, epoch):
    """
    A: (b, 1)
    Q: (b, 1)
    """
    
    leaf_A_loss = mse(f_A_leaf)
    leaf_u_loss = mse(f_u_leaf)
    inlet_loss = mse(f_inlet)  
    A_loss = mse(A_pred, A_true) 
    Q_loss = mse(Q_pred, Q_true)# 
    f_continuity_loss = mse(f_continuity) # 
    f_momentum_loss = mse(f_momentum)# 
    bif_loss = mse(fc_bif) # 
    conj_loss = mse(fc_conj) #
    if epoch is None:
        loss = leaf_A_loss + leaf_u_loss + A_loss + Q_loss + f_continuity_loss + f_momentum_loss + bif_loss + conj_loss+inlet_loss
        return loss
    if epoch > 150000:
        # loss = (10**8)*leaf_A_loss + leaf_u_loss + .1*A_loss + .1*Q_loss + f_continuity_loss + f_momentum_loss + (10**0)*bif_loss + (10**0)*conj_loss+(10**8)*inlet_loss
        loss = leaf_A_loss + leaf_u_loss + .1*A_loss + .1*Q_loss + f_continuity_loss + f_momentum_loss + bif_loss + conj_loss+inlet_loss
    else:
        loss = 1000*A_loss + 1000*Q_loss
    return loss

def extract_from_sample(sample, network, data_path='network_state_5'):
    x, t, idxs = sample[:, 0:3].T
    unique_idxs = np.unique(idxs)
    # Q_artery = {}
    # A_arttery = {}
    for artery_idx in unique_idxs:
        artery_idx = artery_idx.astype(int).item()
        s, tt = network.edges[artery_idx]
        artery = network.vessels[(s, tt)]
        A_true, Q_true = access_data(data_path, artery_idx, t.squeeze().detach().numpy(), x.squeeze().detach().numpy(), artery.M, cycle, config=config)
        # A_arttery[artery_idx] = A_true
        # Q_artery[artery_idx] = Q_true
    
    return A_true, Q_true

def extract_from_data(sample, data):
    """
    extract from loaded data based on the input
    """
    # first step build the indices
    A_true = []
    Q_true = []
    x, t, idxs = sample[:, 0:3].T
    idxs = idxs.astype(int) 
    Ls = sample[:, 5]
    Ms = np.floor(Ls*1000)
    x_idxs = np.floor(x * (Ms - 1)).astype(int)
    times = data[idxs[0]]['t'].squeeze()
    distances = np.abs(times[None, :] - t[:, None])  # shape = (2, 4)
    t_idxs = np.argmin(distances, axis=1)
    for idx, x_idx, t_idx in zip(idxs, x_idxs, t_idxs):
        A_true.append(data[idx]['A'][x_idx, t_idx])
        Q_true.append(data[idx]['Q'][x_idx, t_idx])
    return A_true, Q_true


def test_model(model, sample,  models_path="pinn_model_unsupervised_only3/", path_model=None, exp_transform=True):

    # take the last model in the models_path directory
    #model.load_state_dict(torch.load(f"{models_path}{path_model}", weights_only=True))
   # model.eval()

    x, t = sample[:, 0:2].T
    x = sample[:, 0].view(-1, 1).requires_grad_(True)
    t = sample[:, 1].view(-1, 1).requires_grad_(True)
    sample= torch.cat([x, t, sample[:, 2:]], dim=1)
    A_pred, Q_pred = model(sample) #A_pred, 
    if exp_transform:
        # inverse of y = log(x+1)
        A_pred = torch.exp(A_pred)#-1
        Q_pred = torch.exp(Q_pred)#-1

    # acceess data 

    return  A_pred, Q_pred, x, t #A_pred,


        
def plot_model(A_pred=None, Q_pred=None, A_true=None, Q_true=None, t=None, x=None, idx=None): 
    # plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    if len(t.unique())>1:
        fig.suptitle(f"{idx, x[0].item()}")
    else:
        fig.suptitle(f"{idx, t[0].item()}")
    #ax[0].subplot(1, 2, 1)
    if A_pred is not None:
        if len(t.unique())>1:
            ax[0].plot(np.round(t.detach().cpu().numpy(),4), A_pred.detach().cpu().numpy(), '-') #[:,5])#
        else:
            ax[0].plot(np.round(x.detach().cpu().numpy(),4), A_pred.detach().cpu().numpy(), '-') #[:,5])#
    if A_true is not None:
        if len(t.unique())>1:
            ax[0].plot(np.round(t.detach().cpu().numpy(),4), A_true,'-')
        else:
            ax[0].plot(np.round(x.detach().cpu().numpy(),4), A_true,'-')
    # Force only major ticks and turn off minor ticks
    ax[0].yaxis.set_minor_locator(plt.NullLocator())
    ax[0].set_title("A prediction")
    if len(t.unique())>1:
        ax[0].set_xlabel("time(s)")
    else:
        ax[0].set_xlabel("x(m)")
    ax[0].set_ylabel("A")

    #ax[1].subplot(1, 2, 2)
    if Q_pred is not None:
        if len(t.unique())>1:
            ax[1].plot(np.round(t.detach().cpu().numpy(),4), Q_pred.detach().cpu().numpy(),'-') #[:,5])#
        else:
            ax[1].plot(np.round(x.detach().cpu().numpy(),4), Q_pred.detach().cpu().numpy(),'-') #[:,5])#
    if Q_true is not None:
        if len(t.unique())>1:
            ax[1].plot(np.round(t.detach().cpu().numpy(),4), Q_true, '-')
        else:
            ax[1].plot(np.round(x.detach().cpu().numpy(),4), Q_true, '-')
    ax[1].set_title("Q prediction")
    if len(t.unique())>1:
        ax[1].set_xlabel("time(s)")
    else:
        ax[1].set_xlabel("x(m)")
    ax[1].set_ylabel("Q")

    #ax.tight_layout()
    plt.show()
