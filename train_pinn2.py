if 1:
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
    #from test_batch_builder import build_batch, process_batch, pressure, newtone
    from scipy.interpolate import interp1d
    import math
    #from my_models import Pinn, PINN_FourierMLP, calculate_loss, loss_fn, get_U_bif, bif_fc, get_U_conj, conjunction_fc, SIREN
    #from pinn_library import build_index_map


def calculate_loss_solver(A_true_all, Q_true_all, inputs_all, f_interp, network, extrapolators, rho=None, mu=None, gamma_profile=None , epoch=None):
    #calculate_loss_solver(A_true, Q_true, x, t, sample, artery,  rho=None, mu=None, gamma_profile=None , epoch=None, compare=True)
    """
    the goal of this function is to calculate the error of the data produced by the solver relative to the predefined residuals
    the shape of A_true and Q_true is N by M a 2D spatial temporal Matrix for a valid calculation of different residuals
    the function must also be adapted to different types of x and t (array and/or one element)
    diffrence based gradients: the solver data assume uniformity in x and t and  
    i think it makes more sens if the data is artery centered
    """
    # first part of residuals gradient based 
    # x should not be standarized by L
    A_flatten = A_true_all
    Q_flatten = Q_true_all
    rest_idx, leaf_idx, inlet_flow_samples, inlet, conj_idx, conj_conns_idx, bif_idx, bif_conn1_idx,  bif_conn2_idx = process_batch(inputs_all, f_interp)
    inputs = inputs_all[rest_idx]
    artery_idx = np.unique(inputs[:, 2]).astype(int)
    if len(artery_idx)!=1:
        print('sample contain more than one artery, solver loss is not possible')
        return
    
    x_flatten, t_flatten = inputs[:,  0:2].T
    artery_idx = artery_idx.item()
    s, tt = network.edges[artery_idx]
    artery = network.vessels[(s, tt)]
    x_flatten = x_flatten * artery.L

    A_true = A_true_all[rest_idx]
    Q_true = Q_true_all[rest_idx]
    x_grid = np.sort(np.unique(x_flatten[rest_idx]))
    t_grid = np.sort(np.unique(t_flatten[rest_idx]))
    if 1:#(len(Q_true.shape) < 2) or (Q_true.shape[0]==1) or (Q_true.shape[-1]==1) or True:
        A_true = np.squeeze(A_true)
        Q_true = np.squeeze(Q_true)
        if len(x_grid)==1:
            n = len(x_flatten)
            A_true = np.tile(A_true.T, (n,1))
            Q_true = np.tile(Q_true.T, (n,1))
            t = t_flatten[rest_idx]
            x = x_flatten[rest_idx]
        elif len(t_grid)==1:
            n = len(t_flatten)
            A_true = np.tile(A_true, (n,1))
            Q_true = np.tile(Q_true, (n,1))
            t = t_flatten[rest_idx]
            x = x_flatten[rest_idx]
        else:
            A_true = A_true.reshape((len(x_grid), len(t_grid)))
            Q_true = Q_true.reshape((len(x_grid), len(t_grid)))
            t = t_grid
            x = x_grid
    #if A_true.shape[0] == len(t):
    A_t = np.gradient(A_true, t, axis=1)#[:, 1]  # Gradient of A with respect to t
    A_t = np.nan_to_num(A_t, posinf=0, neginf=0)
    #else:
       # A_t = 0.0
    # Calculate gradient of Q with respect to x  
    #if Q_true.shape[-1]==len(x):   
    Q_x = np.gradient(Q_true, x, axis=0)#axis=1#[:, 0]  # Gradient of Q with respect to x
    Q_x = np.nan_to_num(Q_x, posinf=0, neginf=0)
    #else:
        #Q_x = 0.0
    
    # Continuity equation residual
    f_continuity = A_t + Q_x
    
    # Calculate derivatives for momentum equation
    #if Q_true.shape[0] == len(t):
    Q_t = np.gradient(Q_true, t, axis=1)#[:, 1]  # Gradient of Q with respect to t    
    Q_t = np.nan_to_num(Q_t, posinf=0, neginf=0)
    #else:
        #Q_t = 0.0
    
    # Term 1: ∂/∂x (Q²/A)
    Q2_over_A = Q_true**2 / A_true

    #if Q2_over_A.shape[-1]==len(x):
    d_Q2_over_A_dx = np.gradient(Q2_over_A, x, axis=0)#[:, 0]  # Gradient of Q²/A with respect to x
    d_Q2_over_A_dx = np.nan_to_num(d_Q2_over_A_dx, posinf=0, neginf=0)
    #else:
        #d_Q2_over_A_dx  = 0
    # Term 2: ∂/∂x (β/ρ A^(3/2))
    A32 = A_true ** (3/2)
    #if A32.shape[-1]==len(x):
    d_A32_dx = np.gradient(A32, x, axis=0)#[:, 0]  # Gradient of A^(3/2) with respect to x
    d_A32_dx = np.nan_to_num(d_A32_dx, posinf=0, neginf=0)
    #else:
        #d_A32_dx = 0
    
    beta_flatten = inputs_all[:, 11]
    beta = inputs[:,  11] #artery.beta #inputs[:, 11]  # Pressure-area constant
    if len(x_grid)>1 and len(t_grid)>1:
        beta = beta.reshape((A_true.shape))
    term2 = (beta / rho) * d_A32_dx
    #beta_flatten = inputs[:, 11]
    
    # Viscous dissipation term
    viscous_term = -2 * (gamma_profile + 2) * np.pi * mu * Q_true / (A_true * rho+1e-6)
    
    # Tapering (Area) term

    A0 = inputs[:, 8] #artery.A0  # Reference area
    A0_flatten = inputs_all[:, 8]
    dA0dx = inputs[:, 9]  #artery.dA0dx # Gradient of reference area
    dTaudx = inputs[:, 10] #artery.dTaudx  # Gradient of shear stress
    if len(x_grid)>1 and len(t_grid)>1:
        A0 = A0.reshape((A_true.shape))
        dA0dx = dA0dx.reshape((A_true.shape))
        dTaudx = dTaudx.reshape((A_true.shape))
    
    tapering_area_term = (beta / (2 * rho)) * (A32 / A0) * dA0dx
    
    # Tapering (Shear) term
    sqrt_A_over_A0 = np.sqrt(A_true / A0)
    tapering_shear_term = -(A_true / rho) * (sqrt_A_over_A0 - 1) * dTaudx
    
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
    # secoond part of residualts not gradient based
    #leaf_idx, inlet_flow_samples, inlet, conj_idx, conj_conns_idx, bif_idx, bif_conn1_idx,  bif_conn2_idx  = process_batch(inputs, f_interp)#, n_arteries=self.n_arteries)
    #solver=1

    if len(leaf_idx) > 0:
        U_true = Q_true/A_true
        # gamma = inputs[:, 12]
        # Cc , R1, R2 = [], [], []
        #for l in leaf_idx:
        idxs = inputs[leaf_idx, 2]
        A_leaf_true, u_leaf_true = [], []
        for idx, l in zip(idxs, leaf_idx):
            A_interp = extrapolators[idx.item()]['A'] 
            Q_interp = extrapolators[idx.item()]['Q'] 
            A = A_interp(t_flatten[l].item()).item()
            Q = Q_interp(t_flatten[l].item()).item()
            A_leaf_true.append(A), u_leaf_true.append(Q/A)

        f_A_leaf = torch.from_numpy(A_leaf_true - A_true[leaf_idx])
        f_u_leaf = torch.from_numpy(u_leaf_true - U_true[leaf_idx])
    else:
        f_A_leaf, f_u_leaf = None, None


    if len(inlet) > 0:
        f_inlet = torch.from_numpy(Q_flatten[inlet] -inlet_flow_samples)

    else:
        f_inlet = None

    if len(bif_idx) > 0:
        U_flatten = Q_flatten / A_flatten
        U = get_U_bif(torch.from_numpy(U_flatten[bif_idx]), torch.from_numpy(U_flatten[bif_conn1_idx]), 
                    torch.from_numpy(U_flatten[bif_conn2_idx]), torch.from_numpy(A_flatten[bif_idx]), 
                    torch.from_numpy(A_flatten[bif_conn1_idx]), torch.from_numpy(A_flatten[bif_conn2_idx]))
                                     #u1, u2, u3, A1, A2, A3)
        fc_bif = bif_fc(U, torch.from_numpy(A0_flatten[bif_idx]), torch.from_numpy(A0_flatten[bif_conn1_idx]), torch.from_numpy(A0_flatten[bif_conn2_idx]),
                         torch.from_numpy(beta_flatten[bif_idx]), torch.from_numpy(beta_flatten[bif_conn1_idx]), torch.from_numpy(beta_flatten[bif_conn2_idx]))
    else:
        fc_bif = None
    #get_U_conj(u1, u2, A1, A2)
    if len(conj_idx) > 0 :
        fc_conj = conjunction_fc(torch.from_numpy(U_flatten[conj_idx]), torch.from_numpy(U_flatten[conj_conns_idx]), torch.from_numpy(A_flatten[conj_idx]),
                    torch.from_numpy(A_flatten[conj_conns_idx]), torch.from_numpy(beta_flatten[conj_idx]), torch.from_numpy(beta_flatten[conj_conns_idx]), 
                    torch.from_numpy(A0_flatten[conj_idx]), torch.from_numpy(A0_flatten[conj_conns_idx]), rho)

    else:
        fc_conj = None
    #A, Q = None, None
    loss = loss_fn(None, None, torch.from_numpy(A_flatten), torch.from_numpy(Q_flatten), torch.tensor(f_continuity), torch.tensor(f_momentum), fc_bif, fc_conj, f_inlet, f_A_leaf, f_u_leaf, epoch)
    #(A, Q, A_true, Q_true, f_continuity, f_momentum, fc_bif, fc_conj, f_inlet,  f_A_leaf, f_u_leaf, epoch)
    if loss!=loss:
        pass
    return loss

def process_data(data_path, cycle, topology, every_nth=100):
    """
    load the data from data path into one dict for every artery onther dict with A and Q 
    """
    cycle_path  = os.path.join(data_path, cycle)
    arteries_path = glob.glob(f"{cycle_path}/*")
    name_index = {v["name"]: k for k, v in topology.items() if isinstance(v, dict)}
    data = {}
    for artery_path in arteries_path:
        artery_name = os.path.basename(artery_path)
        artery_index = name_index[artery_name]
        files = sorted(glob.glob(f"{artery_path}/*.npz"))  # sorted by filename → time order
        #metadata = []
        Q_values = np.vstack([np.load(f)["Q"] for f in files[0::every_nth]])
        A_values = np.vstack([np.load(f)["A"] for f in files[0::every_nth]])
        t = np.vstack([float(f[-15:-4]) /1e10 - int(cycle) for f in files[0::every_nth]])
        # we use .T to keep x, t dims 
        data[artery_index] = {'Q': Q_values.T, 'A': A_values.T, 't':t}#, 'x':x}
    return data

def build_extrapol(data_path,cycle, network_topology, config, batch_size):
    pass
    # Example: build extrapolation/interpolation functions for each artery
    extrapolators = {}

    for artery_idx, artery in network_topology.items():
        if len(artery['successor']):
            continue
        artery_name = config["network"][artery_idx]['label']
        artery_path = os.path.join(data_path,cycle, artery_name)
        if not os.path.isdir(artery_path):
            continue
        times_files = [f for f in os.listdir(artery_path) if f.endswith('.npz')]
        if not times_files:
            continue
        times = np.array([float(f[1:-4]) / 1e10 for f in times_files]) - int(cycle)
        indices = sorted(range(len(times)), key=times.__getitem__)
        times = times[indices]
        times_files = [times_files[idx] for idx in indices]
        t_idx = np.linspace(0, len(times_files)-1, batch_size).astype(int)
        times_files = [times_files[idx] for idx in t_idx]
        times = times[t_idx]

        Qs, As = [], []
        for f in times_files:
            arr = np.load(os.path.join(artery_path, f), allow_pickle=False)
            Qs.append(arr['Q'][-1]) # only the last point at x=L
            As.append(arr['A'][-1])
        Qs = np.stack(Qs)  # shape: (T, M)
        As = np.stack(As)
        #plt.figure()
        #plt.plot(times, Qs)
        #plt.show()
        # Interpolators: time axis is 0, space axis is 1
        extrapolators[artery_idx] = {
            'Q': interp1d(times, Qs, axis=0, kind='cubic', fill_value='extrapolate'),
            'A': interp1d(times, As, axis=0, kind='cubic', fill_value='extrapolate'),
            'times': times
        }
    return extrapolators

def wk3(A, Q, A0, gamma, beta, Cc, R1, R2, dt=None, rho=None, Pext=10000, Pc=None, Pout=0):

    # Cc, R1 , R2 = None
    u = A/Q
    #if v.inlet_impedance_matching and 0:
        # v.R1 = rho * wave_speed(v.A[-1], v.gamma[-2]) / v.A[-1]
        #R1 = rho * wave_speed(A,gamma) / A
        # check gamma index
        # v.R2 = abs(v.total_peripheral_resistance - v.R1)
        #R2 = abs(v.total_peripheral_resistance - R1)

    # v.Pc += dt / v.Cc * (v.A[-1] * v.u[-1] - (v.Pc - v.Pout) / v.R2)
    Pc += dt / Cc * (A * u - (Pc - Pout) / R2)

    # As = v.A[-1]
    As = A
    # ssAl = np.sqrt(np.sqrt(v.A[-1]))
    ssAl = torch.sqrt(torch.sqrt(A))
    sgamma = 2 * torch.sqrt(6 * gamma)
    # sA0 = np.sqrt(v.A0[-1])
    sA0 = torch.sqrt(A0)
    # bA0 = v.beta[-1] / sA0
    bA0 = beta / sA0

    def fun(As_):
        return (As_ * R1 * (u + sgamma * (ssAl - torch.sqrt(torch.sqrt(As_)))) -
                (Pext + bA0 * (torch.sqrt(As_) - sA0)) + Pc)

    def dfun(As_):
        return (R1 * (u + sgamma * (ssAl - 1.25 * torch.sqrt(torch.sqrt(As_)))) - bA0 * 0.5 / torch.sqrt(As_))

    As = newtone(fun, dfun, As)
    us = (pressure(As, A0, beta, Pext) - Pout) / (As * R1)

    #v.A[-1] = As
    #v.u[-1] = us
    return As, As/us


def scale_input_batch(x: torch.Tensor, n_arteries=None, scale_last='linear') -> torch.Tensor:
    """
    Scales a batch of inputs with shape (N, 13) before feeding to a model.

    Parameters:
        x (torch.Tensor): Input tensor of shape (N, 13)
        scale_last (str): Method to scale the last element.
                          Options: 'linear' (default), 'log'
                          
    Returns:
        torch.Tensor: Scaled tensor of shape (N, 13)
    [x, t, idx, succ1, succ2, artery.L, artery.Rp, artery.Rd, artery.A0[x_idx],
                artery.dA0dx[x_idx], artery.dTaudx[x_idx], artery.beta[x_idx], artery.gamma[x_idx]]
    """
    if x.shape[1] != 13:
        raise ValueError("Expected input shape (N, 13)")

    x_clone = x.clone()  # Don't modify original tensor

    # Scale indices 2, 3, 4 (columns 2:5)
    x_clone[:, 2:5] = x_clone[:, 2:5] / n_arteries
    # scale x by the length of the artery
    # Ls = x[:, 5].clone()
    #x_clone[:, 0] = x_clone[:, 0]/x_clone[:, 5]

    # Scale last element (index 11)
    if scale_last == 'linear':
        x_clone[:, 11] = x_clone[:, 11] / 1e5  # Scale down to ~0.35
        x_clone[:, 12] = x_clone[:, 12] / 1e3
    elif scale_last == 'log':
        # Use log10 safely: clamp to avoid log(0) or negative
        x_clone[:, 11] = torch.log10(torch.clamp(x_clone[:, 11], min=1e-8))
    else:
        raise ValueError("scale_last must be 'linear' or 'log'")

    return x_clone

"""data = torch.tensor([
        [1.0, 0.228, 0, -1, -1, 0.0, 0.0],  # Leaf
        [1.0, 0.37, 1, 2, -1, 0.0, 0.0],   # One connection (conn_0 valid)
        [1.0, 0.0, 2, -1, 3, 0.0, 0.0],   # One connection (conn_1 valid)
        [1.0, 0.0, 3, 4, 5, 0.0, 0.0],    # Two connections
        [0.0, 0.37, 2, -1, -1, 0.0, 0.0],  # Not x=1.0 → should be ignored
        [0.0, 0.0, 3, -1, -1, 0.0, 0.0],  # Not x=1.0 → should be ignored
        [0.0, 0.0, 4, -1, -1, 0.0, 0.0],  # Not x=1.0 → should be ignored
        [0.0, 0.0, 5, -1, -1, 0.0, 0.0],  # Not x=1.0 → should be ignored
    ])"""


def build_graph_topology(network):
    graph_topology = {}
    for ves_idx in range(len(network.edges)):
        graph_topology[ves_idx] = {}
        
        s, t = network.edges[ves_idx]
        graph_topology[ves_idx]['name'] = network.vessels[(s, t)].label
        #v = network.vessels[(s, t)]
        if s == 1:
            graph_topology[ves_idx]['inlet'] = 1
        outdeg = network.graph.out_degree(t)
        if outdeg == 0:
            graph_topology[ves_idx]['outbc'] = 1
            graph_topology[ves_idx]['successor'] = []
            pass
            # outbc(v, dt, network.blood.rho)
        elif outdeg == 1:
            indeg = network.graph.in_degree(t)
            d = next(iter(network.graph.successors(t)))
            graph_topology[ves_idx]['successor'] = [network.edges.index((t, d))]
            if indeg == 1:
                graph_topology[ves_idx]['conjunction'] = 1
            elif indeg == 2:
                graph_topology[ves_idx]['anastomosis'] = 1

                #ps = list(network.graph.predecessors(t))
                pass
        elif outdeg == 2:
            graph_topology[ves_idx]['bifurcation'] = 1
            ds = list(network.graph.successors(t))
            graph_topology[ves_idx]['successor'] = [network.edges.index((t, ds[0])),
                                                    network.edges.index((t, ds[1]))]
            #join_vessels_bif(v, network.vessels[(t, ds[0])], network.vessels[(t, ds[1])])
    return graph_topology


def get_inlet_file(config):
    return config.get("inlet_file", f"{config['project_name']}_inlet.dat")

class Heart:
    def __init__(self, inlet_file):
        self.input_data = np.loadtxt(inlet_file)
        self.cardiac_period = self.input_data[-1, 0]

class Network:
    def __init__(self, config, blood, heart, Ccfl, jump, tokeep, verbose=True):
        self.blood = blood
        self.heart = heart
        self.Ccfl = Ccfl

        self.graph = nx.DiGraph()
        self.vessels = {}
        self.edges=[]

        if verbose:
            progress = tqdm(total=len(config), desc="Building network:")

        for vessel_config in config:
            vessel = Vessel(vessel_config, blood, jump, tokeep)
            self.graph.add_edge(vessel.sn, vessel.tn)
            self.vessels[(vessel.sn, vessel.tn)] = vessel
            self.edges.append((vessel.sn, vessel.tn))

            if verbose:
                progress.update(1)

        if verbose:
            progress.close()

        #self.edges = list(self.graph.edges())


if __name__ == "__main__":

    #inputs = torch.load('tensor.pt')
    #extract_aligned_conn_data(data=inputs, n_arteries=77)
    #yaml_config = "adan3.yaml"
    yaml_config = "adan56.yaml"
    with open(yaml_config, 'r') as f:
        config = yaml.safe_load(f)
    blood = Blood(config["blood"])
    heart = Heart(get_inlet_file(config))
    batch_size = 128
    T = 1
    f_interp = interp1d(heart.input_data[:, 0], heart.input_data[:, 1], kind='cubic')
    tosave = config.get("write_results", [])
    network = Network(config["network"], blood, heart,
                    config["solver"]["Ccfl"], config["solver"]["jump"],
                    tosave, verbose=True)
    print("Network initialized with arteries:", len(network.vessels))
    graph_topology = build_graph_topology(network)
    data_path = 'network_state_5/'
    cycles = sorted(next(os.walk(data_path))[1])
    cycle = cycles[-2]
    data_dict = process_data(data_path, cycle, graph_topology)
    np.savez("data/aq_dataset.npz", data=data_dict)#A=data_dict['A'], Q=data_dict['Q'], t=data_dict['t'])#, nu=0.01)
    extrapolators = build_extrapol(data_path,cycle, graph_topology, config, batch_size)
    only_xt = False
    #y = extrapolators[1]model_config['sigmas']['Q'](torch.linspace(0, 1, batch_size))
    #plt.figure()
    #plt.plot(y)
    #plt.show()
    

    # Assume X_raw is your input array of shape (n_samples, 13)
    # Let's define the column indices for each group
    if not only_xt:
        continuous_bounded = [0, 1]        # Features 0, 1
        categorical_indices = [2, 3, 4]    # Features 2, 3, 4
        continuous_small = [5, 6, 7, 8, 9, 10]  # Features 5-10
        continuous_large = [11, 12]        # Features 11, 12
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        # Create a column transformer to handle each group appropriately
        preprocessor = ColumnTransformer(
        transformers=[
            ('bounded', MinMaxScaler(), continuous_bounded),#
            ('categories', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_indices),
            ('small_scale', StandardScaler(), continuous_small),
            ('large_scale', StandardScaler(), continuous_large)
        ],
        remainder='passthrough'  # This shouldn't be needed as we've defined all columns
        )

        X_train, A_true, Q_true = build_batch(network, graph_topology, T, data_dict, batch_size=1000000,  idx0s=[1])
        preprocessor.fit(X_train)
        index_map = build_index_map(preprocessor)
        in_dim = preprocessor.transform(np.zeros((1, X_train.shape[-1]))).shape[-1]
        import joblib
        joblib.dump(preprocessor, 'preprocessor.pkl')  # Save fitted preprocessor
    else:
        in_dim = 2

    idx0s = 1

    mu_A = torch.tensor(data_dict[idx0s]['A'].mean())      # mean of log(A) or A
    sigma_A = torch.tensor(data_dict[idx0s]['A'].std())  # std of log(A) or A

    mu_Q = torch.tensor(data_dict[idx0s]['Q'].mean())
    sigma_Q = torch.tensor(data_dict[idx0s]['Q'].std())

    max_Q = torch.tensor(data_dict[idx0s]['Q'].max())
    max_A = torch.tensor(data_dict[idx0s]['A'].max())
    checkpoint = False
    if checkpoint:
        model_registery = {'Pinn':Pinn, 'SIREN':SIREN, 
        'PINN_FourierMLP':PINN_FourierMLP}
        models_path="pinn_model_unsupervised_only3/"
        model_name = 'pinn17091507.pth'  # onlyfiles[-1]
        config_name = f'config{model_name[4:]}'
        model_config = torch.load(f"{models_path}{config_name}")
        standarize_output = model_config['standarize_output']
   # module = __import__(module_name)
        class_ = model_registery[model_config['model']]
        model = class_(**model_config)
        checkpoint_path = f"{models_path}{model_name}"
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer_adam = optim.Adam(model.parameters(), lr=1e-4)
            optimizer_adam.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            total_epochs = 300000
            print(f"Resuming training from epoch {start_epoch}")
    else:

        model_config = {
        "in_dim": in_dim,
        "hidden_dim": 128,
        "out_dim": 2,
        "num_layers": 4,
        "groups": (16,16,32),
        "sigmas":(.15, .1, .05),
        "mu_A": mu_A,
        "sigma_A": sigma_A,
        "mu_Q": mu_Q,
        "sigma_Q": sigma_Q,
        "max_Q": max_Q,
        "max_A": max_A,
        'n_arteries':len(network.vessels),
        'rho':blood.rho,
        'omega_0':2.5,
        'only_xt':True
        }


        model = SIREN(**model_config)#in_dim=config['in_dim'], hidden_dim=config['hidden_dim'], out_dim=config['out_dim'], num_layers=config['num_layers'], groups=config['groups'], sigmas=config['sigmas'],mu_A=config['mu_A'], sigma_A=config['sigma_A'], mu_Q=config['mu_Q'], sigma_Q=config['sigma_Q'], max_Q = config['max_Q'], max_A=config['max_A'])
        #model = PINN_FourierMLP(**model_config)#(in_dim=config['in_dim'], hidden_dim=config['hidden_dim'], out_dim=config['out_dim'], num_layers=config['num_layers'], groups=config['groups'], sigmas=config['sigmas'],mu_A=config['mu_A'], sigma_A=config['sigma_A'], mu_Q=config['mu_Q'], sigma_Q=config['sigma_Q'], max_Q = config['max_Q'], max_A=config['max_A'])
        #+model = Pinn(**model_config)
        model_config['model'] = type(model).__name__


        optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
        gamma = 0.96   # decay rate
        scheduler = optim.lr_scheduler.StepLR(optimizer_adam, step_size=600, gamma=gamma)
        optimizer_lbfgs = optim.LBFGS(model.parameters(), 
                                lr=1,
                                max_iter=500,
                                tolerance_grad=1e-7,
                                tolerance_change=1e-9,
                                history_size=50,
                                line_search_fn='strong_wolfe')
        
        total_epochs = 300000

        best_loss = torch.inf

        standarize_output = False
        model_config['standarize_output'] = standarize_output
        model_config['standarize_sigma_mu'] = True
        model_config['standarize_max'] = False

        # date today
        from datetime import datetime
        save_date = datetime.today().strftime('%d%m%H%M')
        model_name = f'pinn{save_date}.pth'
        config_name = f'config{save_date}.pth'
        start_epoch = 0

    model_dir = "pinn_model_unsupervised_only3"
    config_once = True
    epochs_adam = int(.8*total_epochs)
    every_epochs = 200
    total_loss = 0.0
    all_losses = []
    new_loss_f = True
    
    

    for epoch in range(start_epoch, total_epochs):
        if epoch > -1 and new_loss_f:
            best_loss = torch.inf
            new_loss_f = False
        # print(epoch)
        model.train()
        #sample = sample_artery(network)
        sample, A_true, Q_true = build_batch(network, graph_topology, T, data_dict, batch_size=512, xs=None, idx0s=[idx0s])
        #plt.figure()
        #plt.plot(sample[:, 0], sample[:, 1], '*')
        #plt.show()
        #A_true = torch.log(torch.from_numpy(A_true) ).float()
        #Q_true = torch.log(torch.from_numpy(Q_true) ).float()
        A_true = torch.from_numpy(A_true).float()
        Q_true = torch.from_numpy(Q_true).float()
        if standarize_output:
            A_true = (A_true - model.mu_A)/model.sigma_A
            Q_true = (Q_true - model.mu_Q)/model.sigma_Q
            #A_true = A_true/model.max_A
            #Q_true = Q_true/model.max_Q
        if not only_xt:
            sample_transformed = preprocessor.transform(sample.copy())
        else:
            sample_transformed = sample#preprocessor.transform(sample.copy())
        sample = torch.from_numpy(sample).float()
        
        sample_transformed = torch.from_numpy(sample_transformed).type(torch.float32)
        x = sample_transformed[:, 0].view(-1, 1).requires_grad_(True)
        t = sample_transformed[:, 1].view(-1, 1).requires_grad_(True)
        if not only_xt:
            sample_transformed= torch.cat([x, t , sample_transformed[:, 2:]], dim=1)
        else:
            sample_transformed= torch.cat([x,t], dim=1)
        #sample = torch.load("tensor2.pt")
        if (epoch < epochs_adam) and 1:
            optimizer_adam.zero_grad()
            A_pred, Q_pred = model(sample_transformed) #A_pred,
            
            #A_t = calc_grad(A_pred, t)
            #loss, _ = model.calculate_loss(A_pred.abs(), Q_pred, A_true, Q_true, x, t, sample, extrapolators, f_interp, epoch)
            loss, _ = calculate_loss(A_pred, Q_pred, A_true, Q_true, x, t, sample, extrapolators, f_interp, epoch=epoch,
                        rho = model.rho, gamma_profile=model.gamma_profile, mu=model.mu)
            #loss, _ = loss_fn(A_pred, Q_pred, A_true, Q_true, None, None, None, None, None,  None, None, epoch, detailed_loss=True)
            #A_pred, Q_pred, A_true, Q_true, x, t, inputs, extrapolators, epoch

            loss.backward()
            optimizer_adam.step()
        else:
            def closure():
                optimizer_lbfgs.zero_grad()
                A_pred, Q_pred = model(sample_transformed)
                    #A_t = calc_grad(A_pred, t)
                #loss, _ = model.calculate_loss(A_pred.abs(), Q_pred, A_true, Q_true, x, t, sample, extrapolators, f_interp, epoch)
                loss, _ = calculate_loss(A_pred, Q_pred, A_true, Q_true, x, t, sample, extrapolators, f_interp, epoch=epoch,
                        rho = model.rho, gamma_profile=model.gamma_profile, mu=model.mu)
                #loss, _ = loss_fn(A_pred, Q_pred, A_true, Q_true, None, None, None, None, None,  None, None, epoch, detailed_loss=True)
                loss.backward()
                return loss
            loss = optimizer_lbfgs.step(closure)

        total_loss += loss.item()
        all_losses.append(loss.detach().numpy())
        if (epoch +1 )% every_epochs ==0:
            print(epoch+1,  round(total_loss, 2))
            if (total_loss/every_epochs)<best_loss:
                best_loss =total_loss/every_epochs
                # Save model state
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                print(f"Saving model with best loss {best_loss }")
                
                #torch.save(model.state_dict(), os.path.join(model_dir, model_name))

                torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_adam.state_dict(),
                'loss': loss.item(),
                'best_loss':best_loss}, 
                os.path.join(model_dir, model_name))
                if config_once:
                    torch.save(model_config, os.path.join(model_dir, config_name))
                    config_once = False
            total_loss = 0
        scheduler.step()  # decay LR if needed
    plt.figure()
    plt.plot(np.array(all_losses))#.detach().numpy())
    plt.show()

    
    pass