import torch

from torch.utils.data import Dataset
import os
import numpy as np
import torch.optim as optim
from torch.nn import functional as F
from vessel import  Blood
import yaml
import matplotlib.pyplot as plt #
from test_batch_builder import test_batch
from scipy.interpolate import interp1d
from pinn_library import test_model,  extract_from_data, plot_model
from train_pinn2 import Heart, get_inlet_file, Network, build_graph_topology, process_data, build_extrapol, calculate_loss_solver
from test_batch_builder import build_batch
from my_models import  Pinn, SIREN, PINN_FourierMLP, calculate_loss

if __name__ == "__main__":
    model_registery = {'Pinn':Pinn, 'SIREN':SIREN, 'PINN_FourierMLP':PINN_FourierMLP}

    yaml_config = "adan3.yaml"
    #yaml_config = "adan56.yaml"
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
    extrapolators = build_extrapol(data_path,cycle, graph_topology, config, batch_size)
    
    #model = Pinn(n_arteries=len(network.vessels), in_dim=17, rho=blood.rho)
    #model = SIREN(in_dim=17, hidden_dim=128, out_dim=2, num_layers=5, omega_0=20)

    #print("Model initialized with arteries:", model.n_arteries)

    from os.path import isfile, join
    from os import listdir
    
    models_path="pinn_model_unsupervised_only3/"
    onlyfiles = [f for f in listdir(models_path) if isfile(join(models_path, f))]
    path_model = 'pinn22091027.pth'  # onlyfiles[-1]
    config_name = f'config{path_model[4:]}'

    model_config = torch.load(f"{models_path}{config_name}")
    #model = PINN_FourierMLP(**config)
    #model = SIREN(**config)
   # model = PINN_FourierMLP(**config)
   # module = __import__(module_name)
    class_ = model_registery[model_config['model']]
    model = class_(**model_config)

    #model.mu_A = torch.tensor(data_dict[1]['A'].mean())      # mean of log(A) or A
    #model.sigma_A = torch.tensor(data_dict[1]['A'].std())  # std of log(A) or A
    #model.mu_Q = torch.tensor(data_dict[1]['Q'].mean())
    #model.sigma_Q = torch.tensor(data_dict[1]['Q'].std())

    print(path_model)
    checkpoint_path = f"{models_path}{path_model}"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #model.load_state_dict(torch.load(f"{models_path}{path_model}", weights_only=True))
    model.eval()
    #print(model.ff.B)

    #path_model = "pinn27080949.pth" 

    import joblib
    preprocessor = joblib.load('preprocessor.pkl')
    for i in range(20):
        sample= test_batch(graph_topology, network, T, time_resolutuion= .01, idxs=[1],xs=None, batch_size=128, random_batch = False, x_grid=False)
        xx, tt, idxs = sample[:, [0, 1,2]].T
        if not model_config['only_xt']:
            sample_transformed = preprocessor.transform(sample.copy())
        else:
            sample_transformed = sample.copy()
        #print(sample)
        # sample = torch.from_numpy(sample).float()
            
        sample_transformed = torch.from_numpy(sample_transformed).type(torch.float32)

        
        x = sample_transformed[:, 0].view(-1, 1).requires_grad_(True)
        t =  sample_transformed[:, 1].view(-1, 1).requires_grad_(True)
            # sample = torch.cat([sample[:, 0:5], sample_transformed[:, -8:]], dim=1).type(torch.float32)
        if not model_config['only_xt']:
            sample_transformed = torch.cat([x, t, sample_transformed[:, 2:]], dim=1) # 
        else:
            sample_transformed = torch.cat([x, t], dim=1) 

        A_pred, Q_pred, x, t = test_model(model, sample_transformed, 
                                        models_path=models_path, path_model=path_model, exp_transform=False)
        if model_config['standarize_output']:
            A_pred = A_pred*model.sigma_A+model.mu_A
            Q_pred = Q_pred*model.sigma_Q+model.mu_Q
            #A_pred = A_pred*model.max_A
            #Q_pred = Q_pred*model.max_Q
        A_true, Q_true = extract_from_data(sample, data_dict)
            #A_true, Q_true = extract_from_sample(sample, network, data_path='network_state_5')
        loss_pred, loss_dict_model = calculate_loss(A_pred.abs(), Q_pred, torch.tensor(A_true, dtype=torch.float32),
         torch.tensor(Q_true, dtype=torch.float32), x, t, torch.tensor(sample, dtype=torch.float32), extrapolators, f_interp, epoch=None,
         rho = model.rho, gamma_profile=model.gamma_profile, mu=model.mu)
        loss_solver, loss_dict_solver = calculate_loss_solver(np.array(A_true), np.array(Q_true), sample,  f_interp, network, extrapolators, rho=1060, mu=.004, gamma_profile=2, epoch=None)
            #(A_true_all, Q_true_all, x, t, inputs, A_flatten=None, Q_flatten=None,  rho=None, mu=None, gamma_profile=None , epoch=None)
        for key, item in loss_dict_model.items():
            print(key, item.item(), loss_dict_solver[key].item())
        plot_model(A_pred=A_pred, Q_pred=Q_pred, A_true=A_true, Q_true=Q_true, t=torch.from_numpy(tt), x=torch.from_numpy(xx)/sample[:, 5], idx=idxs[0])
        #print(Q_true)
        #plot_model(A_pred=None, Q_pred=None, A_true=A_true, Q_true=Q_true, t=t, x=x, idx=idxs[0])