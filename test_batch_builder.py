if 1:
    import random
    from collections import defaultdict
    import yaml
    import numpy as np
    import networkx as nx
    from tqdm.auto import tqdm
    from vessel import Vessel, Blood
    import torch
    import os
    import matplotlib.pyplot as plt
    #from pinn_library import get_index_x


def get_index_x(x=None, L=None, M= None):
    # case x=x/L
    if L is None:
        x_idx = int(np.floor(x * (M - 1)))
    else:
    # case NOT x = x/L
        x_idx = int(np.floor((x * (M - 1)) / L))
    return x_idx
    


def newtone(f, df, xn, max_iter=10):
    for _ in range(max_iter):
        xn -= f(xn) / df(xn)
    return xn


def pressure(A, A0, beta, Pext):
    return Pext + beta * (torch.sqrt(A / A0) - 1.0)

def wave_speed(A, gamma):
    return np.sqrt(1.5 * gamma * np.sqrt(A))


def test_batch(successor_dict, network, T, time_resolutuion= .01, idxs=None, xs=None, batch_size=128, random_batch = False, x_grid=True):  
    """"
    the spatial resolution or the lengths of the x grid must always be equal to artery.M
    """
    ####
    

    #idx0 = 0
    if ~x_grid:
        ts = np.linspace(0, 1, int(1/time_resolutuion)+1)
    if random_batch:
        t_idxs = np.random.choice(len(ts), batch_size)
        ts = ts[t_idxs]
        
    #batch_size = len(ts)
    if random_batch and idxs is None:
        idxs = np.random.choice(idx0_pool, batch_size)
        #x = np.where(np.random.random(1) < 0.2, 1, np.random.uniform(0.01, 0.99, 1)).item()
        xs = np.where(np.random.random(batch_size) < 0.2, 1, np.random.uniform(0.01, 0.99, batch_size))

    if ~ random_batch:
        if idxs is None :
        
            idx0_pool = list(successor_dict.keys())
            idxs = [np.random.choice(idx0_pool, 1).item()]#*batch_size
        artery_idx = idxs[0]
        s, tt = network.edges[artery_idx]
        artery = network.vessels[(s, tt)]
        if x_grid:
            xs = np.linspace(0, 1, artery.M) #artery.L
            ts = [np.random.choice(ts, 1).item()]
        else:
            if xs is None:
                xs =[np.random.choice(np.linspace(0, 1, artery.M), 1, replace=False)[0]]*artery.M
                #xs = [np.where(np.random.random(1) < 0.3, 1*artery.L, np.random.uniform(0.01, 0.99, 1)*artery.L).item()]*artery.M
            else:
                xs = [xs[0]]*artery.M
    # the produced x is an x/L

    
    #x = xs[0] # to be generalized later
    batch = []# np.empty(len(xs), len(ts))
    s, tt = network.edges[artery_idx]
    artery = network.vessels[(s, tt)]
    for t in ts:
        for x in np.unique(xs):
            successors = successor_dict[artery_idx]['successor']
            succ1, succ2 = [*successors, -1, -1][:2]
            if x==1:
                x_idx=-1
            else:
                #pass artery.L if x is not x/L else pass None
                x_idx = get_index_x(x=x, L=None, M= artery.M)#np.round(x  * (artery.M - 1)).astype(int)# np.round((x / artery.L) * (artery.M - 1)).astype(int)

            sample = [x, t, artery_idx, succ1, succ2, artery.L, artery.Rp, artery.Rd, artery.A0[x_idx],
                    artery.dA0dx[x_idx], artery.dTaudx[x_idx], artery.beta[x_idx], artery.gamma[x_idx]]

            batch.append(sample)
    if ~random_batch:
        for x in np.unique(xs):
            if x==1:
                for succ in successor_dict[artery_idx]['successor']:
                    s, tt = network.edges[succ]
                    artery_succ = network.vessels[(s, tt)]
                    successors = successor_dict[succ]['successor']
                    succ1, succ2 = [*successors,-1, -1][:2]
                    #s1, s2 = successor_dict.get(succ, [-1, -1])
                    for t in ts:
                        sample = [0, t, succ, succ1, succ2, artery_succ.L, artery_succ.Rp, artery_succ.Rd, artery_succ.A0[0],
                        artery_succ.dA0dx[0], artery_succ.dTaudx[0], artery_succ.beta[0], artery.gamma[0]]
                        batch.append(sample)
                #conjunction bifurcation or wk


    
    batch = np.array(batch)
    batch[:, 0] = batch[:, 0]*batch[:, 5]
    batch[:, 1] = batch[:, 1]*T
    
    return batch#torch.tensor(batch, dtype=torch.float32), idx0, x, ts#[:batch_size]

def build_graph_topology(network):
    graph_topology = {}
    for ves_idx in range(len(network.edges)):
        graph_topology[ves_idx] = {}
        s, t = network.edges[ves_idx]
        v = network.vessels[(s, t)]
        if s == 1:
            graph_topology[ves_idx]['inlet'] = 1

            pass
            # inbc(v, current_time, dt, network.heart)
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
                #v_ = network.vessels[(t, d)]
                
                pass
                # join_vessels_conj(v, network.vessels[(t, d)], network.blood.rho)
            elif indeg == 2:
                graph_topology[ves_idx]['anastomosis'] = 1

                ps = list(network.graph.predecessors(t))
                pass
                #solve_anastomosis(
                #    network.vessels[(ps[0], t)],
                #    network.vessels[(ps[1], t)],
                #   network.vessels[(t, d)],
                #)
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

        if verbose:
            progress = tqdm(total=len(config), desc="Building network:")

        for vessel_config in config:
            vessel = Vessel(vessel_config, blood, jump, tokeep)
            self.graph.add_edge(vessel.sn, vessel.tn)
            self.vessels[(vessel.sn, vessel.tn)] = vessel

            if verbose:
                progress.update(1)

        if verbose:
            progress.close()

        self.edges = list(self.graph.edges())

def build_batch(network, successor_dict, T, data_dict, batch_size=128, time_resolution = 0.01, idx0s=None, xs=None, ts=None):
    """Builds a batch compatible with process_batch
    we dont need to call A and Q as this can be done in access data"""
    # n_arteries = len(network.vessels)
    batch = []
    Q_batch = []
    A_batch = []

    # update for ts or just ts from uniform
    
    time_space = np.linspace(0, T, int(1/time_resolution)+1)
    time_indices = np.random.choice(range(len(time_space)), batch_size)  
    ts = time_space[time_indices]
    
    #ts = np.round(np.random.uniform(0, T, batch_size), 2)
    # idx0s = np.random.choice(idx0_pool, batch_size)
    if idx0s is None:
        idx0_pool = list(successor_dict.keys())
        idx0s = np.where(np.random.random(batch_size) < 0.05, 0, np.random.choice(idx0_pool, batch_size)) # sample arteries
    else:
        if len(idx0s)==1:
            idx0s = idx0s*batch_size
    if xs is None:
        #xs =[np.random.choice(np.linspace(0, 1, int(1/time_resolutuion)+1), 1, replace=False)[0]*artery.L]*artery.M
        #xs = np.where(np.random.random(batch_size) < 0.3, 1, np.random.uniform(0.0, 0.99, batch_size))
        xs = np.where(np.random.random(batch_size) < 0.3, 1,np.random.choice(np.linspace(0, 1, int(1/.01)+1), batch_size, replace=True))
    else:
        xs =  xs*batch_size #

    # Track samples that need connections (x=1 with successors)
    connection_map = defaultdict(list)
    # add one sample for 0 0 for inlet integration
    successors = successor_dict[0]['successor']
    s, tt = network.edges[0]
    artery = network.vessels[(s, tt)]
    succ1, succ2 = [*successors,-1, -1][:2]
    t_idx =  np.random.choice(time_indices, 1)
    t = ts[t_idx].item()
    batch.append([0, t, 0, succ1, succ2, artery.L, artery.Rp, artery.Rd, artery.A0[0],
                artery.dA0dx[0], artery.dTaudx[0], artery.beta[0], artery.gamma[0]])
    convert_index = data_dict[0]['Q'].shape[0]*time_resolution
    A = data_dict[0]['A'][0, int(t_idx*convert_index)]
    Q = data_dict[0]['Q'][0, int(t_idx*convert_index)]
    A_batch.append(A)
    Q_batch.append(Q)
    

    i = 0
    while len(batch) < batch_size:
        t, t_idx, idx0, x = ts[i], time_indices[i], idx0s[i], xs[i]
        i+=1
        s, tt = network.edges[idx0]
        artery = network.vessels[(s, tt)]
        #x_idx = np.random.choice(np., batch_size)
        #xx = np.linspace(0, artery.L, artery.M)
        if x==1:
            idx=-1
        else:
            idx = get_index_x(x=x, L=None, M= artery.M)#np.round(x  * (artery.M - 1)).astype(int)# np.round((x / artery.L) * (artery.M - 1)).astype(int)
        #idx = np.clip(idx, 0, artery.M - 1)
        #print(x*artery.L, xx[idx])
        #t = random.uniform(0, T)
        #idx0 = random.choice(idx0_pool)
        successors = successor_dict[idx0]['successor']
        succ1, succ2 = [*successors,-1, -1][:2]
        #x = 1 if random.random() < 0.3 else random.uniform(.01, .99) # edge include

        
        # Determine sample type strictly from successors
        if succ1 == -1 and succ2 == -1:
            # Leaf - can be x=1 or 0<x<1
            #x = 1 if random.random() < 0.3 else random.uniform(.01, .99) # edge include
            sample = [x, t, idx0, -1, -1, artery.L, artery.Rp, artery.Rd, artery.A0[idx],
                artery.dA0dx[idx], artery.dTaudx[idx], artery.beta[idx], artery.gamma[idx]]
            batch.append(sample)
            convert_index = data_dict[idx0]['Q'].shape[0]*time_resolution
            A = data_dict[idx0]['A'][idx, int(t_idx*convert_index)]
            Q = data_dict[idx0]['Q'][idx, int(t_idx*convert_index)]
            A_batch.append(A)
            Q_batch.append(Q)
        elif (succ1 == -1) ^ (succ2 == -1):  # XOR for exactly one successor
            # Conjunction - exactly one successor is -1
            #x = 1 if random.random() < 0.3 else random.uniform(.01, .99) # edge include
            sample = [x, t, idx0, succ1, succ2, artery.L, artery.Rp, artery.Rd, artery.A0[idx],
                artery.dA0dx[idx], artery.dTaudx[idx], artery.beta[idx], artery.gamma[idx]]
            batch.append(sample)
            convert_index = data_dict[idx0]['Q'].shape[0]*time_resolution
            A = data_dict[idx0]['A'][idx, int(t_idx*convert_index)]
            Q = data_dict[idx0]['Q'][idx, int(t_idx*convert_index)]
            A_batch.append(A)
            Q_batch.append(Q)
            
            # If x=1, mark for connections
            if x == 1:
                active_succ = succ2 if succ1 == -1 else succ1
                connection_map[t].append((idx0, active_succ))
        else:
            # Bifurcation - two successors
            #x = 1 if random.random() < 0.3 else random.uniform(.01, .99) # edge include
            sample = [x, t, idx0, succ1, succ2, artery.L, artery.Rp, artery.Rd, artery.A0[idx],
                artery.dA0dx[idx], artery.dTaudx[idx], artery.beta[idx], artery.gamma[idx]]
            batch.append(sample)
            convert_index = data_dict[idx0]['Q'].shape[0]*time_resolution
            A = data_dict[idx0]['A'][idx, int(t_idx*convert_index)]
            Q = data_dict[idx0]['Q'][idx, int(t_idx*convert_index)]
            A_batch.append(A)
            Q_batch.append(Q)
            
            # If x=1, mark for connections
            if x == 1:
                connection_map[t].append((idx0, succ1))
                connection_map[t].append((idx0, succ2))
    
    # Add connection samples (x=0)
    for t, connections in connection_map.items():
        for idx0, succ in connections:
            #if len(batch) >= batch_size:
                #break
            # Add [0, t, succ, ...]
            s, tt = network.edges[succ]
            artery_succ = network.vessels[(s, tt)]
            successors = successor_dict[succ]['successor']
            succ1, succ2 = [*successors,-1, -1][:2]
            #s1, s2 = successor_dict.get(succ, [-1, -1])
            sample = [0, t, succ, succ1, succ2, artery_succ.L, artery_succ.Rp, artery_succ.Rd, artery_succ.A0[0],
                artery_succ.dA0dx[0], artery_succ.dTaudx[0], artery_succ.beta[0], artery.gamma[0]]
            batch.append(sample)
            
            convert_index = data_dict[succ]['Q'].shape[0]*time_resolution
            A = data_dict[succ]['A'][0, int(t_idx*convert_index)]
            Q = data_dict[succ]['Q'][0, int(t_idx*convert_index)]
            A_batch.append(A)
            Q_batch.append(Q)
    batch = np.array(batch)
    batch[:, 0] = batch[:, 0]*batch[:, 5]
    batch[:, 1] = batch[:, 1]*T
    
    return batch, np.array(A_batch), np.array(Q_batch)#, dtype=torch.float32)#[:batch_size]

def build_batch2(network, successor_dict, T, data_dict, batch_size=128, time_resolution = 0.01, idx0s=None, xs=None, ts=None):
    """Builds a batch compatible with process_batch
    we dont need to call A and Q as this can be done in access data"""
    # n_arteries = len(network.vessels)

    Q_batch = []
    A_batch = []

    # update for ts or just ts from uniform
    
    time_space = np.linspace(0, T, int(1/time_resolution)+1)
    batch_size = len(time_space)
    time_indices = range(len(time_space))
    ts = time_space#[time_indices]
    
    #ts = np.round(np.random.uniform(0, T, batch_size), 2)
    # idx0s = np.random.choice(idx0_pool, batch_size)
    s, tt = network.edges[idx0s[0]]
    artery = network.vessels[(s, tt)]
    if xs is None:
        #xs =[np.random.choice(np.linspace(0, 1, int(1/time_resolutuion)+1), 1, replace=False)[0]*artery.L]*artery.M
        #xs = np.where(np.random.random(batch_size) < 0.3, 1, np.random.uniform(0.0, 0.99, batch_size))
        xs = np.where(np.random.random(1) < 0.3, 1,np.random.choice(np.linspace(0, 1, int(1/time_resolution)+1), 1, replace=True))
        x_idx = get_index_x(x=xs[0], L=None, M= artery.M)
    
        x_idxs = [x_idx]*batch_size
    else:
        x_idxs =  xs*batch_size #

    convert_index = (data_dict[0]['Q'].shape[1]-1)*time_resolution
    convert_index = np.round(convert_index, 2)
    for x_idx, t_idx in zip (x_idxs, time_indices):
      
        A = data_dict[idx0s[0]]['A'][x_idx, int(t_idx*convert_index)]
        Q = data_dict[idx0s[0]]['Q'][x_idx, int(t_idx*convert_index)]
        #plt.figure()
        #plt.plot(data_dict[idx0s[0]]['Q'][x_idx, :])
        #plt.show()
        A_batch.append(A)
        Q_batch.append(Q)
  
    return np.array(A_batch), np.array(Q_batch)#, dtype=torch.float32)#[:batch_size]

def process_batch(batch, f_interp):#, successor_dict):
    """Processes batches from build_batch"""
    rest_idx = []
    leaf_idx = []
    conj_idx = []
    conj_conns_idx = []
    bif_idx = []
    bif_conn1_idx = []
    bif_conn2_idx = []
    inlet = []
    inlet_flow_samples = []
    # Build index maps
    time_map = defaultdict(lambda: defaultdict(list))
    for i, sample in enumerate(batch):
        x, t, idx0 = sample[0].item(),sample[1].item(), sample[2].item()
        if x==0 and idx0==0:
            inlet.append(i)
            inlet_flow_samples.append(f_interp(t).item()) 
            #continue

        if x == 0:# or x == 1:  # Only map points at 0 or 1
            #continue
            time_map[t][idx0].append(i)
    
    # Process samples
    for i, sample in enumerate(batch):
        x, t, idx0, succ1, succ2 = sample[0].item(),sample[1].item(), sample[2].item(), sample[3].item(),sample[4].item()
        if x!=1:
            continue
        # Leaf detection
        if succ1 == -1 and succ2 == -1:
            pass
            leaf_idx.append(i)
        
        # Conjunction detection
        elif (succ1 == -1) ^ (succ2 == -1):  # XOR for exactly one successor
            conj_idx.append(i)
            
            # Find connection (x=0 sample with same t and active successor)
            active_succ = succ2 if succ1 == -1 else succ1
            conn_indices = time_map[t].get(active_succ, [])
            
            
            #if conn_indices:
            conj_conns_idx.append(conn_indices[0])
            #else:
                #conj_conns_idx.append(None)
        
        # Bifurcation detection
        else:# succ1 != -1 and succ2 != -1:
            bif_idx.append(i)
            
            # Find connections
            conn1 = time_map[t].get(succ1, [])
            
            conn2 = time_map[t].get(succ2, [])
            
            bif_conn1_idx.append(conn1[0] if conn1 else None)
            bif_conn2_idx.append(conn2[0] if conn2 else None)

    # Verify output invariants
    assert len(conj_idx) == len(conj_conns_idx)
    assert len(bif_idx) == len(bif_conn1_idx) == len(bif_conn2_idx)
    rest_idx = list(set(range(len(batch))) - set(conj_conns_idx) - set(bif_conn1_idx) -set(bif_conn2_idx))
    
    #return torch.tensor(rest_idx), torch.tensor(leaf_idx), torch.tensor(inlet_flow_samples),torch.tensor(inlet), torch.tensor(conj_idx), torch.tensor(conj_conns_idx), torch.tensor(bif_idx), torch.tensor(bif_conn1_idx), torch.tensor(bif_conn2_idx)
    return rest_idx, leaf_idx, np.array(inlet_flow_samples), inlet, conj_idx, conj_conns_idx, bif_idx, bif_conn1_idx, bif_conn2_idx


# uniforlmly sample points in the domain
"""

def sample_artery2(network, n_samples=10):
    samples = []
    n_arteries = len(network.vessels)
    #artery_index = torch.randperm(n_arteries)[:n_samples]

    for _ in range(n_samples):
        sample = []
        
        artery_index = torch.randint(0, n_arteries-1, (1,)).item() # randomly select an artery
        
        # normalize the artery index to the range of the network edges
        #n_arteries = 1

        s, tt = network.edges[artery_index] # get the start and end nodes of the artery
        x_artery = network.vessels[(s, tt)].x # get the x coordinates of the artery
        # randomly select 10 times teps
        time_ = sample_resolution(a=0,b=1, n=10, resolution=0.0001)# = torch.randint(0, 100, (1,)) # assuming 100 time steps
        # randomly slect 8 interion points and 2 the inlet and outlet
        idx_x = torch.randint(1, len(x_artery)-2, (8,)).tolist() # assuming 8 points
        idx_x.append(0) # add the inlet
        idx_x.append(-1) # add the outlet
        #t = torch.linspace(0, 1, 10) # assuming 10 time 
        successors = graph_topology[artery_index]['successor']

        # sample structure
        # [x/L, tt/T, artery_index/n_arteries, conn_0/n_arteries, conn_1/n_arteries, L, Rp, Rd, A0]
        for tt, x, idx in zip(time_, x_artery[idx_x], idx_x):
            sample.append([x/network.vessels[(s, t)].L, tt.item(), artery_index / n_arteries, *[i / n_arteries for i in [*successors,-1, -1][:2]],network.vessels[(s, t)].L, 
                        network.vessels[(s, t)].Rp, network.vessels[(s, t)].Rd,
                        network.vessels[(s, t)].A0[idx], network.vessels[(s, t)].dA0dx[idx], network.vessels[(s, t)].dTaudx[idx], network.vessels[(s, t)].beta[idx]])
        
        
        for suc in successors:
            s, t = network.edges[suc]
            sample.append([0, time_[-1].item(),suc/n_arteries , *[i / n_arteries for i in [*graph_topology[suc]['successor'], -1, -1][:2]], network.vessels[(s, t)].L, 
                        network.vessels[(s, t)].Rp, network.vessels[(s, t)].Rd,
                        network.vessels[(s, t)].A0[0], network.vessels[(s, t)].dA0dx[idx], network.vessels[(s, t)].dTaudx[idx], network.vessels[(s, t)].beta[idx]])
        sample = torch.as_tensor(sample, dtype=torch.float32)
        samples.append(sample)
    samples = torch.concatenate(samples, axis=0)
    # add dA0 and dTaudx
    return samples"""
    

    #out_x = [network.vessels[(s, t)].x[0] for s,t in network.edges[successors]]

if __name__ == "__main__":
    
    #inputs = torch.load('tensor.pt')
    #extract_aligned_conn_data(data=inputs, n_arteries=77)
    yaml_config = "adan56.yaml"
    with open(yaml_config, 'r') as f:
        config = yaml.safe_load(f)
    blood = Blood(config["blood"])
    heart = Heart(get_inlet_file(config))
    tosave = config.get("write_results", [])
    network = Network(config["network"], blood, heart,
                    config["solver"]["Ccfl"], config["solver"]["jump"],
                    tosave, verbose=True)
    print("Network initialized with arteries:", len(network.vessels))
    graph_topology = build_graph_topology(network)

    for i in range(10000):
        if i%10000==0:
            print(i)
        batch_size = random.randint(1, 1000)
        batch = build_batch(network, graph_topology, .918, 50)#batch_size)
        conj_idx, conj_conns_idx, bif_idx, bif_conn1_idx, bif_conn2_idx = process_batch(batch)
    pass
