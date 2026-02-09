def get_value(data, artery_index, time_idx, space_idx):
    values_Q = data[artery_index]['Q']
    values_A = data[artery_index]['A']
    offsets = data[artery_index]['offsets']
    return values_A[offsets[time_idx] + space_idx], values_Q[offsets[time_idx] + space_idx]

def extract_aligned_conn_data(data, n_arteries=1):
    """
    Extracts and organizes connection data from a tensor, optimized for performance and memory.
    # [x/L, tt/T, artery_index/n_arteries, conn_0/n_arteries, conn_1/n_arteries, L, Rp, Rd, A0, dA0dx, dTaudx]
    
    Args:
        data: torch.Tensor of shape (N, 5+) with columns [x, t, portion, conn_0, conn_1, ...]
    
    Returns:
        tuple: (leafs, conj_samples, conj_conns, bif_samples, bif_conn1, bif_conn2)
    """
    # Early validation
    if data.shape[1] != 12:
        raise ValueError("Input data must have at least 5 columns")
    
    # Extract columns with clear names
    x_values = data[:, 0]
    time_values = data[:, 1]
    portion_idx = (data[:, 2]).long()  # Portion index scaled to number of arteries
    conn_0 = (data[:, 3]).long()
    conn_1 = (data[:, 4]).long()

    # Filter edge rows (x == 1)
    edge_mask = x_values == 1
    edge_idx = torch.where(edge_mask)[0]
    #edge_data = data[edge_mask]
    edge_time = time_values[edge_mask]
    edge_conn_0 = conn_0[edge_mask]
    edge_conn_1 = conn_1[edge_mask]

    # Classify edges: leaf (-1, -1), conjunction (one -1), bifurcation (neither -1)
    #is_leaf = (edge_conn_0 == -1) & (edge_conn_1 == -1)
    is_conj = (edge_conn_0 == -1) != (edge_conn_1 == -1)  # XOR
    is_bifur = (edge_conn_0 != -1) & (edge_conn_1 != -1)

    # Extract samples for each category
    #leaf_idx = edge_idx[is_leaf]
    #leaf_samples = edge_data[is_leaf]
    conj_idx = edge_idx[is_conj]
    #conj_samples = edge_data[is_conj]
    bif_idx = edge_idx[is_bifur]
    #bif_samples = edge_data[is_bifur]

    # Extract connection indices and values
    #conj_conn_idx = torch.where(is_conj)[0]
    conj_conn_val = torch.where(edge_conn_0[is_conj] != -1, edge_conn_0[is_conj], edge_conn_1[is_conj])
    #bif_conn_idx = torch.where(is_bifur)[0]
    bif_conn0_val = edge_conn_0[is_bifur]
    bif_conn1_val = edge_conn_1[is_bifur]

    # Filter x == 0 data for connection lookups
    x0_mask = x_values == 0
    x0_idx = torch.where(x0_mask)[0]
    #x0_data = data[x0_mask]
    x0_time = time_values[x0_mask]
    x0_portion = portion_idx[x0_mask]

    # Vectorized connection row lookup
    def find_conn_rows(target_times, target_portions):
        """
        Vectorized lookup for rows in x0_data matching given times and portion indices.
        
        Args:
            target_times: torch.Tensor of time values
            target_portions: torch.Tensor of portion indices
            
        Returns:
            torch.Tensor of matched rows or NaN-filled rows if no match
        """
        if target_times.numel() == 0:
            return torch.empty(0, data.shape[1], device=data.device, dtype=data.dtype)
        
        # Create broadcastable masks
        time_match = x0_time.view(-1, 1) == target_times
        portion_match = x0_portion.view(-1, 1) == target_portions
        matches = time_match & portion_match
        
        # Find first match per target
        match_idx = matches.any(dim=0).nonzero(as_tuple=True)[0]
        #result = torch.full((target_times.size(0), data.shape[1]), float('nan'), 
                           #device=data.device, dtype=data.dtype)
        result_idx = torch.full((target_times.size(0),), -1, 
                              dtype=torch.long, device=data.device)       
        if match_idx.numel() > 0:
            # Select first matching row for each column
            #matched_rows = matches[:, match_idx].max(dim=0).indices
            valid_matches = matches[:, match_idx]
            matched_rows = torch.where(valid_matches)[0]
            #result[match_idx] = x0_data[matched_rows]
            try:
                result_idx[match_idx] = x0_idx[matched_rows]
            except:
                pass
        
        #return result, result_idx
        return result_idx

    # Compute aligned connection rows
    # Compute aligned connection rows and their indices
    #conj_conns, conj_conns_idx = find_conn_rows(edge_time[is_conj], conj_conn_val)
    conj_conns_idx = find_conn_rows(edge_time[is_conj], conj_conn_val)
    #bif_conn1, bif_conn1_idx = find_conn_rows(edge_time[is_bifur], bif_conn0_val)
    bif_conn1_idx = find_conn_rows(edge_time[is_bifur], bif_conn0_val)
    #bif_conn2, bif_conn2_idx = find_conn_rows(edge_time[is_bifur], bif_conn1_val)
    bif_conn2_idx = find_conn_rows(edge_time[is_bifur], bif_conn1_val)

    # return leaf_samples, conj_samples, conj_conns, bif_samples, bif_conn1, bif_conn2
    # return leaf_idx, conj_idx, conj_conns_idx, bif_idx, conj_conn_idx, bif_conn_idx, #leaf_samples.numpy(), conj_samples.numpy(), conj_conns.numpy(), bif_samples.numpy(), bif_conn1.numpy(), bif_conn2.numpy()
    return conj_idx, conj_conns_idx, bif_idx, bif_conn1_idx,  bif_conn2_idx

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
    
def sample_resolution(a=0, b=1, n=1, resolution=0.01):
    """
    Sample n unique points in [a, b] with given resolution.
    """
    num_steps = int((b - a) / resolution)
    if n > num_steps + 1:
        raise ValueError("Cannot sample more points than available steps")
    rand_ints = torch.randperm(num_steps + 1)[:n]
    samples = a + rand_ints * resolution
    return samples

def sample_resolution2(a=0,b=1, n=1, resolution=0.01):
    num_steps = int((b - a) / resolution)

    # Generate random integers in [0, num_steps], then scale
    rand_ints = torch.randint(0, num_steps + 1, (n,))
    #rand_ints = torch.randperm(num_steps + 1)[:n]
    rand_vals = a + rand_ints * resolution
    return rand_vals

if 0:
    from PyPDF2 import PdfReader, PdfWriter

    def split_pdf(input_pdf, output_prefix="page"):
        reader = PdfReader(input_pdf)
        
        for i, page in enumerate(reader.pages, start=1):
            writer = PdfWriter()
            writer.add_page(page)
            
            output_filename = f"{output_prefix}_{i}.pdf"
            with open(output_filename, "wb") as f:
                writer.write(f)
            print(f"Saved {output_filename}")

    # Example usage
    split_pdf("document_.pdf", output_prefix="page")
if 0:
    from PyPDF2 import PdfMerger

    def merge_pdfs(pdf_list, output="merged.pdf"):
        merger = PdfMerger()
        for pdf in pdf_list:
            merger.append(pdf)
        merger.write(output)
        merger.close()
        print(f"Merged into {output}")

    # Example usage
    pdf_files = ["page_5_.pdf", "page_6_.pdf", "page_7_.pdf", "page_8_.pdf"]
    merge_pdfs(pdf_files, "combined_.pdf")

class TimeSpaceDataset(Dataset):
    def __init__(self, metadata, values, offsets):
        self.metadata = metadata
        self.values = values
        self.offsets = offsets
        # Build global index of (file_idx, space_idx)
        self.global_index = [(i, j) 
            for i,(f,t,l) in enumerate(metadata) for j in range(0, l, 1)]
    
    def __len__(self):
        return len(self.global_index)
    
    def __getitem__(self, idx):
        file_idx, space_idx = self.global_index[idx]
        value = self.values[self.offsets[file_idx] + space_idx]
        time = self.metadata[file_idx][1]
        return {
            "time": torch.tensor(time, dtype=torch.float32),
            "space_idx": torch.tensor(space_idx, dtype=torch.long),
            "value": torch.tensor(value, dtype=torch.float32),
        }


def encode_time(t, period):
    return torch.stack([
        torch.sin(2*np.pi*t/period),
        torch.cos(2*np.pi*t/period)
    ], dim=-1)

def get_batch(x_all, t_all, batch_size):
    # 80% random points + 20% on a coarse grid
    n_random = int(0.8 * batch_size)
    n_grid = batch_size - n_random
    
    # Random samples
    idx_rand = torch.randperm(len(x_all))[:n_random]
    x_rand, t_rand = x_all[idx_rand], t_all[idx_rand]
    
    # Grid samples
    x_grid = torch.linspace(x_all.min(), x_all.max(), int(n_grid**0.5))
    t_grid = torch.linspace(t_all.min(), t_all.max(), int(n_grid**0.5))
    x_grid, t_grid = torch.meshgrid(x_grid, t_grid)
    
    return (
        torch.cat([x_rand, x_grid.flatten()]),
        torch.cat([t_rand, t_grid.flatten()])
    )


def sample_artery(network, n_samples=512):
    """
    Samples artery data ensuring binary x values and unique t/portion pairs for x == 0 rows.
    
    Args:
        network: Network object with vessels, edges, and graph_topology.
        n_samples: Number of artery samples to generate.
    
    Returns:
        torch.Tensor: Shape (N, 12) with columns [x, t, portion, conn_0, conn_1, L, Rp, Rd, A0, dA0dx, dTaudx, beta]
    """
    n_arteries = len(network.vessels)
    if n_arteries < 2:
        raise ValueError("Network must have at least 2 arteries")
    
    # Sample unique arteries
    #n_samples = min(n_samples, n_arteries)
    selected_arteries = torch.randperm(n_arteries)[:n_samples]
    
    # Generate unique time values for all samples
    total_time_points = n_samples * 10
    time_ = sample_resolution(a=0, b=1, n=total_time_points, resolution=0.0001)
    time_ = time_.reshape(n_samples, 10)
    
    samples = []
    seen_successors = {}  # Track (suc, t) pairs to avoid duplicates
    for sample_idx, artery_index in enumerate(selected_arteries):
        artery_index = artery_index.item()
        sample = []
        s, t = network.edges[artery_index]
        artery = network.vessels[(s, t)]
        L = network.vessels[(s, t)].L
        x_artery = artery.x
        sample_times = time_[sample_idx]
        idx_x = torch.randint(1, len(x_artery)-2, (8,)).tolist() + [0, -1]  # Inlet, interior, outlet
        successors = graph_topology[artery_index]['successor']

        # Sample artery points with binary x
        for tt, idx in zip(sample_times, idx_x):
            tt = (round(tt.item() * 1e4) / 1e4)
            
            x_val = x_artery[idx]# 1.0 if idx == -1 else 0.0  # Outlet: x = 1, others: x = 0
            xx = (round(x_val/L * 1e4) / 1e4)
            conn_0, conn_1 = successors[:2] if len(successors) >= 2 else (successors + [-1, -1])[:2]
            sample.append([
                xx, tt, artery_index, conn_0, conn_1,
                artery.L, artery.Rp, artery.Rd, artery.A0[idx],
                artery.dA0dx[idx], artery.dTaudx[idx], artery.beta[idx]
            ])
        
        # Sample successor points with x = 0 and unique times
        # outlet_time = (round(sample_times[-1].item()* 1e4) / 1e4)
        outlet_time = (round(sample_times[-1].item()* 1e4) / 1e4)
        for suc in successors:
            if suc >= n_arteries:
                continue  # Skip invalid successor indices
            if (suc, outlet_time) in seen_successors:
                continue  # Skip to avoid duplicate t/portion pair
            seen_successors[(suc, outlet_time)] = True
            s, t = network.edges[suc]
            artery = network.vessels[(s, t)]
            suc_successors = graph_topology[suc]['successor']
            conn_0, conn_1 = suc_successors[:2] if len(suc_successors) >= 2 else (suc_successors + [-1, -1])[:2]
            sample.append([
                0.0, outlet_time, suc, conn_0, conn_1,
                artery.L, artery.Rp, artery.Rd, artery.A0[0],
                artery.dA0dx[0], artery.dTaudx[0], artery.beta[0]
            ])
        
        sample = torch.tensor(sample, dtype=torch.float32)
        samples.append(sample)
    
    if not samples:
        return torch.empty((0, 12), dtype=torch.float32)
    return torch.cat(samples, dim=0)


def test_batch(successor_dict, network, T, idx0=None, batch_size=128, solver=False, config=None):
    #####
    if solver:
        batch = []
        A = []
        Q = []
        idx0_pool = list(successor_dict.keys())
        random_batch = 0
        cycle = 7
        directory_path = "network_state_5"
        if random_batch :
            idx0s = np.random.choice(idx0_pool, batch_size)
            for artery_idx in idx0s:
                s, tt = network.edges[artery_idx] #edges and vessels use to be not aligned are not aligned
                artery = network.vessels[(s, tt)]
                #artery_idx = 0
                artery_name = config["network"][artery_idx]['label']
                #x_idx = min(5, len(artery.x))
                x = artery.L*np.random.uniform(0,1, 1)
                # case x=x/L
                #x_idx = int(np.floor(x * (artery.M - 1)))
                    # Compute the index (0-based)
                # case NOT x = x/L
                x_idx = int(np.floor((x * (artery.M - 1)) / artery.L))
                # Clamp to valid indices (in case of floating-point rounding errors)
                x_idx = max(0, min(x_idx, artery.M - 1))
                artery_path = f"network_state_2/{cycle}/{artery_name}/"
                times = sorted([f for f in os.listdir(artery_path) if f.endswith(".npz")]) 
                t = np.random.choice(times, 1)[0]
                curr = np.load(os.path.join(artery_path, t), allow_pickle=False)
                if not len(curr['Q'])==len(curr["A"]) == artery.M:
                    pass
                Q_true = torch.tensor(curr['Q'][x_idx], dtype=torch.float32)
                #plt.figure()
                #plt.plot(curr['Q'])
                #plt.show()
                A_true = torch.tensor(curr['A'][x_idx], dtype=torch.float32)
                A.append(A_true.item()) 
                Q.append(Q_true.item())
                t = float(t[1:-4])/1e10 - cycle
                successors = successor_dict[artery_idx]['successor']
            
                succ1, succ2 = [*successors,-1, -1][:2]
                sample = [x.item(), t, artery_idx, succ1, succ2, artery.L, artery.Rp, artery.Rd, artery.A0[x_idx],
                    artery.dA0dx[x_idx], artery.dTaudx[x_idx], artery.beta[x_idx], artery.gamma[x_idx]]
            return torch.tensor(batch), torch.tensor(A), torch.tensor(Q)
        else:
            artery_idx = np.random.choice(idx0_pool, 1).item()
            #artery_idx = 0
            #for artery_idx in idx0s:
            s, tt = network.edges[artery_idx] #edges and vessels use to be not aligned are not aligned
            artery = network.vessels[(s, tt)]
                #artery_idx = 0
            artery_name = config["network"][artery_idx]['label']
            print(artery_name)
                #x_idx = min(5, len(artery.x))
            x = artery.x
            artery_path = f"{directory_path}/{cycle}/{artery_name}/"
            times_files = np.array([f for f in os.listdir(artery_path) if f.endswith(".npz")])
            times = np.array([float(f[1:-4])/1e10  for f in times_files])
            times_index_arg = np.argsort(times) 
            times = times[times_index_arg]-cycle
            times = np.clip(times, a_min=0, a_max=1)
            times_files = times_files[times_index_arg]

            # data is not aligned -> now okay
            t_idx = np.linspace(0, len(times_files)-1, batch_size).astype(int)
            times_files = times_files[t_idx]
            times = times[t_idx]
            successors = successor_dict[artery_idx]['successor']
            succ1, succ2 = [*successors,-1, -1][:2]
            #times_points = []
            A, Q = np.empty((batch_size, len(x))),  np.empty((batch_size, len(x)))
            A_flatten, Q_flatten = [], []
            for (idx, t_file), t in zip(enumerate(times_files), times):
                curr = np.load(os.path.join(artery_path, t_file), allow_pickle=False)
                Q_true = curr['Q']
                
                #plt.figure()
                #plt.plot(curr['Q'])
                #plt.show()
                A_true = curr['A']
                #file = artery_path+t+'.npz'
                A[idx]= A_true 
                Q[idx] = Q_true
                #t = min(float(t_file[1:-4])/1e10 - cycle,1)
                #times_points.append(t)

                
                for x_idx, xx in enumerate(x):
                    xx = xx.item()/artery.L
                    sample = [xx, t, artery_idx, succ1, succ2, artery.L, artery.Rp, artery.Rd, artery.A0[x_idx],
                    artery.dA0dx[x_idx], artery.dTaudx[x_idx], artery.beta[x_idx], artery.gamma[x_idx]]
                    batch.append(sample)
                    A_flatten.append(A_true[x_idx])
                    Q_flatten.append(Q_true[x_idx])
                    
                pass
                for suc in successors:
                    s_succ, t_succ = network.edges[suc]
                    artery_succ = network.vessels[(s_succ, t_succ)]
                    artery_name_succ = config["network"][suc]['label']
                    suc_successors = successor_dict[suc]['successor']
                    conn_0, conn_1 = (suc_successors + [-1, -1])[:2]
                    batch.append([
                        0.0, t, suc, conn_0, conn_1, artery_succ.L, artery_succ.Rp, artery_succ.Rd, artery_succ.A0[0],
                        artery.dA0dx[0], artery_succ.dTaudx[0], artery_succ.beta[0]], artery.beta[0])
                    artery_path_succ = f"{directory_path}/{cycle}/{artery_name_succ}/"
                    curr_succ = np.load(os.path.join(artery_path_succ, t_file), allow_pickle=False)
                    A_flatten.append(curr_succ['A'][0])
                    Q_flatten.append(curr_succ['Q'][0])
                    
            return torch.tensor(batch), A, Q, x, times, artery, torch.tensor(A_flatten), torch.tensor(Q_flatten)
def calculate_loss_solver2(A_pred, Q_pred, A_true, Q_true, A_flatten, Q_flatten, x, t, inputs, artery,  rho=None, mu=None, gamma_profile=None , epoch=None):
    #inputs_non_std = h.clone().detach()
    #x = inputs[:, 0]
    #t = inputs[:, 1]
    x = x / artery.L
    A_t = np.gradient(A_pred, t, axis=0)#[:, 1]  # Gradient of A with respect to t
    # Calculate gradient of Q with respect to x  
    if Q_pred.shape[-1]==len(x):   
        Q_x = np.gradient(Q_pred, x, axis=1)#[:, 0]  # Gradient of Q with respect to x
    else:
        Q_x = 0#np.gradient(Q_pred.repeat(len(x),1), x, axis=1)
    
    # Continuity equation residual
    f_continuity = A_t + Q_x
    
    # Calculate derivatives for momentum equation
    Q_t = np.gradient(Q_pred, t, axis=0)#[:, 1]  # Gradient of Q with respect to t    
    
    # Term 1: ∂/∂x (Q²/A)
    Q2_over_A = Q_pred**2 / A_pred
    if Q2_over_A.shape[-1]==len(x):
        d_Q2_over_A_dx = np.gradient(Q2_over_A, x, axis=1)#[:, 0]  # Gradient of Q²/A with respect to x
    else:
        d_Q2_over_A_dx  = 0
    # Term 2: ∂/∂x (β/ρ A^(3/2))
    A32 = A_pred ** (3/2)
    if A32.shape[-1]==len(x):
        d_A32_dx = np.gradient(A32, x, axis=1)#[:, 0]  # Gradient of A^(3/2) with respect to x
    else:
        d_A32_dx = 0
    beta = artery.beta#inputs[:, 11]  # Pressure-area constant
    term2 = (beta / rho) * d_A32_dx
    beta_flatten = inputs[:, 11]
    
    # Viscous dissipation term
    viscous_term = -2 * (gamma_profile + 2) * np.pi * mu * Q_pred / (A_pred * rho+1e-6)
    
    # Tapering (Area) term
    A0 = artery.A0#inputs[:, 8]  # Reference area
    A0_flatten = inputs[:, 8]
    dA0dx = artery.dA0dx#inputs[:, 9]  # Gradient of reference area
    dTaudx = artery.dTaudx#inputs[:, 10]  # Gradient of shear stress
    
    tapering_area_term = (beta / (2 * rho)) * (A32 / A0) * dA0dx
    
    # Tapering (Shear) term
    sqrt_A_over_A0 = np.sqrt(A_pred / A0)
    tapering_shear_term = -(A_pred / rho) * (sqrt_A_over_A0 - 1) * dTaudx
    
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
    
    leaf_idx, inlet_flow_samples, inlet, conj_idx, conj_conns_idx, bif_idx, bif_conn1_idx,  bif_conn2_idx  = process_batch(inputs, f_interp)#, n_arteries=self.n_arteries)
    solver=1

    if leaf_idx.numel() > 0:
        # gamma = inputs[:, 12]
        # Cc , R1, R2 = [], [], []
        #for l in leaf_idx:
        U_pred = Q_pred/A_pred
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


    if inlet.numel() > 0:
        if solver:
            f_inlet = Q_flatten[inlet] -inlet_flow_samples
        else:
            f_inlet = Q_pred[inlet] - inlet_flow_samples
    else:
        f_inlet = None
    
    if solver:
        U_flatten = Q_flatten / A_flatten
        if bif_idx.numel() > 0:
            U = get_U_bif(U_flatten[bif_idx], U_flatten[bif_conn1_idx], U_flatten[bif_conn2_idx],
                        A_flatten[bif_idx], A_flatten[bif_conn1_idx], A_flatten[bif_conn2_idx])#u1, u2, u3, A1, A2, A3)
            fc_bif = bif_fc(U, A0_flatten[bif_idx], A0_flatten[bif_conn1_idx], A0_flatten[bif_conn2_idx], beta_flatten[bif_idx], beta_flatten[bif_conn1_idx], beta_flatten[bif_conn2_idx])
        else:
            fc_bif = None
        #get_U_conj(u1, u2, A1, A2)
        if conj_idx.numel() > 0 :
            fc_conj = conjunction_fc(U_flatten[conj_idx], U_flatten[conj_conns_idx], A_flatten[conj_idx], A_flatten[conj_conns_idx],
                        beta_flatten[conj_idx], beta_flatten[conj_conns_idx], A0_flatten[conj_idx], A0_flatten[conj_conns_idx], rho)

        else:
            fc_conj = None
        #A, Q = None, None
        loss = loss_fn(torch.tensor(A_true), torch.tensor(Q_true), A_flatten, Q_flatten, torch.tensor(f_continuity), torch.tensor(f_momentum), fc_bif, fc_conj, f_inlet, f_A_leaf, f_u_leaf, epoch)
        #(A, Q, A_pred, Q_pred, f_continuity, f_momentum, fc_bif, fc_conj, f_inlet,  f_A_leaf, f_u_leaf, epoch)
    else:
        U_pred = Q_pred/A_pred
        if bif_idx.numel() > 0:
            U = get_U_bif(U_pred[bif_idx], U_pred[bif_conn1_idx], U_pred[bif_conn2_idx],
                        A_pred[bif_idx], A_pred[bif_conn1_idx], A_pred[bif_conn2_idx])#u1, u2, u3, A1, A2, A3)
            fc_bif = bif_fc(U, A0[bif_idx], A0[bif_conn1_idx], A0[bif_conn2_idx], beta[bif_idx], beta[bif_conn1_idx], beta[bif_conn2_idx])
        else:
            fc_bif = None
        #get_U_conj(u1, u2, A1, A2)
        if conj_idx.numel() > 0 :
            fc_conj = conjunction_fc(U_pred[conj_idx], U_pred[conj_conns_idx], A_pred[conj_idx], A_pred[conj_conns_idx],
                        beta[conj_idx], beta[conj_conns_idx], A0[conj_idx], A0[conj_conns_idx], rho)

        else:
            fc_conj = None
        # A, Q = None, None
        loss = loss_fn(torch.tensor(A_true), torch.tensor(Q_true), A_pred, Q_pred, f_continuity, f_momentum, fc_bif, fc_conj, f_inlet, f_A_leaf, f_u_leaf, epoch)
    if loss!=loss:
        pass
    return loss
    """        {
        "preds": preds,
        "loss": loss,
        "losses": losses,
    }"""    
