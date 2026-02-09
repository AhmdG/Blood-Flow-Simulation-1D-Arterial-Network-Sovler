import numpy as np
from pathlib import Path
import os
import json

def save_network_state(network, current_time,dt,mu, rho, output_dir=None):
    """
    Save current state of the network's arteries for training.
    
    Args:
        network: network object with .arteries list
        current_time: current simulation time (used for file name)
        output_dir: path to save dataset
    """

    time_str = f"t{int(current_time*1e10):010d}"  # e.g., t000100 for 0.0001s
    for artery_id, artery in network.vessels.items():
        artery_dir = os.path.join(output_dir, f"{artery.label}/")
        if not os.path.exists(artery_dir):
            os.makedirs(artery_dir, exist_ok=True)

        # Save meta info once
        meta_path = os.path.join(artery_dir, "meta.json")
        if not os.path.exists(meta_path):
            meta = {
                "Rd": artery.Rd,
                "Ru": artery.Rp,
                "L": artery.L,
                "M": artery.M,
                "gamma_profile":artery.gamma_profile,
                "dt": dt,
                "mu": mu, 
                "rho": rho, 
                "tapered": artery.tapered, 
                "viscoelastic": artery.viscoelastic
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            if not (len(artery.Q) == len(artery.A) == artery.M):
                pass
            # Save A0 once
            np.save(os.path.join(artery_dir, "x.npy"), artery.x)
            np.save(os.path.join(artery_dir, "beta.npy"), artery.beta)
            np.save(os.path.join(artery_dir, "A0.npy"), artery.A0)
            np.save(os.path.join(artery_dir, "dA0dx.npy"), artery.dA0dx)
            np.save(os.path.join(artery_dir, "dTaudx.npy"), artery.dTaudx)
            np.save(os.path.join(artery_dir, "gamma.npy"), artery.gamma)
            np.save(os.path.join(artery_dir, "Cv.npy"), artery.Cv)


        # Save time step state
        np.savez_compressed(
            os.path.join(artery_dir, f"{time_str}.npz"),
            Q=artery.Q,
            A=artery.A,
            Q_in=artery.Q[0], #Q_in is the flow at the inlet for the next time step
            P_out= pressure(artery.A[-1], artery.A0[-1], artery.beta[-1], artery.Pext) # pressure at the outlet for the next time step
        )


def pressure(A, A0, beta, Pext):
    return Pext + beta * (np.sqrt(A / A0) - 1.0)

def save_waveforms(idx: int, t: float, v):
    for k in v.waveforms:
        if k == "A":
            v.waveforms[k][idx, :] = [
                t, v.A[0], v.A[v.node2], v.A[v.node3], v.A[v.node4], v.A[-1]
            ]
        elif k == "Q":
            v.waveforms[k][idx, :] = [
                t, v.Q[0], v.Q[v.node2], v.Q[v.node3], v.Q[v.node4], v.Q[-1]
            ]
        elif k == "u":
            v.waveforms[k][idx, :] = [
                t, v.u[0], v.u[v.node2], v.u[v.node3], v.u[v.node4], v.u[-1]
            ]
        elif k == "P":
            pressures = [
                pressure(v.A[i], v.A0[i], v.beta[i], v.Pext) - v.Pout
                for i in [0, v.node2, v.node3, v.node4, len(v.A) - 1]
            ]
            v.waveforms[k][idx, :] = [t] + pressures

def flush_waveforms(v):
    for k, data in v.waveforms.items():
        filename = f"{v.label}_{k}.last"
        np.savetxt(filename, data, fmt="%.8f")

def append_last_to_out(v):
    for k in v.waveforms:
        last_file = Path(f"{v.label}_{k}.last")
        out_file = Path(f"{v.label}_{k}.out")
        if last_file.exists():
            data = np.loadtxt(last_file)
            with open(out_file, "ab") as f:
                np.savetxt(f, data if data.ndim > 1 else data[np.newaxis, :], fmt="%.8f")

