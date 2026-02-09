import os
import shutil
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob

from .network import Network, Heart 
from .vessel import Vessel
from .solver import solve, calculate_dt
from .boundary_conditions import update_ghost_cells
from .output import save_waveforms, flush_waveforms, append_last_to_out,  save_network_state
#from .heart import Heart
from .vessel import Blood
from tqdm.auto import tqdm


def load_yaml_config(yaml_config):
    with open(yaml_config, 'r') as f:
        return yaml.safe_load(f)


def get_conv_error(network):
    errors = [(v.label, get_vessel_conv_error(v)) for v in network.vessels.values()]
    max_error, label = max((e, l) for l, e in errors)
    return max_error, label


def get_vessel_conv_error(v):
    current = v.waveforms['P'][:, 3]  # index 3 == 4th col
    prev = np.loadtxt(f"{v.label}_P.last")[:, 3]
    return np.sqrt(np.mean((current - prev) ** 2)) / 133.332


def get_inlet_file(config):
    return config.get("inlet_file", f"{config['project_name']}_inlet.dat")


def preamble(yaml_config, verbose=True, savedir=""):
    if verbose:
        print("Loading config...")
    config = load_yaml_config(yaml_config)

    project_name = config["project_name"]
    if verbose:
        print(f"project name: {project_name}")

    results_dir = savedir or config.get("output_directory", f"{project_name}_results")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    yaml_config_name = os.path.basename(yaml_config)
    shutil.copy(yaml_config, os.path.join(results_dir, yaml_config_name))

    inlet_file = get_inlet_file(config)
    try:
        shutil.copy(inlet_file, os.path.join(results_dir, inlet_file))
    except:
        # Try to copy relative to YAML file location
        inlet_path = os.path.join(Path(yaml_config).parent, inlet_file)
        shutil.copy(inlet_path, os.path.join(results_dir, inlet_file))

    os.chdir(results_dir)
    return config


def run_simulation(yaml_config, verbose=True, out_files=False, save_stats=False, savedir="", savedata=False):
    initial_dir = os.getcwd()
    config = preamble(yaml_config, verbose, savedir)

    blood = Blood(config["blood"])
    heart = Heart(get_inlet_file(config))

    tosave = config.get("write_results", [])
    if "P" not in tosave:
        tosave.append("P")

    network = Network(config["network"], blood, heart,
                      config["solver"]["Ccfl"], config["solver"]["jump"],
                      tosave, verbose=verbose)

    total_time = float(config["solver"]["cycles"]) * heart.cardiac_period
    jump = config["solver"]["jump"]
    checkpoints = np.linspace(0, heart.cardiac_period, jump)

    if verbose:
        print("\nStart simulation")

    current_time = 0.0
    passed_cycles = 0
    prog = tqdm(total=None, desc=f"Solving cycle #{passed_cycles}") if verbose else None
    counter = 1
    conv_error = float('inf')
    converged = False

    try:
        import time
        start = time.time()
        while True:
            dt = calculate_dt(network)
            solve(network, dt, current_time, NN=False)
            update_ghost_cells(network)
            if savedata and (passed_cycles>5) or True:
                save_network_state(network, current_time,dt, blood.mu, blood.rho,  output_dir=f'network_state_5/{passed_cycles}/')
            if verbose:
                prog.update(1)

            if current_time >= checkpoints[counter - 1]:
                for v in network.vessels.values():
                    save_waveforms(counter-1, current_time, v)
                counter += 1

            if (current_time - heart.cardiac_period * passed_cycles) >= heart.cardiac_period and \
               (current_time - heart.cardiac_period * passed_cycles + dt) > heart.cardiac_period:

                if passed_cycles > 0:
                    conv_error, error_loc = get_conv_error(network)

                for v in network.vessels.values():
                    flush_waveforms(v)
                    if out_files:
                        append_last_to_out(v)

                if verbose:
                    if passed_cycles > 0:
                        prog.set_postfix({"RMSE (mmHg)": f"{conv_error:.4f}", "@": error_loc})
                        prog.close()
                    else:
                        prog.close()
                    print()

                checkpoints += heart.cardiac_period
                passed_cycles += 1
                prog = tqdm(total=None, desc=f"Solving cycle #{passed_cycles}") if verbose else None
                counter = 1

            if current_time >= total_time or \
               passed_cycles == config["solver"]["cycles"] or \
               conv_error < config["solver"]["convergence_tolerance"]:
                if verbose:
                    prog.close()
                converged = True
                break

            current_time += dt

        # Cleanup
        for v in network.vessels.values():
            cleanup(v, tosave)

        if verbose:
            printstats(time.time() - start, converged, passed_cycles)

        if save_stats:
            savestats(time.time() - start, converged, passed_cycles, config["project_name"])

    except Exception as e:
        print("\nAn error occurred, terminating simulation.\n")
        print(str(e))

    os.chdir(initial_dir)


def cleanup(v, tokeep):
    if not v.tosave:
        for f in glob.glob(f"{v.label}_*"):
            os.remove(f)
    for l in ("P", "A", "Q", "u"):
        if l not in tokeep:
            for f in glob.glob(f"{v.label}_{l}.*"):
                os.remove(f)


def printstats(duration, converged, passed_cycles):
    print(f"Simulation {'converged' if converged else 'not converged'}, cycles: {passed_cycles}")
    print(f"Elapsed time: {duration:.2f}s")


def savestats(duration, converged, passed_cycles, name):
    with open(f"{name}.conv", 'w') as f:
        f.write(f"{converged}\n{passed_cycles}\n{duration}\n")

