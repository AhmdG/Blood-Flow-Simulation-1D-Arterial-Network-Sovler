# openbf/__init__.py

# Import dependencies

# Custom modules (equivalent of `include(...)`)
from . import simulation

# Public interface

def run_simulation(config_file: str, savedata=False):
    """
    Run the arterial network simulation.

    Args:
        config_file (str): Path to the YAML configuration file.
    """
    simulation.run_simulation(config_file, savedata=savedata)



def test_network():
    import os, shutil
    print("test network")

    test_folder = "network"
    yaml_path = os.path.join(test_folder, f"{test_folder}.yaml")
    result_dir = os.path.join(test_folder + "_results")

    shutil.rmtree(result_dir, ignore_errors=True)
    run_simulation(yaml_path, verbose=False, out_files=False)
    shutil.rmtree(result_dir, ignore_errors=True)

#if __name__ == "__main__":
run_simulation('adan3.yaml', savedata=True)
    #pass
