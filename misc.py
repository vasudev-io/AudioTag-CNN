import argparse
import json

#logs and tensorboard
def get_summary_writer_log_dir(args: argparse.Namespace, arguments: list) -> str:
    """
    Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: Parsed CLI Arguments
        arguments: List of argument specifications

    Returns:
        Subdirectory of log_dir with a unique subdirectory name.
    """
    tb_log_dir_components = ['CNN']

    #print if print_flag is 1
    for arg in arguments:
        if arg["print_flag"] == 1:
            name = arg["name"].lstrip('--').replace('-', '_')
            value = getattr(args, name, arg["default"])
            tb_log_dir_components.append(f"{name}={value}")

    tb_log_dir_prefix = '_'.join(tb_log_dir_components) + "_run_"

    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    
    return str(tb_log_dir)

#normalise data b/w -1 to 1
def compute_global_min_max(dataset):
    min_val, max_val = float('inf'), -float('inf')

    for _, samples, _ in dataset:
        current_min = samples.min()
        current_max = samples.max()

        if current_min < min_val:
            min_val = current_min
        if current_max > max_val:
            max_val = current_max

    return min_val.item(), max_val.item()

#save/load to json with min max vals
def save_min_max_values(min_val, max_val, file_path):
    with open(file_path, 'w') as file:
        json.dump({'min': min_val, 'max': max_val}, file)

def load_min_max_values(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data['min'], data['max']
    except FileNotFoundError:
        return None, None
    
#load any json
def load_data(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)  
    
#switch cases for dataset key
def get_val_key(model_name):
    switcher = {
        "BaseCNN": "val",
        "ChunkResCNN1": "valSpec",
        "ChunkResCNN2": "valSpecChunk",
        "CRNN": "valSpec"
    }
    return switcher.get(model_name, "Missing val key")

def get_test_key(model_name):
    switcher = {
        "BaseCNN": "test",
        "ChunkResCNN1": "testSpec",
        "ChunkResCNN2": "testSpecChunk",
        "CRNN": "testSpec"
    }
    return switcher.get(model_name, "Missing test key")