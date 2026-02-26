import sys
import yaml
import numpy as np
import random
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
import torch  # Should be imported after RDKit for some reason


def disable_rdkit_logging():
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # added
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def parse_yaml_config(args):
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        print(f'{key: <40} -> {value}')
        arg_dict[key] = value
    args.config = args.config.name
    return args

# added

def make_worker_init_fn(base_seed: int):
    def _fn(worker_id: int):
        s = base_seed + worker_id
        import random, numpy as np, torch
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
    return _fn

def make_torch_generator(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return g