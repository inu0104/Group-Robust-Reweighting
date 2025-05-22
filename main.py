import argparse
import random
import torch
import json
import sys

import numpy as np

from utils.data_loader import load_data
from utils.metrics import evaluate_group_metrics
from method.group_dro import run_group_dro
from method.xgb_method import (
    run_xgb_erm, run_xgb_GroupIndep, run_xgb_IS, run_xgb_GW, run_xgb_SW)
from method.erm import run_erm
from models import node, tabnet

MODEL_CLASSES = {
    'node': node.NODE,
    'tabnet': tabnet.TabNet
}

METHODS = {
    'erm': run_erm,
    'group-dro' : run_group_dro,
    'xgb_erm' : run_xgb_erm,
    'xgb_gind' : run_xgb_GroupIndep,
    'xgb_is' : run_xgb_IS,
    'xgb_gw' : run_xgb_GW,
    'xgb_sw' : run_xgb_SW
}

def main(config_file):
    
    with open(config_file, 'r') as f:
        config = json.load(f)

    seed = config.get("seed", 2026)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, train_df = load_data(config)
    
    if config['model_type'] == 'xgb':
        method_fn = METHODS[config["method"]]  
        model = method_fn(train_loader, valid_loader, test_loader, config['train_params'])
    else:
        model_class = MODEL_CLASSES[config['model_type']]
        model = model_class(**config['model_params']).to(device)
        method_fn = METHODS[config["method"]]  
        model = method_fn(model, train_loader, valid_loader, train_df, config['train_params'], device)
    
    evaluate_group_metrics(model, test_loader, device, config['model_type'])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments with optional post-processing.')
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    main(args.config)
