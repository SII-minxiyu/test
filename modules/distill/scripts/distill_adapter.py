import os
import sys
import yaml
import argparse
from datetime import datetime
sys.path.append('.')
import random
import numpy as np
import torch
import torch_geometric as pyg
from modules.distill.utils import get_distill_algorithm

os.environ["WANDB_MODE"] = "offline"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    pyg.seed_everything(seed)
    
def unset_seed():
    random.seed(None)
    np.random.seed(None)
    torch.seed()
    torch.cuda.seed()
    torch.cuda.seed_all()

    
def parse_args():
    parser = argparse.ArgumentParser(
        description='Distillation script with YAML configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        help='Path to the YAML configuration file'
    )
    
    parser.add_argument(
        '-s',
        '--save_dir',
        type=str,
        help='Save directory'
    )
    
    parser.add_argument(
        '-e',
        '--experts_dir',
        required=True, 
        type=str,
        help='Expert trajectory directory'
    )
    
    return parser.parse_args()

def load_config(config_path, args):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if args.experts_dir:
            config['distill_cfg']['expert_trajectory_dir_list'] = [os.path.join(args.experts_dir, expert_dir) for expert_dir in os.listdir(args.experts_dir)]
            config['distill_cfg']['data_path_list'] = [os.path.join(args.experts_dir, expert_dir, 'train_dataset.pt') for expert_dir in os.listdir(args.experts_dir)]
        expert_config_path = os.path.join(config['distill_cfg']['expert_trajectory_dir_list'][0], 'config.yaml')
        with open(expert_config_path, 'r') as f:
            expert_config = yaml.safe_load(f)
            config['data_cfg'] = expert_config['data_cfg']
            config['distill_cfg']['p'] = expert_config['train_cfg']['p']
            config['network_cfg'] = expert_config['network_cfg']
            config['train_cfg'] = expert_config['train_cfg']
        if args.save_dir:
            config['distill_cfg']['save_dir'] = args.save_dir
        else:
            config['distill_cfg']['save_dir'] = os.path.join(
                '.log',
                'distill', 
                config['distill_cfg']["algorithm"],
                config['data_cfg']["dataset_name"], 
                '' if config['data_cfg']["name"] is None else config['data_cfg']["name"],
                datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            )
            os.makedirs(config['distill_cfg']['save_dir'])
        
        
        cfg_save_path = os.path.join(config['distill_cfg']['save_dir'], 'config.yaml')
        with open(cfg_save_path, 'w') as f:
            yaml.safe_dump(config, f)
            
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading config file: {str(e)}")

def main():
    args = parse_args()
    config = load_config(args.config, args)
    
    # Parse configurations
    distill_cfg, data_cfg, expert_network_cfg = config['distill_cfg'], config['data_cfg'], config['network_cfg']
    set_seed(data_cfg['seed'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize algorithm for distillation
    expert_network_name = expert_network_cfg.pop('name')
    distiller = get_distill_algorithm(algorithm=distill_cfg['algorithm'])
    
    # Perform distillation 
    distiller.distill(
        expert_network_name=expert_network_name, 
        expert_network_dict=expert_network_cfg, 
        device=device,
        **distill_cfg
    )

if __name__ == "__main__":
    main()