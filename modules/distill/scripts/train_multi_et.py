import os
import sys
import argparse
import random
import numpy as np
import torch_geometric as pyg
import subprocess
from datetime import datetime
from multiprocessing import Pool

sys.path.append('.')
import torch
os.environ["WANDB_MODE"] = "offline"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

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

def parse_args():
    parser = argparse.ArgumentParser(
        description='Training script with YAML configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        help='Path to the YAML configuration file'
    )
    parser.add_argument(
        '-d',
        '--dataset_name',
        type=str,
        help='Dataset name to override config'
    )
    parser.add_argument(
        '-n',
        '--name',
        type=str,
        help='Dataset subset name to override config'
    )
    parser.add_argument(
        '-s',
        '--save_dir',
        type=str,
        help='Save directory', 
        default=None
    )
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        default=200, 
        help='Train epoch'
    )
    parser.add_argument(
        '--num_expert', 
        type=int, 
        default=1, 
        help='Number of expert trajectory'
    )
    parser.add_argument(
        '-w',
        '--wandb_run_id',
        type=str,
        help='wandb run id for resume'
    )
    parser.add_argument('--gpu_ids', type=str, default='0',
                      help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    return parser.parse_args()

def run_command(args):
    command, log_file, env = args
    with open(log_file, "w") as f:
        cuda_env = env.copy()
        cuda_str = command[0]
        if cuda_str.startswith("CUDA_VISIBLE_DEVICES="):
            cuda_env["CUDA_VISIBLE_DEVICES"] = cuda_str.split("=")[1]
            cmd = command[1:]
        else:
            cmd = command
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=cuda_env)
    return result.returncode

def main():
    args = parse_args()
    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    task_list = []
    env = os.environ.copy()

    if args.save_dir:
        base_save_dir = args.save_dir
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        dataset_name = args.dataset_name if args.dataset_name else f"dataset_{timestamp}"
        subset_name = args.name if args.name is None else f"mole_{timestamp}"
        base_save_dir = os.path.join(
            '.log',
            'distill', 
            'expert_trajectory',
            dataset_name,
            subset_name,
            f'{timestamp}',
        )
    
    for i in range(args.num_expert):
        seed = random.randint(0, 2**32 - 1)
        save_dir = os.path.join(base_save_dir, f"{i}")
        os.makedirs(save_dir, exist_ok=True)

        command = [
            "CUDA_VISIBLE_DEVICES=" + str(gpu_ids[i % len(gpu_ids)] if gpu_ids else "-1"), 
            "python",
            "scripts/train_et.py",
            "--config", args.config, 
            "--seed", str(seed), 
            "--save_dir", str(save_dir)
        ]
        if args.dataset_name:
            command.extend(["--dataset_name", str(args.dataset_name)])
        if args.name:
            command.extend(["--name", str(args.name)])
        if args.epochs:
            command.extend(["--epochs", str(args.epochs)])
        if args.wandb_run_id:
            command.extend(["--wandb_run_id", args.wandb_run_id])

        log_file = os.path.join(save_dir, "stdout_stderr.log")
        print(f"Prepared: {' '.join(command)}")
        print(f"Logging to: {log_file}")

        task_list.append((command, log_file, env.copy()))

    # max_workers = min(len(task_list), len(gpu_ids)) if gpu_ids else len(task_list)
    max_workers = len(task_list)
    
    with Pool(processes=max_workers) as pool:
        results = [pool.apply_async(run_command, (task,)) for task in task_list]
        for i, res in enumerate(results):
            retcode = res.get()
            print(f"Task {i} finished with return code {retcode}")

if __name__ == "__main__":
    main()