import os
import sys
import random
import numpy as np
import threading
import torch_geometric as pyg
import subprocess
import re
import shutil
import yaml
from .utils.configs import DataConf, TrainConf, DistillConf, NetworkConf
from .utils import get_distill_algorithm

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


def run_command(args):
    cmd, log_file, env = args
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return result.returncode


class Distiller:
    def __init__(self, data_cfg: DataConf, train_cfg: TrainConf, distill_cfg: DistillConf, network_cfg: NetworkConf,
                 save_dir: str, clear_cache: bool = False):
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.distill_cfg = distill_cfg
        self.network_cfg = network_cfg
        self.config = {
            'data_cfg': data_cfg.to_dict(),
            'train_cfg': train_cfg.to_dict(),
            'distill_cfg': distill_cfg.to_dict(),
            'network_cfg': network_cfg.to_dict()
        }
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.clear_cache = clear_cache
        if clear_cache:
            shutil.rmtree(self.save_dir)
        self.run_pattern = self.config['train_cfg'].get('pattern')
        self.expert_trajectory_base_dir = os.path.join(self.save_dir, self.run_pattern)
        self.distill_dir = os.path.join(self.save_dir, 'distill')
        os.makedirs(self.expert_trajectory_base_dir, exist_ok=True)
        os.makedirs(self.distill_dir, exist_ok=True)

        self.config_path = os.path.join(save_dir, 'config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

    def _check_expert_trajectory_integrity(self):
        pattern = re.compile(r'checkpoint_iters_(\d+)\.pt')
        if self.train_cfg.pattern == 'expert_trajectory':
            target_iters = list(range(0, self.train_cfg.epochs * 4 // 5 *
                                      torch.load(self.data_cfg.train_dataset_path, map_location='cpu')['y'].shape[
                                          0] + self.train_cfg.save_iters, self.train_cfg.save_iters))
        elif self.train_cfg.pattern == 'pretrain':
            target_iters = list(range(0, self.train_cfg.pretrain_epochs * 19 // 20 *
                                      torch.load(self.data_cfg.pretrain_dataset_path, map_location='cpu')['y'].shape[
                                          0] + self.train_cfg.save_iters, self.train_cfg.save_iters))
        elif self.train_cfg.pattern == 'finetune':
            target_iters = list(range(0, self.train_cfg.finetune_epochs * 19 // 20 *
                                      torch.load(self.data_cfg.finetune_dataset_path, map_location='cpu')['y'].shape[
                                          0] + self.train_cfg.save_iters, self.train_cfg.save_iters))
        expert_trajectory_dir_list = [os.path.join(self.expert_trajectory_base_dir, item) for item in
                                      os.listdir(self.expert_trajectory_base_dir)]
        if len(expert_trajectory_dir_list) == 0:
            return False
        for expert_trajectoy_dir in expert_trajectory_dir_list:
            exist_iters = []
            for ckpt_name in os.listdir(expert_trajectoy_dir):
                match = pattern.match(ckpt_name)
                if match:
                    idx = int(match.group(1))
                    exist_iters.append(idx)
            exist_iters.sort()
            if not all(elem in exist_iters for elem in target_iters):
                return False
        return True

    def prepare_expert_trajectory(self, num_expert_trajectory=1, gpu_ids_list=None):
        task_list = []
        env = os.environ.copy()
        for i in range(num_expert_trajectory):
            seed = random.randint(0, 2 ** 32 - 1)
            save_dir = os.path.join(self.expert_trajectory_base_dir, f"{i}")
            origin_dir = self.save_dir
            os.makedirs(save_dir, exist_ok=True)
            cuda_env = env.copy()
            if gpu_ids_list:
                cuda_env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids_list[i % len(gpu_ids_list)])
            else:
                cuda_env["CUDA_VISIBLE_DEVICES"] = "-1"
            cmd = [
                sys.executable,  # use current python interpreter
                "modules/distill/scripts/train_et.py",
                "--config", self.config_path,
                "--origin_dir", origin_dir,
                "--seed", str(seed),
                "--save_dir", str(save_dir),
                "--run_pattern", self.run_pattern
            ]
            log_file = os.path.join(save_dir, "stdout_stderr.log")
            task_list.append((cmd, log_file, cuda_env))
        max_workers = len(task_list)
        from multiprocessing import Pool
        with Pool(processes=max_workers) as pool:
            results = [pool.apply_async(run_command, (task,)) for task in task_list]
            for i, res in enumerate(results):
                retcode = res.get()
                print(f"Expert trajectory task {i} finished with return code {retcode}")

    def non_blocking_prepare_expert_trajectory(self, num_expert_trajectory=1, gpu_ids_list=None):
        done_event = threading.Event()

        def worker():
            self.prepare_expert_trajectory(num_expert_trajectory, gpu_ids_list)
            done_event.set()

        from multiprocessing import Process
        p = Process(target=worker)
        p.start()
        return done_event

    def distill(self):
        env = os.environ.copy()
        cuda_env = env.copy()
        if torch.cuda.is_available():
            cuda_env["CUDA_VISIBLE_DEVICES"] = "5"
        else:
            cuda_env["CUDA_VISIBLE_DEVICES"] = "-1"
        cmd = [
            sys.executable,  # use current python interpreter
            "modules/distill/scripts/distill_adapter.py",
            "--config", self.config_path,
            "--experts_dir", str(self.expert_trajectory_base_dir),
            "--save_dir", str(self.distill_dir)
        ]
        log_file = os.path.join(self.distill_dir, "stdout_stderr.log")
        task = (cmd, log_file, cuda_env)
        from multiprocessing import Pool
        with Pool(processes=1) as pool:
            res = pool.apply_async(run_command, (task,))
            retcode = res.get()
            print(f"Distillation task finished with return code {retcode}")

    def non_blocking_distill(self):
        done_event = threading.Event()

        def worker():
            self.distill()
            done_event.set()

        from multiprocessing import Process
        p = Process(target=worker)
        p.start()
        return done_event

    def __call__(self, non_blocking=True, num_expert_trajectory=1, gpu_ids_list=None, done_callback=None):
        integrity = self._check_expert_trajectory_integrity()
        if non_blocking:
            def pipeline():
                if not integrity:
                    print("Preparing ..." + self.train_cfg.pattern)
                    et_event = self.non_blocking_prepare_expert_trajectory(num_expert_trajectory, gpu_ids_list)
                    et_event.wait()
                else:
                    print("Skipping preparing expert trajectory...")

                print("Distilling...")
                distill_event = self.non_blocking_distill()
                distill_event.wait()
                if done_callback:
                    done_callback()

            threading.Thread(target=pipeline, daemon=True).start()
        else:
            if not integrity:
                print("Preparing expert trajectory...")
                self.prepare_expert_trajectory(num_expert_trajectory, gpu_ids_list)
            else:
                print("Skipping preparing expert trajectory...")

            print("Distilling...")
            self.distill()
