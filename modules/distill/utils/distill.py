import os
import gc
import time

import numpy as np
import torch
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Sampler
from tqdm import tqdm
import wandb
import re
import random
from packaging import version
from typing import List
import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from torch.nn import MSELoss
from .reparam_module import ReparamModule
from .datasets import DistillDataset, IndexControlSampler, BasicMDDataset
from .nets import get_network
from .adapters import get_adapter_model
from .optimizers import get_dynamic_optimizer
from .schedulers import get_dynamic_scheduler
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
pyg_version = version.parse(torch_geometric.__version__)


class CustomIndexSampler(Sampler):
    def __init__(self, data_source, shuffle=True, seed=None):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.dataset_size = len(data_source)
        self.custom_indices = None
        self.current_indices = None

    def __iter__(self):
        if self.custom_indices is not None:
            indices = self.custom_indices
        else:
            if self.shuffle:
                g = torch.Generator()
                if self.seed is not None:
                    g.manual_seed(self.seed)
                indices = torch.randperm(self.dataset_size, generator=g).tolist()
            else:
                indices = list(range(self.dataset_size))

        self.current_indices = indices.copy()
        return iter(indices)

    def __len__(self):
        return self.dataset_size

    def set_indices(self, indices):
        self.custom_indices = indices

    def reset_indices(self):
        self.custom_indices = None

    def get_current_indices(self):
        return self.current_indices.copy() if self.current_indices is not None else None


def create_optimizer(param, lr, optimizer_type='adam', momentum=0.0):
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(param, lr=lr) if isinstance(param, list) else optim.Adam([param], lr=lr)
    else:
        optimizer = optim.SGD(param, lr=lr, momentum=momentum) if isinstance(param, list) else optim.SGD([param], lr=lr,
                                                                                                         momentum=momentum)
    return optimizer


def create_scheduler(optimizer, scheduler_type='step', decay_step=None, decay_rate=None):
    if scheduler_type.lower() == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_rate)
    elif str.lower(scheduler_type.lower()) == 'expdecaylr':
        decay_lambda = lambda step: max(decay_rate ** (step / decay_step), 1e-6)
        expert_scheduler = LambdaLR(optimizer, lr_lambda=decay_lambda)
        return expert_scheduler
    return None


def restore_module_parameters(reparam_model, flat_params):
    reparam_model._unflatten_param(flat_params)

    for mn, n in reparam_model._param_infos:
        module = reparam_model._get_module_from_name(mn)

        tensor = getattr(module, n)
        if hasattr(module, n):
            delattr(module, n)

        module.register_parameter(n, torch.nn.Parameter(tensor))

    for mn, n, shared_mn, shared_n in reparam_model._shared_param_infos:
        module = reparam_model._get_module_from_name(mn)
        shared_module = reparam_model._get_module_from_name(shared_mn)
        shared_param = getattr(shared_module, shared_n)

        if hasattr(module, n):
            delattr(module, n)
        module.register_parameter(n, shared_param)

    return reparam_model


class AdapterMTT:
    def __init__(self):
        pass

    def distill(
            self,
            num_iteration: int,
            expert_network_name: str,
            expert_network_dict: dict,
            expert_trajectory_dir_list: List[str],
            data_path_list: List[str],
            valdata_path: str,
            valbatch: int,
            max_start_iter: int,
            min_start_iter: int,
            num_expert: int,
            save_step: int,
            save_dir: str,
            distill_batch_size: int,
            device: str,
            project_name: str = 'AdapterDistill',
            train_data_path: str = None,
            p: float = 100,
            enable_adapter: bool = True,
            data_mask_rate: float = 0.0,
            use_coarse_label: bool = False,
            adapter_dict: dict = None,
            distill_lr_adapter: float = 1e-4,
            distill_energy: bool = True,
            distill_force: bool = True,
            distill_optimizer_type: str = 'adam',
            distill_scheduler_type: str = "stepLR",
            distill_scheduler_decay_step: int = 100,
            distill_scheduler_decay_rate: float = 0.2,
            max_grad_norm_clip: float = None,
            enable_log: bool = True,
            distill_sequence: str = None,
            **kwargs
    ):
        assert distill_energy or distill_force, f"At least one of 'distill_energy' and 'distill_force' should be true."
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)

        student_net = get_network(name=expert_network_name, **expert_network_dict).to(device)

        student_net_params = None
        base_model_params_mask = None
        adapter_params = None
        if enable_adapter and adapter_dict:
            print("学生模型构建Adapter")
            student_net = get_adapter_model(adapter_dict, student_net).to(device)
            student_net.train()
            student_net_params = []
            base_model_params_mask = []
            for name, param in student_net.named_parameters():
                if not 'lora_' in name:
                    base_model_params_mask.append(torch.ones_like(param.reshape(-1), dtype=torch.bool))
                else:
                    base_model_params_mask.append(torch.zeros_like(param.reshape(-1), dtype=torch.bool))

                student_net_params.append(param.reshape(-1).requires_grad_(True))

            student_net_params = torch.nn.Parameter(torch.cat(student_net_params, dim=0))
            base_model_params_mask = torch.cat(base_model_params_mask, dim=0)

            adapter_params = torch.nn.Parameter(student_net_params[~base_model_params_mask])
            base_model_params = torch.nn.Parameter(student_net_params[base_model_params_mask])

            student_net_params = torch.zeros_like(student_net_params)
            student_net_params.scatter_(0, torch.nonzero(~base_model_params_mask).squeeze(), adapter_params)
            student_net_params.scatter_(0, torch.nonzero(base_model_params_mask).squeeze(), base_model_params)

            idx_A = torch.nonzero(~base_model_params_mask, as_tuple=False).view(-1)
            idx_B = torch.nonzero(base_model_params_mask, as_tuple=False).view(-1)

        else:
            student_net_params = []
            for name, param in student_net.named_parameters():
                student_net_params.append(param.reshape(-1).requires_grad_(True))
            student_net_params = torch.nn.Parameter(torch.cat(student_net_params, dim=0))

        load_net = get_network(name=expert_network_name, **expert_network_dict).to(device)
        student_net = ReparamModule(student_net)

        if enable_log:
            tag2 = wandb.init(project=project_name, name=time.strftime('%m%d%H%M') + 'distill' + distill_sequence,
                              config=kwargs,
                              resume="allow")
            tag2.config.update({
                'num_iteration': num_iteration,
                'expert_trajectory_dir_list': expert_trajectory_dir_list,
                'max_start_iter': max_start_iter,
                'min_start_iter': min_start_iter
            })
            tag2.config.update(expert_network_dict)

        self.optimizer_adapter = None
        self.scheduler_adapter = None
        if adapter_params is not None:
            self.optimizer_adapter = create_optimizer(
                [adapter_params],
                distill_lr_adapter,
                distill_optimizer_type
            )
            # if distill_scheduler_type == 'step':
            self.scheduler_adapter = create_scheduler(
                self.optimizer_adapter,
                distill_scheduler_type,
                distill_scheduler_decay_step,
                distill_scheduler_decay_rate
            )

        param_loss_list = []
        with tqdm(range(0, num_iteration)) as pbar:
            for distill_it in pbar:
                if self.optimizer_adapter is not None:
                    self.optimizer_adapter.zero_grad()
                if train_data_path is None:
                    data_path = data_path_list[distill_it % len(data_path_list)]
                else:
                    data_path = train_data_path
                expert_trajectory_dir = expert_trajectory_dir_list[distill_it % len(expert_trajectory_dir_list)]
                expert_iter_list = []
                expert_file_list = {}
                for file in os.listdir(expert_trajectory_dir):
                    if re.search(r'checkpoint_iters_\d', os.path.basename(file)):
                        idx = int(re.findall(r'\d+', os.path.basename(file))[0])
                        expert_file_list[idx] = os.path.join(expert_trajectory_dir, file)
                        expert_iter_list.append(idx)

                expert_file_list = dict(sorted(expert_file_list.items()))
                expert_iter_list = sorted(expert_iter_list)
                max_start_iter = min(max_start_iter, expert_iter_list[-1])  # 给定最大采样点
                min_start_iter = max(min_start_iter, expert_iter_list[0])  # 给定最小采样点
                logger.info(f'max_start_iter: {max_start_iter}, min_start_iter: {min_start_iter}')
                filtered_expert_iter_list = [(i, exp_it) for i, exp_it in enumerate(expert_iter_list) if
                                             min_start_iter <= exp_it < max_start_iter]

                if distill_sequence == 'ord':
                    # 采样方式一：顺序采样
                    start = int(int((len(filtered_expert_iter_list) - 1) / num_iteration) * distill_it)  # 0-500   0  4
                    end = int(
                        int((len(filtered_expert_iter_list) - 1) / num_iteration) * (distill_it + 1))  # 0-500  4  8、
                    random_index = random.choice(range(start, end))
                    start_i, start_it = filtered_expert_iter_list[random_index]
                    end_i = min(start_i + num_expert, len(filtered_expert_iter_list) - 1)
                    end_it = expert_iter_list[end_i]
                elif distill_sequence == 'rand':
                    # 采样方式二：随机采样
                    start_i, start_it = random.choice(filtered_expert_iter_list)
                    end_i = min(start_i + num_expert, len(expert_iter_list) - 1)
                    end_it = expert_iter_list[end_i]

                logger.info(f'start_it -> end_it: {start_it} -> {end_it}')

                # Load Expert Trajectory
                start_expert_params = []
                load_net.load_state_dict(torch.load(expert_file_list[start_it], weights_only=False)['model_state_dict'])
                for param in load_net.parameters():
                    start_expert_params.append(param.data.reshape(-1))
                start_expert_params = torch.cat(start_expert_params, dim=0).to(device).requires_grad_(True)

                target_expert_params = []
                load_net.load_state_dict(torch.load(expert_file_list[end_it], weights_only=False)['model_state_dict'])
                for param in load_net.parameters():
                    target_expert_params.append(param.data.reshape(-1))
                target_expert_params = torch.cat(target_expert_params, dim=0).to(device)

                start_opt_params_dict = {'t': None, 'm': None, 'v': None, 'betas': None, 'eps': None, 'lr': None}
                start_schd_params_dict = {'lr_decay_factor': None, 'lr_decay_step_size': None}
                start_expert_state_dict = torch.load(expert_file_list[start_it], weights_only=False)
                dynamic_optimizer_type = start_expert_state_dict['optimizer_type']
                dynamic_optimizer_state_dict = start_expert_state_dict['optimizer_state_dict']
                dynamic_scheduler_type = start_expert_state_dict['scheduler_type']
                dynamic_scheduler_state_dict = start_expert_state_dict['scheduler_state_dict']

                if dynamic_optimizer_type == 'adam':
                    betas = dynamic_optimizer_state_dict['param_groups'][0]['betas']
                    eps = dynamic_optimizer_state_dict['param_groups'][0]['eps']
                    lr = dynamic_optimizer_state_dict['param_groups'][0]['initial_lr']
                    states = dynamic_optimizer_state_dict['state']
                    if len(states) > 0 and states[0]['step'] is not None and states[0]['exp_avg'] is not None and \
                            states[0]['exp_avg_sq'] is not None:
                        start_opt_params_dict['t'] = states[0]['step']
                        start_opt_params_dict['m'] = []
                        start_opt_params_dict['v'] = []
                        start_opt_params_dict['betas'] = betas
                        start_opt_params_dict['eps'] = eps
                        start_opt_params_dict['lr'] = lr
                        for i, state in enumerate(states.items()):
                            start_opt_params_dict['m'].append(state[1]['exp_avg'].flatten())
                            start_opt_params_dict['v'].append(state[1]['exp_avg_sq'].flatten())
                        start_opt_params_dict['m'] = torch.concat(start_opt_params_dict['m'])
                        start_opt_params_dict['v'] = torch.concat(start_opt_params_dict['v'])
                    else:
                        start_opt_params_dict['t'] = 0
                        start_opt_params_dict['betas'] = betas
                        start_opt_params_dict['eps'] = eps
                        start_opt_params_dict['lr'] = lr

                if dynamic_scheduler_type == 'expdecaylr':
                    start_schd_params_dict['lr_decay_factor'] = start_expert_state_dict['lr_decay_factor']
                    start_schd_params_dict['lr_decay_step_size'] = start_expert_state_dict['lr_decay_step_size']
                    start_schd_params_dict['step'] = dynamic_scheduler_state_dict['last_epoch']

                if base_model_params_mask is not None:
                    base_model_params = start_expert_params.detach().clone().requires_grad_(True)
                    start_expert_params = torch.zeros_like(student_net_params)
                    start_expert_params.scatter_(0, torch.nonzero(~base_model_params_mask).squeeze(), adapter_params)
                    start_expert_params.scatter_(0, torch.nonzero(base_model_params_mask).squeeze(), base_model_params)
                    dynamic_optimizer = get_dynamic_optimizer(optimizer_type=dynamic_optimizer_type,
                                                              params=base_model_params, **start_opt_params_dict)
                else:
                    dynamic_optimizer = get_dynamic_optimizer(optimizer_type=dynamic_optimizer_type,
                                                              params=start_expert_params, **start_opt_params_dict)

                student_params_list = [start_expert_params]
                dynamic_scheduler = None

                if dynamic_scheduler_type is not None:
                    dynamic_scheduler = get_dynamic_scheduler(scheduler_type=dynamic_scheduler_type,
                                                              **start_schd_params_dict)

                # Load data indices
                data_indices_list = []
                for i in range(start_i + 1, end_i + 1):
                    it = expert_iter_list[i]
                    data_indices_list.extend(torch.load(expert_file_list[it], weights_only=False)['data_indices_list'])

                # Distill
                energy_criterion = MSELoss()
                force_criterion = MSELoss()
                student_net.train()