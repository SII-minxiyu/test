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


# --------------------
# Helper for differentiable assembly of flat parameter vector
# --------------------
def assemble_flat_params(adapter_params, base_params, base_mask):
    """
    Build a full-length flat parameter vector from adapter (low-rank) params and base params
    in a *differentiable* way so gradients can flow back to adapter_params and base_params.

    adapter_params: 1D tensor (requires_grad True) or None
    base_params: 1D tensor (requires_grad may be True) or None
    base_mask: boolean 1D tensor (True where base params live)
    """
    device = adapter_params.device if adapter_params is not None else base_params.device
    dtype = adapter_params.dtype if adapter_params is not None else base_params.dtype
    L = base_mask.shape[0]
    full = torch.zeros(L, device=device, dtype=dtype)

    # indices for adapter and base parts
    base_idx = base_mask.nonzero(as_tuple=True)[0]
    adapter_idx = (~base_mask).nonzero(as_tuple=True)[0]

    if adapter_params is not None:
        # scatter_add is differentiable wrt adapter_params
        full = full.scatter_add(0, adapter_idx.to(device), adapter_params)
    if base_params is not None:
        full = full.scatter_add(0, base_idx.to(device), base_params)

    return full


class AdapterMTT7:
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

        #student_net_params = None
        base_model_params_mask = None
        adapter_params = None
        if enable_adapter and adapter_dict:
            print("学生模型构建Adapter")
            student_net = get_adapter_model(adapter_dict, student_net).to(device)
            student_net.train()

            flat_param_list = []
            base_model_params_mask_list = []
            for name, param in student_net.named_parameters():
                if 'lora_' in name:
                    base_model_params_mask_list.append(torch.zeros_like(param.reshape(-1), dtype=torch.bool))
                else:
                    base_model_params_mask_list.append(torch.ones_like(param.reshape(-1), dtype=torch.bool))
                flat_param_list.append(param.reshape(-1))

            student_net_flat = torch.nn.Parameter(torch.cat(flat_param_list, dim=0))
            base_model_params_mask = torch.cat(base_model_params_mask_list, dim=0).to(device)

            # MODIFICATION: create adapter_params and base_params as separate Parameters (cloned)
            adapter_params = torch.nn.Parameter(student_net_flat[~base_model_params_mask].clone())
            base_model_params = torch.nn.Parameter(student_net_flat[base_model_params_mask].clone())

            # print requires_grad for module params for debugging only
            for name, param in student_net.named_parameters():
                print(f'{name} requires grad: {param.requires_grad}')
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            # We will *assemble* full flat params on the fly using `assemble_flat_params` so
            # that the resulting full vector remains differentiable with respect to
            # adapter_params and base_model_params. Do NOT use in-place assignment (.scatter_) which
            # copies data and breaks the autograd chain.
        else:
            student_net_flat_list = []
            for name, param in student_net.named_parameters():
                student_net_flat_list.append(param.reshape(-1).requires_grad_(True))
            student_net = ReparamModule(student_net)
            student_net_params = torch.nn.Parameter(torch.cat(student_net_flat_list, dim=0))

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
        print("蒸馏数据路径")
        print(train_data_path)
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
                elif dynamic_scheduler_type == 'steplr':
                    start_schd_params_dict['lr_decay_factor'] = start_expert_state_dict['lr_decay_factor']
                    start_schd_params_dict['lr_decay_step_size'] = start_expert_state_dict['lr_decay_step_size']
                    start_schd_params_dict['step'] = dynamic_scheduler_state_dict['last_epoch']
                # --------------------
                # MODIFICATION here: build initial base_current and student full vector in a differentiable way
                # --------------------
                if base_model_params_mask is not None:
                    # Extract only base-region parameters from the expert start vector
                    #base_start = start_expert_params[base_model_params_mask].detach().clone().requires_grad_(True)
                    base_start = start_expert_params.detach().clone().requires_grad_(True)
                    # base_current holds base-region parameters and will be updated by dynamic optimizer
                    base_current = base_start

                    # Assemble initial full params (depends on adapter_params and base_current)
                    start_expert_full = assemble_flat_params(adapter_params=adapter_params, base_params=base_current,
                                                             base_mask=base_model_params_mask)

                    dynamic_optimizer = get_dynamic_optimizer(optimizer_type=dynamic_optimizer_type,
                                                              params=base_current, **start_opt_params_dict)
                else:
                    # no adapter mode - use full start_expert_params directly
                    base_current = None
                    start_expert_full = start_expert_params
                    dynamic_optimizer = get_dynamic_optimizer(optimizer_type=dynamic_optimizer_type,
                                                              params=start_expert_params, **start_opt_params_dict)

                student_params_list = [start_expert_full]
                dynamic_scheduler = None

                if dynamic_scheduler_type is not None:
                    dynamic_scheduler = get_dynamic_scheduler(scheduler_type=dynamic_scheduler_type,
                                                              **start_schd_params_dict)

                # Load data indices
                # data_indices_list = []
                # for i in range(start_i + 1, end_i + 1):
                #     it = expert_iter_list[i]
                #     data_indices_list.extend(torch.load(expert_file_list[it], weights_only=False)['data_indices_list'])

                # Distill
                energy_criterion = MSELoss()
                force_criterion = MSELoss()
                student_net.train()

                # num_params = sum([np.prod(param.size()) for param in (student_net.parameters())])

                if train_data_path is None:
                    dataset = DistillDataset.from_pt(data_path=data_path, use_coarse_label=use_coarse_label)
                else:
                    dataset = DistillDataset.from_pt(data_path=data_path, use_coarse_label=False)
                sampler = IndexControlSampler(data_source=dataset)
                data_len = num_expert * 100
                total_len = len(dataset)
                if data_len > total_len:
                    data_len = total_len
                indices = np.random.choice(total_len, size=data_len, replace=False)
                sampler = IndexControlSampler(data_source=dataset)
                sampler.set_indices(indices.tolist())

                dataloder = DataLoader(dataset=dataset, batch_size=distill_batch_size, sampler=sampler)
                floss = 0
                eloss = 0
                # e_mean = dataset.y.mean().to(device)
                for batch_data in tqdm(dataloder, disable=True):
                    batch_data = batch_data.to(device)
                    if batch_data.y is None:
                        setattr(batch_data, 'y', batch_data.energy)
                    if distill_force:
                        batch_data.pos.requires_grad_(True)

                    # Re-assemble full params BEFORE forward so that the forward depends on adapter_params
                    if adapter_params is not None:
                        # full depends on adapter_params and base_current -> differentiable
                        current_full = assemble_flat_params(adapter_params=adapter_params, base_params=base_current,
                                                            base_mask=base_model_params_mask)
                    else:
                        current_full = student_params_list[-1]

                    output = student_net(batch_data, flat_param=current_full)
                    # output = student_net(z=batch_data.z, pos=batch_data.pos, batch=batch_data.batch, flat_param=student_params_list[-1])
                    if distill_energy and distill_force:
                        loss = 1 / p * energy_criterion(output, batch_data.y)
                        eloss += energy_criterion(output, batch_data.y)
                        force = - \
                        torch.autograd.grad(outputs=output, inputs=batch_data.pos, grad_outputs=torch.ones_like(output),
                                            create_graph=True, retain_graph=True)[0]
                        f_loss = force_criterion(force, batch_data.force)
                        floss += f_loss
                        loss += f_loss
                    elif distill_force:
                        force = - \
                        torch.autograd.grad(outputs=output, inputs=batch_data.pos, grad_outputs=torch.ones_like(output),
                                            create_graph=True, retain_graph=True)[0]
                        loss = force_criterion(force, batch_data.force)
                        floss += loss

                    # compute gradient of loss w.r.t the *full* flat param vector (so we can extract base-region grads)
                    grad = torch.autograd.grad(loss, current_full, create_graph=True, retain_graph=True)[0]

                    if torch.isnan(grad).sum().item() > 0:
                        logger.info(f'nan values exist in dynamic grad.')
                    if adapter_params is not None:
                        if adapter_params.grad is not None:
                            logger.info(f'adapter_params.grad sum: {torch.sum(adapter_params.grad)}')

                    # Update base-region params via dynamic optimizer using gradient sliced to base-region
                    if base_model_params_mask is not None:
                        grad_base = grad[base_model_params_mask]
                        next_base = dynamic_optimizer.step(params=base_current, grad=grad_base,
                                                           lr=dynamic_scheduler.step_lr(dynamic_optimizer.get_lr()))
                    else:
                        # no adapter: dynamic optimizer handles full param vector
                        next_base = dynamic_optimizer.step(params=current_full, grad=grad,
                                                           lr=dynamic_scheduler.step_lr(dynamic_optimizer.get_lr()))
                    param_loss = torch.tensor(0.0).to(device)
                    param_dist = torch.tensor(0.0).to(device)
                    param_loss = torch.nn.functional.mse_loss(next_base,
                                                               target_expert_params, reduction="sum")
                    param_dist = torch.nn.functional.mse_loss(start_expert_params,
                                                               target_expert_params, reduction="sum")
                    param_loss_list.append(param_loss.detach().cpu().item())
                    param_loss.backward()
                    self.optimizer_adapter.step()
                    self.scheduler_adapter.step()
                    # Update tracking variables
                    if adapter_params is not None:
                        # assemble next full vector (depends on adapter_params and next_base)
                        next_full = assemble_flat_params(adapter_params=adapter_params, base_params=next_base,
                                                         base_mask=base_model_params_mask)
                        student_params_list.append(next_full)
                        base_current = next_base
                    else:
                        student_params_list.append(next_base)
                        base_current = None
                logger.info(f'eloss: {eloss / len(dataloder)}, floss: {floss / len(dataloder)}')
                param_loss = torch.tensor(0.0).to(device)
                param_dist = torch.tensor(0.0).to(device)
                if base_model_params_mask is not None:
                    # final base-region is in student_params_list[-1][base_model_params_mask]
                    param_loss += torch.nn.functional.mse_loss(student_params_list[-1][base_model_params_mask],
                                                               target_expert_params, reduction="sum")
                    param_dist += torch.nn.functional.mse_loss(start_expert_params,
                                                               target_expert_params, reduction="sum")
                else:
                    param_loss += torch.nn.functional.mse_loss(student_params_list[-1], target_expert_params,
                                                               reduction="sum")
                    param_dist += torch.nn.functional.mse_loss(start_expert_params, target_expert_params,
                                                               reduction="sum")
                    # 0-250  distill - 1.

                # 不÷基本参数变化，强调前期参数变化
                #param_loss /= (param_dist + 1e-9)

                param_loss_list.append(param_loss.detach().cpu().item())

                if adapter_params is not None:
                    if adapter_params.grad is not None:
                        logger.info(f'adapter_params.grad sum: {torch.sum(adapter_params.grad)}')

                if max_grad_norm_clip:
                    if self.optimizer_adapter is not None:
                        torch.nn.utils.clip_grad_norm_(parameters=adapter_params, max_norm=max_grad_norm_clip)

                if enable_log:
                    tag2.log(
                        {"start_it": start_it, "end_it": end_it, "distill/param_loss": param_loss.detach().cpu().item(),
                         "distill/mean_param_loss": sum(param_loss_list) / len(param_loss_list),
                         "distill/min_param_loss": min(param_loss_list), "distill/floss": floss / len(dataloder),
                         "distill/eloss": eloss / len(dataloder)}, step=distill_it)
                    tag2.log({"distill_adapter_lr": self.optimizer_adapter.param_groups[0]["lr"]}, step=distill_it)
                    # tag2.log({"target_label_loss": target_loss}, step=distill_it)
                logger.info(
                    f'iter: {distill_it}, params_loss: {param_loss.detach().cpu().item()}, mean_params_loss: {sum(param_loss_list) / len(param_loss_list)}, min_params_loss: {min(param_loss_list)}')
                # logger.info(f'target_label_loss:{target_loss.detach().cpu().item()}')

                if self.optimizer_adapter is not None:
                    # param_loss depends (through the dynamic steps) on adapter_params when using the differentiable assembly above
                    param_loss.backward()
                    self.optimizer_adapter.step()
                    logger.info(f'distill_adapter_lr: {self.optimizer_adapter.param_groups[0]["lr"]}')
                    if self.scheduler_adapter is not None:
                        self.scheduler_adapter.step()
                pbar.set_postfix(param_loss=param_loss.detach().cpu(), step=distill_it)

                # Save
                if adapter_params is not None and distill_it % save_step == 0:
                    template_net = get_network(name=expert_network_name, **expert_network_dict).to(device)
                    template_net = get_adapter_model(adapter_dict, template_net).to(device)
                    template_net = ReparamModule(template_net)
                    restored_adapter_module = restore_module_parameters(template_net, student_params_list[-1])
                    restored_adapter_module.module.save_pretrained(os.path.join(save_dir, str(distill_it) + '_adapter'))
                    del restored_adapter_module
                    del template_net
                    gc.collect()
                    torch.cuda.empty_cache()
                ## 最后一个也保存 从0-499
                if adapter_params is not None and distill_it == num_iteration - 1:
                    template_net = get_network(name=expert_network_name, **expert_network_dict).to(device)
                    template_net = get_adapter_model(adapter_dict, template_net).to(device)
                    template_net = ReparamModule(template_net)
                    restored_adapter_module = restore_module_parameters(template_net, student_params_list[-1])
                    restored_adapter_module.module.save_pretrained(os.path.join(save_dir, str(distill_it) + '_adapter'))
                    del restored_adapter_module
                    del template_net
                    gc.collect()
                    torch.cuda.empty_cache()

                for _ in student_params_list:
                    del _
                del output, batch_data, grad
                if distill_force:
                    del force
                gc.collect()
                torch.cuda.empty_cache()

        if enable_log:
            tag2.finish()
