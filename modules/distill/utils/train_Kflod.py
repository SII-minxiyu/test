import os
import re
import torch
import wandb
import random
import time
from torch.optim import Adam, AdamW, SGD
from torch_geometric.loader import DataLoader
from packaging import version
import torch_geometric
from torch.utils.data import Sampler
from torch.autograd import grad
from torch.optim.lr_scheduler import StepLR, LambdaLR
from tqdm import tqdm
from dig.threedgraph.evaluation import ThreeDEvaluator
import warnings
from modules.distill.utils.datasets import save_dataset, get_and_split_dataset

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import KFold


class IndexTrackingSampler(Sampler):
    def __init__(self, data_source, shuffle=True, seed=None):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self._last_indices = []

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        if self.shuffle:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(indices)
        self._last_indices = indices  # 保存上次采样索引
        return iter(indices)

    def __len__(self):
        return len(self.data_source)

    def get_indices(self):
        """返回最近一次采样的索引"""
        return self._last_indices


# 当使用k折交叉验证时使用如下采样
class IndexTrackingSamplerwithindices(Sampler):
    def __init__(self, indices, shuffle=True, seed=None):
        self.indices = list(indices)
        self.shuffle = shuffle
        self.seed = seed
        self._last_indices = []

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
            self._last_indices = self.indices
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def get_indices(self):
        """返回最近一次采样的索引"""
        return self._last_indices


class Trainer:
    """
    Trainer class for 3DGN methods with wandb logging support
    """

    def __init__(self):
        """Initialize trainer"""
        self.best_valid = float('inf')
        self.best_valid_energy = float('inf')
        self.best_valid_force = float('inf')
        self.best_test = float('inf')
        self.best_test_energy = float('inf')
        self.best_test_force = float('inf')

    def _train_epoch(self, model, optimizer, scheduler, train_loader,
                     energy_and_force, p, loss_func, epoch, save_iters, device, save_dir, e_mean):
        """Training for one epoch"""
        model.train()
        model = model.to(device)  # 直接使用标准模型
        iters = (epoch - 1) * len(train_loader) * train_loader.batch_size
        last_loss = None
        loss_list = []
        data_indices = None
        data_indices_list = None

        for i, batch_data in enumerate(tqdm(train_loader, disable=True)):
            if data_indices is None:
                if self.sampler is not None:
                    data_indices = self.sampler.get_indices()
                    data_indices_list = []
                else:
                    data_indices = list(range(len(train_loader)))
                    data_indices_list = []

            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            if batch_data.y is None:
                setattr(batch_data, 'y', batch_data.energy)

            if data_indices is not None:
                batch_size = batch_data.y.shape[0]
                data_indices_list.extend(data_indices[i * batch_size: (i + 1) * batch_size])

            batch_data.pos.requires_grad_(True)

            out = model(batch_data)
            #out = model(batch_data.z, batch_data.pos, batch=batch_data.batch)

            if energy_and_force:
                force = -grad(outputs=out, inputs=batch_data.pos,
                              grad_outputs=torch.ones_like(out),
                              create_graph=True, retain_graph=True)[0]
                e_loss = loss_func(out, batch_data.y)
                f_loss = loss_func(force, batch_data.force)
                loss = e_loss + p * f_loss
            else:
                force = -grad(outputs=out, inputs=batch_data.pos,
                              grad_outputs=torch.ones_like(out),
                              create_graph=True, retain_graph=True)[0]
                loss = loss_func(force, batch_data.force)
            loss.backward()
            optimizer.step()
            iters += batch_data.batch_size
            # if scheduler:
            #     scheduler.step()
            loss_list.append(loss.detach().cpu().item())
            if last_loss is None:
                last_loss = sum(loss_list) / len(loss_list)

            if iters % save_iters == 0:
                last_loss = sum(loss_list) / len(loss_list)
                self._save_checkpoint(save_dir, f'checkpoint_iters_{iters}.pt', model, optimizer, scheduler, last_loss,
                                      None, None, epoch, iters, torch.as_tensor(data_indices_list))
                if data_indices_list is not None:
                    data_indices_list = []
        return sum(loss_list) / len(loss_list)

    def _evaluate(self, model, data_loader, energy_and_force,
                  p, evaluation, device, e_mean):
        """Evaluation step"""
        model.eval()
        preds = torch.Tensor([])
        targets = torch.Tensor([])
        preds_force = torch.Tensor([])
        targets_force = torch.Tensor([])

        for batch_data in tqdm(data_loader):
            batch_data = batch_data.to(device)

            if batch_data.y is None:
                setattr(batch_data, 'y', batch_data.energy)

            batch_data.pos.requires_grad_(True)

            out = model(batch_data)
            #out = model(batch_data.z, batch_data.pos, batch=batch_data.batch)
            force = -grad(outputs=out, inputs=batch_data.pos,
                          grad_outputs=torch.ones_like(out),
                          create_graph=True, retain_graph=False)[0]
            preds_force = torch.cat([preds_force, force.detach().cpu()], dim=0)
            targets_force = torch.cat([targets_force, batch_data.force.detach().cpu()], dim=0)
            torch.cuda.empty_cache()
            preds = torch.cat([preds, out.detach().cpu()], dim=0)
            targets = torch.cat([targets, batch_data.y.detach().cpu()], dim=0)
        #targets = targets.squeeze(-1)
        input_dict = {"y_true": targets, "y_pred": preds}
        input_dict_force = {"y_true": targets_force, "y_pred": preds_force}
        energy_mae = evaluation.eval(input_dict)['mae']
        force_mae = evaluation.eval(input_dict_force)['mae']
        return energy_mae + p * force_mae, energy_mae, force_mae

    def _save_checkpoint(self, save_dir, filename, model, optimizer,
                         scheduler, train_err, valid_err, test_err, epoch, iters, data_indices_list=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'iters': iters,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_type': self.optimizer_type,
            'scheduler_type': None,
            'lr_decay_step_size': None,
            'lr_decay_factor': None,
            'best_valid_mae': self.best_valid,
            'best_valid_force_mae': self.best_valid_force,
            'best_valid_energy_mae': self.best_valid_energy,
            'best_test_mae': self.best_test,
            'best_test_force_mae': self.best_test_force,
            'best_test_energy_mae': self.best_test_energy,
            'train_err': train_err,
            'valid_err': valid_err,
            'test_err': test_err,
            'data_indices_list': data_indices_list
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            checkpoint['scheduler_type'] = self.scheduler_type
            checkpoint['lr_decay_step_size'] = self.lr_decay_step_size
            checkpoint['lr_decay_factor'] = self.lr_decay_factor

        torch.save(checkpoint, os.path.join(save_dir, filename))
        print(f'Saved checkpoint to {os.path.join(save_dir, filename)}')

    def _load_checkpoint(self, checkpoint_path, model, optimizer, scheduler):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found.")

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_valid = checkpoint.get('best_valid_mae', float('inf'))

        print(f"Loaded checkpoint from '{checkpoint_path}' (Epoch {checkpoint['epoch']})")
        return checkpoint['epoch']

    def train(self, device, train_dataset, valid_dataset, test_dataset,
              model, pattern=None, loss_func=None, evaluation=None, epochs=500, batch_size=32, patience=30,
              vt_batch_size=32, optimizer_name='Adam', lr=0.0005,
              scheduler_name=None, lr_decay_factor=0.5,
              lr_decay_step_size=50, weight_decay=0,
              energy_and_force=False, p=100, save_dir='',
              project_name='3DGN-Training',
              val_step=10, test_step=10, save_iters=50, save_epochs=100,
              finetune_dataset=None, finetune_epochs=None, merge_adapter=True, finetune_batch_size=32,
              pretrain_dataset=None, pretrain_batch_size=32, pretrain_epochs=None,
              enable_log=True, wandb_run_id=None, **kwargs):

        print("开始:", pattern)
        project_name = project_name + pattern.lower()
        if pattern.lower() == 'pretrain':
            assert pretrain_dataset is not None
            train_dataset = pretrain_dataset
            batch_size = pretrain_batch_size
            epochs = pretrain_epochs
        elif pattern.lower() == 'finetune':
            assert finetune_dataset is not None
            train_dataset = finetune_dataset
            batch_size = finetune_batch_size
            epochs = finetune_epochs
        elif pattern.lower() == 'expert_trajectory':
            assert train_dataset is not None
        assert test_dataset is not None

        save_dataset(train_dataset, os.path.join(save_dir, 'train_dataset.pt'))
        if valid_dataset:
            save_dataset(valid_dataset, os.path.join(save_dir, 'valid_dataset.pt'))
        if test_dataset:
            save_dataset(test_dataset, os.path.join(save_dir, 'test_dataset.pt'))
        if enable_log:
            tag = wandb.init(project=project_name, name=time.strftime('%m%d%H%M%S'), resume="allow")
            tag.config.update({
                "pattern": pattern,
                "epochs": epochs,
                "batch_size": batch_size,
                "vt_batch_size": vt_batch_size,
                "learning_rate": lr,
                "optimizer": optimizer_name,
                "scheduler": scheduler_name,
            })
        model = model.to(device)
        num_params = sum(param.numel() for param in model.parameters())
        print(f'#Params: {num_params}')
        for name, param in model.named_parameters():
            print(f'{name} requires grad: {param.requires_grad}')

        if str.lower(optimizer_name) == 'adam':
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif str.lower(optimizer_name) == 'adamw':
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif str.lower(optimizer_name) == 'sgd':
            optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")

        scheduler = None
        if scheduler_name is not None:
            if str.lower(scheduler_name) == 'steplr':
                scheduler = StepLR(optimizer, step_size=lr_decay_step_size,
                                   gamma=lr_decay_factor)
            elif str.lower(scheduler_name) == 'expdecaylr':
                decay_lambda = lambda step: max(lr_decay_factor ** (step / lr_decay_step_size), 1e-6)
                scheduler = LambdaLR(optimizer, lr_lambda=decay_lambda)
            else:
                scheduler = None
        self.optimizer_type = str.lower(optimizer_name)
        self.scheduler_type = str.lower(scheduler_name)
        self.lr_decay_step_size = lr_decay_step_size
        self.lr_decay_factor = lr_decay_factor
        print(f'energy_and_force: {energy_and_force}')

        self._save_checkpoint(save_dir, 'checkpoint_epoch_0.pt',
                              model, optimizer, scheduler, None, None, None, 0, 0)
        self._save_checkpoint(save_dir, 'checkpoint_iters_0.pt',
                              model, optimizer, scheduler, None, None, None, 0, 0)
        if valid_dataset is None:
            kfold = KFold(n_splits=20, shuffle=True, random_state=42)
            fold_results = []

            for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
                print(f"\n=====Fold {fold + 1}", flush=True)

                # 划分数据集
                train_subset = torch.utils.data.Subset(train_dataset, train_idx)
                valid_subset = torch.utils.data.Subset(train_dataset, val_idx)
                self.sampler = IndexTrackingSamplerwithindices(train_subset.indices, shuffle=True)
                #sampler2 = IndexTrackingSampler(train_dataset,shuffle=True)
                train_loader = DataLoader(train_subset.dataset, batch_size=batch_size, sampler=self.sampler)
                valid_loader = DataLoader(valid_subset, batch_size=vt_batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=vt_batch_size, shuffle=False)

                #sampler2 = IndexTrackingSampler(train_subset.dataset, shuffle=False)

                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if evaluation is None:
                    evaluation = ThreeDEvaluator()

                if loss_func == 'l1':
                    loss_func = torch.nn.L1Loss()
                else:
                    loss_func = torch.nn.MSELoss()
                start_epoch = 1
                # if fold > 0:
                #     checkpoint_path = os.path.join(save_dir, 'checkpoint_epoch_0.pt')
                #     assert os.path.exists(checkpoint_path)
                #     start_epoch = self._load_checkpoint(checkpoint_path, model, optimizer, scheduler) + 1
                # 如下执行，使得其余4折直接省略，大大缩减时间，如需严格执行，使用上面注释代码
                if save_dir:
                    checkpoint_files = [f for f in os.listdir(save_dir) if re.match(r'checkpoint_epoch_\d+\.pt', f)]
                    if checkpoint_files:
                        latest_checkpoint = max(checkpoint_files, key=lambda f: int(re.search(r'\d+', f).group()))
                        checkpoint_path = os.path.join(save_dir, latest_checkpoint)
                        start_epoch = self._load_checkpoint(checkpoint_path, model, optimizer,
                                                            scheduler) + 1
                indices = train_subset.indices
                ys = train_subset.dataset.y[indices]  # 形状 [800, 1]
                train_e_mean = ys.mean()
                valid_e_mean = train_e_mean
                test_e_mean = train_e_mean
                early_stop_counter = 0
                for epoch in range(start_epoch, epochs + 1):
                    print(f"\n=====Epoch {epoch}", flush=True)

                    # Training
                    print('\nTraining...', flush=True)
                    train_err = self._train_epoch(model, optimizer, scheduler, train_loader,
                                                  energy_and_force, p, loss_func, epoch, save_iters, device, save_dir,
                                                  train_e_mean)
                    print({'train_err': train_err, 'lr': optimizer.param_groups[0]['lr']})
                    scheduler.step()
                    if enable_log:
                        tag.log({'train_err': train_err}, step=epoch)
                        tag.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)

                    # Validation
                    if valid_loader is not None and epoch % val_step == 0:
                        print('\n\nEvaluating...', flush=True)
                        valid_err, energy_valid_err, force_valid_err = self._evaluate(model, valid_loader,
                                                                                      energy_and_force, p, evaluation,
                                                                                      device, valid_e_mean)
                        if energy_and_force == True:
                            if valid_err < self.best_valid:
                                self.best_valid = valid_err
                                early_stop_counter = 0
                                print(f"EarlyStopping counter: {early_stop_counter}")
                                if save_dir:
                                    self._save_checkpoint(save_dir, 'best_valid_checkpoint.pt',
                                                          model, optimizer, scheduler, None, valid_err, None,
                                                          epoch, (epoch - 1) * len(train_loader))
                            else:
                                early_stop_counter += 1
                                print(f"EarlyStopping counter: {early_stop_counter}/{patience}")
                                # if early_stop_counter >= patience:
                                #     print("调整学习率")
                                #     #scheduler.step()
                                #     early_stop_counter = 0
                            if force_valid_err < self.best_valid_force:
                                self.best_valid_force = force_valid_err
                                if save_dir:
                                    self._save_checkpoint(save_dir, 'best_valid_force_checkpoint.pt',
                                                          model, optimizer, scheduler, None, valid_err, None,
                                                          epoch, (epoch - 1) * len(train_loader))
                        else:
                            if force_valid_err < self.best_valid_force:
                                self.best_valid_force = force_valid_err
                                early_stop_counter = 0
                                print(f"EarlyStopping counter: {early_stop_counter}")
                                if save_dir:
                                    self._save_checkpoint(save_dir, 'best_valid_force_checkpoint.pt',
                                                          model, optimizer, scheduler, None, valid_err, None,
                                                          epoch, (epoch - 1) * len(train_loader))
                            else:
                                early_stop_counter += 1
                                print(f"EarlyStopping counter: {early_stop_counter}/{patience}")
                                # if early_stop_counter >= patience:
                                #     print("调整学习率")
                                #     scheduler.step()
                                #     early_stop_counter = 0
                        if energy_valid_err < self.best_valid_energy:
                            self.best_valid_energy = energy_valid_err
                            if save_dir:
                                self._save_checkpoint(save_dir, 'best_valid_energy_checkpoint.pt',
                                                      model, optimizer, scheduler, None, valid_err, None,
                                                      epoch, (epoch - 1) * len(train_loader))

                        print({'valid_err': valid_err, 'valid_energy_err': energy_valid_err,
                               'valid_force_err': force_valid_err, \
                               'best_valid': self.best_valid, 'best_valid_energy': self.best_valid_energy,
                               'best_valid_force': self.best_valid_force})
                        if enable_log:
                            tag.log({'valid_err': valid_err, 'valid_energy_err': energy_valid_err,
                                     'valid_force_err': force_valid_err, 'best_valid': self.best_valid,
                                     'best_valid_energy': self.best_valid_energy,
                                     'best_valid_force': self.best_valid_force},
                                    step=epoch)
                            # Testing
                    if test_loader is not None and epoch % test_step == 0:
                        print('\n\nTesting...', flush=True)
                        test_err, energy_test_err, force_test_err = self._evaluate(model, test_loader,
                                                                                   energy_and_force,
                                                                                   p, evaluation, device,
                                                                                   test_e_mean)
                        if test_err < self.best_test:
                            self.best_test = test_err
                            if save_dir:
                                self._save_checkpoint(save_dir, 'best_test_checkpoint.pt',
                                                      model, optimizer, scheduler, None, None, test_err,
                                                      epoch, (epoch - 1) * len(train_loader))

                        if force_test_err < self.best_test_force:
                            self.best_test_force = force_test_err
                            if save_dir:
                                self._save_checkpoint(save_dir, 'best_test_force_checkpoint.pt',
                                                      model, optimizer, scheduler, None, None, test_err,
                                                      epoch, (epoch - 1) * len(train_loader))
                        if energy_test_err < self.best_test_energy:
                            self.best_test_energy = energy_test_err
                            if save_dir:
                                self._save_checkpoint(save_dir, 'best_test_energy_checkpoint.pt',
                                                      model, optimizer, scheduler, None, None, test_err,
                                                      epoch, (epoch - 1) * len(train_loader))

                        print(
                            {'test_err': test_err, 'energy_test_err': energy_test_err, 'force_test_err': force_test_err, \
                             'best_test': self.best_test, 'best_test_energy': self.best_test_energy,
                             'best_test_force': self.best_test_force})

                        if enable_log:
                            tag.log(
                                {'test_err': test_err, 'energy_test_err': energy_test_err,
                                 'force_test_err': force_test_err,
                                 'best_test': self.best_test, 'best_test_energy': self.best_test_energy,
                                 'best_test_force': self.best_test_force}, step=epoch)
                            # Periodic checkpoint saving
                    if save_dir and epoch % save_epochs == 0:
                        self._save_checkpoint(save_dir, f'checkpoint_epoch_{epoch}.pt',
                                              model, optimizer, scheduler, train_err, None, None, epoch,
                                              epoch * len(train_loader))
                fold_results.append({
                    'valid_err': self.best_valid,
                    'test_err': self.best_test,
                    'valid_energy_err': self.best_valid_energy,
                    'valid_force_err': self.best_valid_force,
                    'test_energy_err': self.best_test_energy,
                    'test_force_err': self.best_test_force
                })
            # Compute the average of all folds
            avg_valid_err = sum(f['valid_err'] for f in fold_results) / len(fold_results)
            avg_test_err = sum(f['test_err'] for f in fold_results) / len(fold_results)
            avg_valid_energy_err = sum(f['valid_energy_err'] for f in fold_results) / len(fold_results)
            avg_valid_force_err = sum(f['valid_force_err'] for f in fold_results) / len(fold_results)
            avg_test_energy_err = sum(f['test_energy_err'] for f in fold_results) / len(fold_results)
            avg_test_force_err = sum(f['test_force_err'] for f in fold_results) / len(fold_results)
            print(f"\nCross-validation results:")
            print(f"Avg Valid Error: {avg_valid_err}")
            print(f"Avg Test Error: {avg_test_err}")
            print(f"Avg Valid Energy Error: {avg_valid_energy_err}")
            print(f"Avg Valid Force Error: {avg_valid_force_err}")
            print(f"Avg Test Energy Error: {avg_test_energy_err}")
            print(f"Avg Test Force Error: {avg_test_force_err}")
            print("完成:", pattern)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            self.sampler = IndexTrackingSampler(train_dataset, shuffle=True)

            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self._save_checkpoint(save_dir, 'checkpoint_epoch_0.pt',
                                  model, optimizer, scheduler, None, None, None, 0, 0)
            self._save_checkpoint(save_dir, 'checkpoint_iters_0.pt',
                                  model, optimizer, scheduler, None, None, None, 0, 0)
            if evaluation is None:
                evaluation = ThreeDEvaluator()

            if loss_func == 'l1':
                loss_func = torch.nn.L1Loss()
            else:
                loss_func = torch.nn.MSELoss()

            start_epoch = 1
            if save_dir:
                checkpoint_files = [f for f in os.listdir(save_dir) if re.match(r'checkpoint_epoch_\d+\.pt', f)]
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=lambda f: int(re.search(r'\d+', f).group()))
                    checkpoint_path = os.path.join(save_dir, latest_checkpoint)
                    start_epoch = self._load_checkpoint(checkpoint_path, model, optimizer,
                                                        scheduler) + 1
            early_stop_counter = 0
            train_e_mean = float(train_dataset.y.mean())
            valid_e_mean = float(valid_dataset.y.mean())
            test_e_mean = float(test_dataset.y.mean())
            for epoch in range(start_epoch, epochs + 1):
                print(f"\n=====Epoch {epoch}", flush=True)

                # Training
                print('\nTraining...', flush=True)
                train_err = self._train_epoch(model, optimizer, scheduler, train_loader,
                                              energy_and_force, p, loss_func, epoch, save_iters, device, save_dir,
                                              train_e_mean)
                print({'train_err': train_err, 'lr': optimizer.param_groups[0]['lr']})
                if enable_log:
                    tag.log({'train_err': train_err}, step=epoch)
                    tag.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)

                # Validation
                if valid_loader is not None and epoch % val_step == 0:
                    print('\n\nEvaluating...', flush=True)
                    valid_err, energy_valid_err, force_valid_err = self._evaluate(model, valid_loader, energy_and_force,
                                                                                  p, evaluation, device, valid_e_mean)

                    # Save checkpoint on validation improvement
                    if valid_err < self.best_valid:
                        self.best_valid = valid_err
                        early_stop_counter = 0
                        if save_dir:
                            self._save_checkpoint(save_dir, 'best_valid_checkpoint.pt',
                                                  model, optimizer, scheduler, None, valid_err, None,
                                                  epoch, (epoch - 1) * len(train_loader))
                    else:
                        early_stop_counter += 1
                        print(f"EarlyStopping counter: {early_stop_counter}/{patience}")

                    if force_valid_err < self.best_valid_force:
                        self.best_valid_force = force_valid_err
                        if save_dir:
                            self._save_checkpoint(save_dir, 'best_valid_force_checkpoint.pt',
                                                  model, optimizer, scheduler, None, valid_err, None,
                                                  epoch, (epoch - 1) * len(train_loader))
                    if energy_valid_err < self.best_valid_energy:
                        self.best_valid_energy = energy_valid_err
                        if save_dir:
                            self._save_checkpoint(save_dir, 'best_valid_energy_checkpoint.pt',
                                                  model, optimizer, scheduler, None, valid_err, None,
                                                  epoch, (epoch - 1) * len(train_loader))

                    print({'valid_err': valid_err, 'valid_energy_err': energy_valid_err,
                           'valid_force_err': force_valid_err, \
                           'best_valid': self.best_valid, 'best_valid_energy': self.best_valid_energy,
                           'best_valid_force': self.best_valid_force})
                    if enable_log:
                        tag.log({'valid_err': valid_err, 'valid_energy_err': energy_valid_err,
                                 'valid_force_err': force_valid_err, 'best_valid': self.best_valid,
                                 'best_valid_energy': self.best_valid_energy,
                                 'best_valid_force': self.best_valid_force},
                                step=epoch)
                        # Testing
                if test_loader is not None and epoch % test_step == 0:
                    print('\n\nTesting...', flush=True)
                    test_err, energy_test_err, force_test_err = self._evaluate(model, test_loader,
                                                                               energy_and_force,
                                                                               p, evaluation, device,
                                                                               test_e_mean)
                    if test_err < self.best_test:
                        self.best_test = test_err
                        if save_dir:
                            self._save_checkpoint(save_dir, 'best_test_checkpoint.pt',
                                                  model, optimizer, scheduler, None, None, test_err,
                                                  epoch, (epoch - 1) * len(train_loader))

                    if force_test_err < self.best_test_force:
                        self.best_test_force = force_test_err
                        if save_dir:
                            self._save_checkpoint(save_dir, 'best_test_force_checkpoint.pt',
                                                  model, optimizer, scheduler, None, None, test_err,
                                                  epoch, (epoch - 1) * len(train_loader))
                    if energy_test_err < self.best_test_energy:
                        self.best_test_energy = energy_test_err
                        if save_dir:
                            self._save_checkpoint(save_dir, 'best_test_energy_checkpoint.pt',
                                                  model, optimizer, scheduler, None, None, test_err,
                                                  epoch, (epoch - 1) * len(train_loader))

                    print({'test_err': test_err, 'energy_test_err': energy_test_err, 'force_test_err': force_test_err, \
                           'best_test': self.best_test, 'best_test_energy': self.best_test_energy,
                           'best_test_force': self.best_test_force})

                    if enable_log:
                        tag.log(
                            {'test_err': test_err, 'energy_test_err': energy_test_err,
                             'force_test_err': force_test_err,
                             'best_test': self.best_test, 'best_test_energy': self.best_test_energy,
                             'best_test_force': self.best_test_force}, step=epoch)
                        # Periodic checkpoint saving
                if save_dir and epoch % save_epochs == 0:
                    self._save_checkpoint(save_dir, f'checkpoint_epoch_{epoch}.pt',
                                          model, optimizer, scheduler, train_err, None, None, epoch,
                                          epoch * len(train_loader))
        return self.best_test, self.best_test_energy, self.best_test_force
