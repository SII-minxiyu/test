import torch
from tqdm import tqdm

from modules.distill.utils.nets import get_network
from torch_geometric.data import Data, Dataset
#
painnet = get_network('painnet')
checkpoint_dir = '/data2_hdd/xh/dis/distill2/fuxian/expert_trajectory/0/best_valid_checkpoint.pt'
ckpt = torch.load(checkpoint_dir)

ckpt = torch.load(checkpoint_dir)
ckpt = torch.load(checkpoint_dir, weights_only=False)
state_dict = ckpt.get('model_state_dict', ckpt)
painnet.load_state_dict(ckpt['model_state_dict'])
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
        for i, batch_data in enumerate(tqdm(train_loader)):
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
            # out = model(batch_data.z, batch_data.pos, batch=batch_data.batch)
            if energy_and_force:
                force = -grad(outputs=out, inputs=batch_data.pos,
                              grad_outputs=torch.ones_like(out),
                              create_graph=True, retain_graph=True)[0]
                e_loss = loss_func(out, batch_data.y)
                f_loss = loss_func(force, batch_data.force)
                loss = 1 / p * e_loss + f_loss
            else:
                force = -grad(outputs=out, inputs=batch_data.pos,
                              grad_outputs=torch.ones_like(out),
                              create_graph=True, retain_graph=True)[0]
                loss = loss_func(force, batch_data.force)
            loss.backward()
            optimizer.step()
            iters += batch_data.batch_size
            if scheduler:
                scheduler.step()
            loss_list.append(loss.detach().cpu().item())
            if last_loss is None:
                last_loss = sum(loss_list) / len(loss_list)
            if iters % save_iters == 0:
                last_loss = sum(loss_list) / len(loss_list)
                self._save_checkpoint(save_dir, f'checkpoint_iters_{iters}.pt', model, optimizer, scheduler, last_loss,
                                      None, None, epoch, iters, torch.as_tensor(data_indices_list))
                if data_indices_list is not None:
                    data_indices_list = []
            del loss, out, batch_data
            if energy_and_force:
                del e_loss, f_loss
            torch.cuda.empty_cache()
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
            # out = model(batch_data.z, batch_data.pos, batch=batch_data.batch)
            force = -grad(outputs=out, inputs=batch_data.pos,
                          grad_outputs=torch.ones_like(out),
                          create_graph=True, retain_graph=False)[0]
            preds_force = torch.cat([preds_force, force.detach().cpu()], dim=0)
            targets_force = torch.cat([targets_force, batch_data.force.detach().cpu()], dim=0)
            torch.cuda.empty_cache()
            preds = torch.cat([preds, out.detach().cpu()], dim=0)
            targets = torch.cat([targets, batch_data.y.detach().cpu()], dim=0)
            del batch_data, out, force
            torch.cuda.empty_cache()
        # targets = targets.squeeze(-1)
        input_dict = {"y_true": targets, "y_pred": preds}
        input_dict_force = {"y_true": targets_force, "y_pred": preds_force}
        energy_mae = evaluation.eval(input_dict)['mae']
        force_mae = evaluation.eval(input_dict_force)['mae']
        return 1 / p * energy_mae + force_mae, energy_mae, force_mae


testdataset = torch.load('/home/xh/project/MDsys/modules/distill/test/asprin/asprin_ccsd_test_data_nomean.pt',
                         weights_only=False)


class BasicMDDataset(Dataset):
    def __init__(self, pos, z, y, force, **kwargs):
        self.pos = pos
        self.z = z
        self.y = y
        self.force = force

    def __getitem__(self, index):
        return Data(
            pos=self.pos[index:index + 1].reshape(-1, 3),
            y=self.y[index:index + 1],
            force=self.force[index:index + 1].reshape(-1, 3),
            z=self.z[index:index + 1].reshape(-1)
        )

    def __len__(self):
        return len(self.y)

    def get(self, index):
        self.__getitem__(index)

    def len(self):
        return self.__len__()

    @classmethod
    def from_pt(cls, data_path: str, use_coarse_label: bool = False, **kwargs):
        data = torch.load(data_path, weights_only=False)
        if len(data['pos']) == 2:
            _num_atom_per_mole = data['pos'].shape[0] // data['y'].shape[0]
            return cls(
                pos=data['pos'].reshape(-1, _num_atom_per_mole, 3),
                z=data['z'].reshape(-1, _num_atom_per_mole, 1),
                y=data['y'],
                force=data['force'].reshape(-1, _num_atom_per_mole, 3),
                use_coarse_label=use_coarse_label,
                **kwargs
            )
        return cls(pos=data['pos'], z=data['z'], y=data['y'], force=data['force'], use_coarse_label=use_coarse_label,
                   **kwargs)

    @classmethod
    def from_dataset(cls, dataset, use_coarse_label: bool = False, **kwargs):
        pos = []
        z = []
        y = []
        force = []
        for data in tqdm(dataset):
            pos.append(data.pos)
            z.append(data.z)
            y.append(data.energy)
            force.append(data.force)
        pos = torch.concat(pos, dim=0)
        z = torch.concat(z, dim=0)
        y = torch.concat(y, dim=0)
        force = torch.concat(force, dim=0)
        if len(pos.shape) == 2:
            _num_atom_per_mole = pos.shape[0] // y.shape[0]
            pos = pos.reshape(-1, _num_atom_per_mole, 3)
            z = z.reshape(-1, _num_atom_per_mole, 1)
            force = force.reshape(-1, _num_atom_per_mole, 3)
        return cls(pos=pos, z=z, y=y, force=force, use_coarse_label=use_coarse_label, **kwargs)
from torch_geometric.data import Data, Dataset
class BasicMDDataset(Dataset):
    def __init__(self, pos, z, y, force, **kwargs):
        self.pos = pos
        self.z = z
        self.y = y
        self.force = force

    def __getitem__(self, index):
        return Data(
            pos=self.pos[index:index + 1].reshape(-1, 3),
            y=self.y[index:index + 1],
            force=self.force[index:index + 1].reshape(-1, 3),
            z=self.z[index:index + 1].reshape(-1)
        )

    def __len__(self):
        return len(self.y)

    def get(self, index):
        self.__getitem__(index)

    def len(self):
        return self.__len__()

    @classmethod
    def from_pt(cls, data_path: str, use_coarse_label: bool = False, **kwargs):
        data = torch.load(data_path, weights_only=False)
        if len(data['pos']) == 2:
            _num_atom_per_mole = data['pos'].shape[0] // data['y'].shape[0]
            return cls(
                pos=data['pos'].reshape(-1, _num_atom_per_mole, 3),
                z=data['z'].reshape(-1, _num_atom_per_mole, 1),
                y=data['y'],
                force=data['force'].reshape(-1, _num_atom_per_mole, 3),
                use_coarse_label=use_coarse_label,
                **kwargs
            )
        return cls(pos=data['pos'], z=data['z'], y=data['y'], force=data['force'], use_coarse_label=use_coarse_label,
                   **kwargs)

    @classmethod
    def from_dataset(cls, dataset, use_coarse_label: bool = False, **kwargs):
        pos = []
        z = []
        y = []
        force = []
        for data in tqdm(dataset):
            pos.append(data.pos)
            z.append(data.z)
            y.append(data.energy)
            force.append(data.force)
        pos = torch.concat(pos, dim=0)
        z = torch.concat(z, dim=0)
        y = torch.concat(y, dim=0)
        force = torch.concat(force, dim=0)
        if len(pos.shape) == 2:
            _num_atom_per_mole = pos.shape[0] // y.shape[0]
            pos = pos.reshape(-1, _num_atom_per_mole, 3)
            z = z.reshape(-1, _num_atom_per_mole, 1)
            force = force.reshape(-1, _num_atom_per_mole, 3)
        return cls(pos=pos, z=z, y=y, force=force, use_coarse_label=use_coarse_label, **kwargs)


test_dataset = BasicMDDataset.from_pt('/home/xh/project/MDsys/modules/distill/test/asprin/asprin_ccsd_test_data_nomean.pt')
from torch_geometric.loader import DataLoader
from dig.threedgraph.evaluation import ThreeDEvaluator
evaluation = ThreeDEvaluator()
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
from torch.autograd import grad
preds_force = torch.Tensor([])
targets_force = torch.Tensor([])
preds = torch.Tensor([])
targets = torch.Tensor([])
for batch_data in test_loader:
    batch_data.pos.requires_grad_(True)
    out = painnet(batch_data)
    force = -grad(outputs=out, inputs=batch_data.pos,
                  grad_outputs=torch.ones_like(out),
                  create_graph=True, retain_graph=False)[0]
    preds_force = torch.cat([preds_force, force.detach().cpu()], dim=0)
    targets_force = torch.cat([targets_force, batch_data.force.detach().cpu()], dim=0)
    torch.cuda.empty_cache()
    preds = torch.cat([preds, out.detach().cpu()], dim=0)
    targets = torch.cat([targets, batch_data.y.detach().cpu()], dim=0)
input_dict = {"y_true": targets, "y_pred": preds}
input_dict_force = {"y_true": targets_force, "y_pred": preds_force}
energy_mae = evaluation.eval(input_dict)['mae']
force_mae = evaluation.eval(input_dict_force)['mae']
print(energy_mae, force_mae)
