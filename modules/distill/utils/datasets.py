import os
import os.path as osp
import random

import torch
from tqdm import tqdm
from torch.utils.data import Sampler
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import MD17, QM9


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
        data = torch.load(data_path,weights_only=False)
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


def split_dataset(dataset, seed, train_ratio=0.1, valid_ratio=0.1, test_ratio=None,
                  train_size=None, valid_size=None, test_size=None,
                  resample_train_ratio=None, resample_val_ratio=None, resample_test_ratio=None, shuffle=False,
                  train_shuffle=False, val_shuffle=False, test_shuffle=False, **kwargs):
    """
    Split a dataset into train, validation, and test subsets.

    Args:
        dataset: The dataset to be split.
        seed: Random seed for reproducibility (used only when shuffle=True).
        train_ratio: Proportion of the dataset to be used for training.
        valid_ratio: Proportion of the dataset to be used for validation.
        test_ratio: Proportion of the dataset to be used for testing (ignored if test_size is provided).
        train_size: Absolute size of the training set (overrides train_ratio if provided).
        valid_size: Absolute size of the validation set (overrides valid_ratio if provided).
        test_size: Absolute size of the test set (overrides test_ratio if provided).
        shuffle: Whether to shuffle the dataset before splitting.
        **kwargs: Additional arguments (unused).

    Returns:
        train_dataset, valid_dataset, test_dataset: The three subsets of the dataset.
    """
    import numpy as np
    from torch.utils.data import Subset

    data_size = len(dataset)

    # Calculate test size if not provided
    if test_size is None:
        if test_ratio is None:
            # Default test_ratio if neither test_ratio nor test_size is provided
            test_ratio = 1.0 - train_ratio - valid_ratio
            assert 0. < test_ratio <= 1., f"test_ratio({test_ratio}) must be in (0, 1]."
        test_size = int(data_size * test_ratio)
    else:
        assert test_size <= data_size, f"test_size({test_size}) must not exceed dataset size({data_size})."

    # Calculate train and validation sizes if not provided
    if train_size is None:
        assert 0. < train_ratio <= 1., f"train_ratio({train_ratio}) must be in (0, 1]."
        train_size = int(data_size * train_ratio)
    if valid_size is None:
        assert 0. < valid_ratio <= 1., f"valid_ratio({valid_ratio}) must be in (0, 1]."
        valid_size = int(data_size * valid_ratio)

    # Ensure train + valid + test does not exceed dataset size
    assert train_size + valid_size + test_size <= data_size, (
        f"train_size + valid_size + test_size ({train_size} + {valid_size} + {test_size}) "
        f"must not exceed dataset size ({data_size})."
    )

    # Generate indices
    ids = np.arange(data_size)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(ids)

    # Split indices
    train_idx = ids[:train_size]
    val_idx = ids[train_size:train_size + valid_size]
    test_idx = ids[train_size + valid_size:train_size + valid_size + test_size]

    # Resample indices
    if resample_train_ratio is not None:
        resample_train_size = int(train_size * resample_train_ratio)
        train_idx = train_idx[np.random.randint(0, train_size, resample_train_size)]

    if resample_val_ratio is not None:
        resample_val_size = int(valid_size * resample_val_ratio)
        val_idx = val_idx[np.random.randint(0, valid_size, resample_val_size)]

    if resample_test_ratio is not None:
        resample_test_size = int(test_size * resample_test_ratio)
        test_idx = test_idx[np.random.randint(0, test_size, resample_test_size)]

    if train_shuffle:
        train_size = len(train_idx)
        train_idx = np.arange(data_size)
        np.random.seed(seed)
        np.random.shuffle(train_idx)
        train_idx = train_idx[:train_size]

    if val_shuffle:
        val_size = len(val_idx)
        val_idx = np.arange(data_size)
        np.random.seed(seed)
        np.random.shuffle(val_idx)
        val_idx = val_idx[:val_size]

    if test_shuffle:
        test_size = len(test_idx)
        test_idx = np.arange(data_size)
        np.random.seed(seed)
        np.random.shuffle(test_idx)
        test_idx = test_idx[:test_size]

    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    return train_dataset, valid_dataset, test_dataset


def save_dataset(dataset, save_path: str):
    z = []
    pos = []
    force = []
    y = []
    print(f'len(dataset): {len(dataset)}')
    from torch_geometric.loader import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for data in tqdm(dataloader):
        if not hasattr(data, 'y') or data.y is None:
            y.append(data.energy)
        else:
            y.append(data.y)
        z.append(data.z)
        pos.append(data.pos)
        force.append(data.force)

    z = torch.stack(z)
    pos = torch.stack(pos)
    force = torch.stack(force)
    y = torch.stack(y)

    torch.save({
        'z': z.detach().cpu(),
        'pos': pos.detach().cpu(),
        'y': y.detach().cpu(),
        'force': force.detach().cpu(),
    }, save_path)

    print(f"Dataset saved to {save_path}")


def get_dataset(dataset_name: str, root: str, name: str = None, **kwargs):
    """
    Factory function to get molecular datasets from PyG (PyTorch Geometric)
    
    Args:
        dataset_name (str): Type of dataset, can be 'MD17' or 'QM9'
        root (str): Root directory to store the dataset
        name (str, optional): Name of molecule for MD17, available options:
                            ['aspirin', 'benzene', 'ethanol', 'malonaldehyde',
                             'naphthalene', 'salicylic acid', 'toluene', 'uracil']
                            Not needed for QM9 dataset
    
    Returns:
        dataset: PyG dataset object (MD17 or QM9)
    
    Examples:
        >>> md17_data = get_molecular_dataset('MD17', './data', 'aspirin')
        >>> qm9_data = get_molecular_dataset('QM9', './data')
    """

    # Create directory if not exists
    if not osp.exists(root):
        os.makedirs(root)

    # Convert dataset name to uppercase for comparison
    dataset_name = dataset_name.upper()

    # Check if dataset type is valid
    valid_datasets = ['MD17', 'QM9']
    if dataset_name not in valid_datasets:
        raise ValueError(f"Dataset must be one of {valid_datasets}")

    try:
        if dataset_name == 'MD17':
            # List of valid molecule names for MD17
            valid_names = ['aspirin', 'benzene', 'ethanol', 'malonaldehyde',
                           'naphthalene', 'salicylic acid', 'toluene', 'uracil']

            # Check if molecule name is provided and valid
            if name is None:
                raise ValueError("Molecule name must be provided for MD17 dataset")
            if name.lower() not in valid_names:
                raise ValueError(f"Molecule name must be one of {valid_names}")

            # Load the MD17 dataset
            return MD17(root=root, name=name)

        else:  # dataset == 'QM9'
            # Load the QM9 dataset
            return QM9(root=root)

    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")


def get_and_split_dataset(**kwargs):
    train_dataset_path = kwargs.get('train_dataset_path', None)
    val_dataset_path = kwargs.get('val_dataset_path', None)
    test_dataset_path = kwargs.get('test_dataset_path', None)
    finetune_dataset_path = kwargs.get('finetune_dataset_path', None)
    pretrain_dataset_path = kwargs.get('pretrain_dataset_path', None)
    train_dataset, valid_dataset, test_dataset, finetune_dataset, pretrain_dataset = None, None, None, None, None

    if train_dataset_path is not None and osp.exists(train_dataset_path):
        train_dataset = BasicMDDataset.from_pt(train_dataset_path)
    if val_dataset_path is not None and osp.exists(val_dataset_path):
        valid_dataset = BasicMDDataset.from_pt(val_dataset_path)
    if test_dataset_path is not None and osp.exists(test_dataset_path):
        test_dataset = BasicMDDataset.from_pt(test_dataset_path)
    if finetune_dataset_path is not None and osp.exists(finetune_dataset_path):
        finetune_dataset = BasicMDDataset.from_pt(finetune_dataset_path)
    if pretrain_dataset_path is not None and osp.exists(pretrain_dataset_path):
        pretrain_dataset = BasicMDDataset.from_pt(pretrain_dataset_path)
    if train_dataset is None and  finetune_dataset is None and pretrain_dataset is None:
        raise ValueError(f"Dataset must be provided for Training or Finetune or Pretrain dataset")
    return train_dataset, valid_dataset, test_dataset, finetune_dataset, pretrain_dataset
    #return split_dataset(dataset=get_dataset(**kwargs), **kwargs)


class IndexControlSampler(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source
        self.current_indices = None

    def __iter__(self):
        if self.current_indices is None:
            return list(range(len(self.data_source)))

        return iter(self.current_indices)

    def __len__(self):
        return len(self.current_indices)

    def get_indices(self):
        return self.current_indices

    def set_indices(self, indices):
        self.current_indices = indices

    def reset_indices(self):
        self.current_indices = list(range(len(self.data_source)))


class DistillDataset(Dataset):
    def __init__(self, pos, z, y, force, use_coarse_label: bool = False, **kwargs):
        self.use_coarse_label = use_coarse_label
        self.pos = pos
        self.z = z
        self.y = y
        self.force = force
        if use_coarse_label:
            self._coarse_force()

    def _coarse_force(self):
        import ase
        from ase.calculators.emt import EMT
        num_atoms_per_mole = self.pos.shape[1]
        coarse_force = []
        # 通过 ASE 的 经验势函数（Effective Medium Theory）模型 （非常快速的近似量子力学方法） 用于估计能量和力
        for i in range(len(self.y)):
            atoms = ase.Atoms(numbers=self.z[i].reshape(-1), positions=self.pos[i].reshape(num_atoms_per_mole, 3))
            calc = EMT()
            atoms.calc = calc
            coarse_force.append(torch.tensor(atoms.get_forces()))
        gt_force = self.force.clone()
        coarse_force = torch.stack(coarse_force, dim=0).to(device=self.pos.device, dtype=torch.float32)

        # Calculate MAE difference
        mae = torch.mean(torch.abs(coarse_force - gt_force)).item()

        # Calculate mean and variance for both forces
        gt_mean = torch.mean(gt_force).item()
        gt_var = torch.var(gt_force).item()

        coarse_mean = torch.mean(coarse_force).item()
        coarse_var = torch.var(coarse_force).item()

        print(f"MAE between coarse force and ground truth force: {mae:.6f}")
        print(f"Ground truth force - Mean: {gt_mean:.6f}, Variance: {gt_var:.6f}")
        print(f"Coarse force - Mean: {coarse_mean:.6f}, Variance: {coarse_var:.6f}")
        print(f"Mean difference: {abs(coarse_mean - gt_mean):.6f}")
        print(f"Variance difference: {abs(coarse_var - gt_var):.6f}")

        self.force = coarse_force

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
        data = torch.load(data_path,weights_only=False)
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
            y.append(data.y)
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

    def sample_subset(self, num_samples: int):
        """从当前数据集中随机采样 num_samples 个样本，返回一个新的 DistillDataset 对象"""
        assert num_samples <= len(self), "样本数超出数据集总量"

        indices = random.sample(range(len(self)), num_samples)

        # 取出子集
        pos = self.pos[indices]
        z = self.z[indices]
        y = self.y[indices]
        force = self.force[indices]

        return DistillDataset(
            pos=pos,
            z=z,
            y=y,
            force=force,
            use_coarse_label=False  # 已经构建过，不需要再生成 coarse_force
        )
