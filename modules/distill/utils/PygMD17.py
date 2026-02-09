import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
from modules.distill.utils.nets import get_network
# 从 PyTorch Geometric 导入必要的类和函数
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data, DataLoader

# 从 PyTorch Geometric 导入数据变换工具
from torch_geometric.transforms import RadiusGraph, Distance


class MD17(InMemoryDataset):
    r"""
    一个用于 :obj:`MD17` 数据集的 `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ 数据接口，
    该数据集来自论文 `"Machine learning of accurate energy-conserving molecular force fields" <https://advances.sciencemag.org/content/3/5/e1603015.short>`_。
    MD17 包含了八种小有机分子的分子动力学模拟数据。
    
    参数:
        root (字符串): 数据集文件夹将位于 root/name 路径下。
        name (字符串): 数据集的名称。可用的数据集名称如下: :obj:`aspirin`（阿司匹林）, :obj:`benzene_old`（旧版苯）, :obj:`ethanol`（乙醇）, :obj:`malonaldehyde`（丙二醛）,
            :obj:`naphthalene`（萘）, :obj:`salicylic`（水杨酸）, :obj:`toluene`（甲苯）, :obj:`uracil`（尿嘧啶）。(默认: :obj:`benzene_old`)
        transform (可调用对象, 可选): 一个函数/变换，它接收一个 :obj:`torch_geometric.data.Data` 对象并返回一个变换后的版本。
            每次访问数据对象时都会进行变换。(默认: :obj:`None`)
        pre_transform (可调用对象, 可选): 一个函数/变换，它接收一个 :obj:`torch_geometric.data.Data` 对象并返回一个变换后的版本。
            数据对象在保存到磁盘之前会被变换。(默认: :obj:`None`)
        pre_filter (可调用对象, 可选): 一个函数，它接收一个 :obj:`torch_geometric.data.Data` 对象并返回一个布尔值，
            指示该数据对象是否应包含在最终的数据集中。(默认: :obj:`None`)

    示例:
    --------

    >>> dataset = MD17(name='aspirin')
    >>> split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=42)
    >>> train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    >>> train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    >>> data = next(iter(train_loader))
    >>> data
        Batch(batch=[672], force=[672, 3], pos=[672, 3], ptr=[33], y=[32], z=[672])

    输出数据的属性含义如下:
        * :obj:`z`: 原子类型。
        * :obj:`pos`: 原子的三维坐标位置。
        * :obj:`y`: 图（分子）的性质（能量）。
        * :obj:`force`: 原子的三维受力。
        * :obj:`batch`: 分配向量，将每个节点映射到其对应的图标识符，有助于重构单个图。
    """

    def __init__(self, root='dataset/', name='benzene_old', transform=None, pre_transform=None, pre_filter=None):
        # 初始化数据集名称和文件夹路径
        self.name = name
        self.folder = osp.join(root, self.name)
        # 构建数据集的下载URL
        self.url = 'http://quantum-machine.org/gdml/data/npz/md17_' + self.name + '.npz'

        # 调用父类构造函数
        super(MD17, self).__init__(self.folder, transform, pre_transform, pre_filter)

        # 加载处理后的数据
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # 返回原始文件的名称
        return 'md17_' + self.name + '.npz'

    @property
    def processed_file_names(self):
        # 返回处理后文件的名称
        return self.name + '_pyg.pt'

    def download(self):
        # 下载数据文件到原始数据目录
        download_url(self.url, self.raw_dir)

    def process(self):
        # 加载原始的 npz 数据文件
        data = np.load(osp.join(self.raw_dir, self.raw_file_names))

        # 提取并处理数据
        E = data['E'] - data['E'].mean()  # 能量减去平均值，进行中心化
        F = data['F']  # 原子受力
        R = data['R']  # 原子位置
        z = data['z']  # 原子类型

        data_list = []
        # 遍历每一个分子构型
        for i in tqdm(range(len(E))):
            # 将numpy数组转换为PyTorch张量
            R_i = torch.tensor(R[i], dtype=torch.float32)  # 位置
            z_i = torch.tensor(z, dtype=torch.int64)      # 原子类型
            E_i = torch.tensor(E[i], dtype=torch.float32) # 能量
            F_i = torch.tensor(F[i], dtype=torch.float32) # 受力

            # 创建一个全零的张量，与原子类型同形状
            b = torch.zeros_like(z_i)
            # 将原子类型和全零张量堆叠，形成特征矩阵x
            x = torch.stack((z_i, b), 1)

            # 创建Data对象，存储图数据
            data = Data(id=i, pos=R_i, x=x, z=z_i, y=E_i, force=F_i)


            # 将处理好的Data对象加入列表
            data_list.append(data)

        # 应用预过滤器（如果提供了）
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        # 应用预变换器（如果提供了）
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # 将数据列表整理成一个大的Data对象和切片字典
        data, slices = self.collate(data_list)

        print('正在保存...')
        # 保存处理后的数据到磁盘
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        """
        将数据集按指定大小分割为训练集、验证集和测试集。
        
        参数:
            data_size: 数据集总大小
            train_size: 训练集大小
            valid_size: 验证集大小
            seed: 随机种子，确保结果可复现
            
        返回:
            一个字典，包含 'train', 'valid', 'test' 的索引张量
        """
        # 打乱索引
        ids = shuffle(range(data_size), random_state=seed)
        # 划分索引
        train_idx = torch.tensor(ids[:train_size])
        val_idx = torch.tensor(ids[train_size:train_size + valid_size])
        test_idx = torch.tensor(ids[train_size + valid_size:])
        # 返回分割字典
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict


# 主程序入口
if __name__ == '__main__':
    # 创建阿司匹林数据集实例
    dataset = MD17(root='/data2_hdd/xh/MDdata',name='aspirin')
    print(dataset)
    print(dataset.data.z.shape)      # 原子类型形状
    print(dataset.data.pos.shape)    # 原子位置形状
    print(dataset.data.y.shape)      # 能量形状
    print(dataset.data.force.shape)  # 受力形状
    
    # 获取数据集分割索引
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=42)
    # print(split_idx)
    # print(dataset[split_idx['train']])
    
    # 创建训练、验证、测试数据集
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    test_dataset = test_dataset[:1000]
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    painnet = get_network('painnet')

    from torch.autograd import grad
    from dig.threedgraph.evaluation import ThreeDEvaluator

    evaluation = ThreeDEvaluator()
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
    preds = preds.squeeze(-1)
    input_dict = {"y_true": targets, "y_pred": preds}
    input_dict_force = {"y_true": targets_force, "y_pred": preds_force}
    energy_mae = evaluation.eval(input_dict)['mae']
    force_mae = evaluation.eval(input_dict_force)['mae']
    print(energy_mae, force_mae)