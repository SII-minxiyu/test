import numpy as np
import torch
import random


def process_npz2pt(npz_path, output_pt_name):
    data = np.load(npz_path)
    pos = data[data.files[4]].astype(np.float32)  # 位置数据 -> float32
    y = data[data.files[0]].astype(np.float32)  # 目标值 -> float32
    force = data[data.files[2]].astype(np.float32)  # 力数据 -> float32
    z = data[data.files[5]].astype(np.int64)  # 原子类型 -> int64
    z = np.tile(z, (y.shape[0], 1))
    y_min = y.min()
    y_max = y.max()
    y_normalized = (y - y_min) / (y_max - y_min + 1e-8)  # 加小量防止除零
    y_mean = y.mean()
    y_std = y.std()
    y_mid = y - y_mean

    data_dict = {
        'z': torch.from_numpy(z),
        'pos': torch.from_numpy(pos),
        'y': torch.from_numpy(y-y_mean).float(),
        'y_mid': torch.tensor(y_mid).float(),
        'y_mean': torch.tensor(y_mean).float(),
        'y_std': torch.tensor(y_std).float(),
        'y_normalized': torch.from_numpy(y_normalized),
        'force': torch.from_numpy(force),
        'origin_y': torch.from_numpy(y),
        'y_min': torch.tensor(y_min).float(),
        'y_max': torch.tensor(y_max).float(),
    }
    torch.save(data_dict, output_pt_name)
    print(f"数据已保存为 {output_pt_name}")
    print(f"各数据形状: z={z.shape}, pos={pos.shape}, y={y.shape}, force={force.shape}")


#process_npz2pt('/home/xh/project/MDsys/modules/distill/test/malonaldehyde/malonaldehyde_ccsd_t/malonaldehyde_ccsd_t-train.npz', 'malonaldehyde_ccsd_train_data_nomean.pt')
#process_npz2pt('/home/xh/project/MDsys/modules/distill/test/malonaldehyde/malonaldehyde_ccsd_t/malonaldehyde_ccsd_t-test.npz', 'malonaldehyde_ccsd_test_data_nomean.pt')


def sample_pt_file(input_pt, train_pt, val_pt, test_pt, train_size, val_size, test_size):
    # 加载原始数据
    data = torch.load(input_pt)

    # 假设所有数据维度的第 0 维是样本维（检查 pos/y/force）
    total_samples = data['pos'].shape[0]
    assert train_size+val_size <= total_samples, "采样数量超过数据总量"

    # 随机选择索引
    indices = list(range(total_samples))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:train_size+val_size+test_size]

    def extract_subset(data, idx):
        new_data = {}
        for key, tensor in data.items():
            if isinstance(tensor, torch.Tensor) and tensor.ndim > 0 and tensor.shape[0] == total_samples:
                new_data[key] = tensor[idx]
            else:
                new_data[key] = tensor  # 标量或非样本维度字段直接复制
        return new_data

    train_data = extract_subset(data, train_indices)
    val_data = extract_subset(data, val_indices)
    test_data = extract_subset(data, test_indices)
    # 保存新文件
    torch.save(train_data, train_pt)
    torch.save(val_data, val_pt)
    torch.save(test_data, test_pt)

    print(f"训练集已保存: {train_pt} (样本数: {train_size})")
    print(f"验证集已保存: {val_pt} (样本数: {val_size})")
    print(f"测试集已保存: {test_pt} (样本数: {test_size})")


sample_pt_file('/home/xh/project/MDsys/modules/distill/test/asprin/asprin_coarse_train_1000_nomean.pt', 'coarseasprin_train_100_nomean.pt', '_nomean.pt',"_1000_nomean.pt" ,train_size=100, val_size=100,test_size=100)
