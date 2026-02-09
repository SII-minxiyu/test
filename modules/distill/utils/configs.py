from typing import Dict
from dataclasses import dataclass, field

@dataclass
class DictConf:
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith('_') or callable(value):
                continue
            if isinstance(value, DictConf):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
@dataclass
class DataConf(DictConf):
    # From custom datasets
    #train_dataset_path: str = '/home/xh/project/MDsys/modules/distill/test/asprin/asprin_coarse_train_1000_nomean.pt'
    train_dataset_path: str = '/home/xh/project/MDsys/modules/distill/test/ccsdasprin_train_100_nomean.pt'

    #val_dataset_path: str = '/home/xh/project/MDsys/modules/distill/test/asprin/asprin_coarse_val_500_nomean.pt'

    test_dataset_path:str = '/home/xh/project/MDsys/modules/distill/test/asprin/asprin_ccsd_test_data_nomean.pt'
    #test_dataset_path:str ='/home/xh/project/MDsys/modules/distill/test/asprin/asprin_coarse_test_1000_nomean.pt'

    pretrain_dataset_path: str = '/home/xh/project/MDsys/modules/distill/test/asprin/asprin_coarse_train_1000_nomean.pt'
    finetune_dataset_path: str = '/home/xh/project/MDsys/modules/distill/test/ccsdasprin_train_100_nomean.pt'
    # pretrain_dataset_path: str = '/home/xh/project/MDsys/modules/distill/test/malonaldehyde/malonaldehyde_coarse_train_1000_nomean.pt'
    # finetune_dataset_path: str = '/home/xh/project/MDsys/modules/distill/test/malonaldehyde/malonaldehyde_ccsd_train_data_nomean.pt'

    # Or from PyG datasets
    dataset_name: str = 'MD17'
    name: str = 'Asprin'
    root: str = 'data/MD17'
    seed: int = 47
    shuffle: bool = True
    test_size: int = 1000
    train_size: int = 1000
    valid_size: int = 1000
    

@dataclass
class NetworkConf(DictConf):
    name: str = 'painnet'
    # For painn
    cutoff: float = 5.0 # painn%schnet
    num_interactions: int = 3  # 相互作用块的数量（网络深度 # painn&schnet
    hidden_state_size: int = 128 # painn
    

@dataclass
class TrainConf(DictConf):
    pattern: str = 'expert_trajectory'   # expert_trajectory/pretrain/finetune
    energy_and_force: bool = True
    epochs: int = 400  # 生成专家轨迹epoch
    pretrain_epochs: int = 400
    finetune_epochs: int = 400
    finetune_dataset = None
    pretrain_dataset = None
    batch_size: int = 10
    pretrain_batch_size: int = 10
    finetune_batch_size: int = 10
    merge_adapter: bool = True
    patience: int = 50
    save_epochs: int = 100
    save_iters: int = 250
    loss_func: str = 'l1'
    lr: float = 0.001
    #lr: float = 0.0001
    lr_decay_factor: float = 0.8
    #lr_decay_factor = 0.5
    lr_decay_step_size: int = 50  # batch 50  size 800 -->16  800/50=16*50=800
    #lr_decay_step_size: int = 500  # finetune使用
    optimizer_name: str = 'Adam'
    p: float = 100
    project_name: str = 'MoleculeDynamics-'
    # save_dir: str = '.log/distill/expert_trajectory/MD17/benzene/schnet'   # Suggest setting `save_dir` maunally to store intermediate files
    save_dir: str = None
    origin_dir:str = None
    scheduler_name: str = 'steplr'
    test_step: int = 1
    val_step: int = 1
    vt_batch_size: int = 32
    # vt_batch_size: int = 64
    weight_decay: float = 0


@dataclass
class DistillConf(DictConf):
    project_name: str = "MoleculeDynamics-Adapter-Distill"
    #train_data_path: str = None
    # train_data_path: str = '/home/xh/project/MDsys/modules/distill/test/asprin/aspirin_coarse_train_1000.pt'  # 蒸馏对象数据路径，为None时默认是 生成专家轨迹数据 人工加噪
    # valdata_path: str = '/home/xh/project/MDsys/modules/distill/test/asprin/aspirin_coarse_test_1000.pt'

    train_data_path:str = '/home/xh/project/MDsys/modules/distill/test/asprin/asprin_coarse_test_1000_nomean.pt'
    valdata_path:str = '/home/xh/project/MDsys/modules/distill/test/asprin/asprin_coarse_test_1000_nomean.pt'
    # train_data_path: str = '/home/xh/project/MDsys/modules/distill/test/malonaldehyde/malonaldehyde_coarse_train_1000_nomean.pt'
    # valdata_path: str = '/home/xh/project/MDsys/modules/distill/test/malonaldehyde/malonaldehyde_coarse_train_1000_nomean.pt'

    valbatch: int = 25
    distill_sequence: str = 'rand'
    algorithm: str = "adapter_mtt7"
    use_coarse_label: bool = False
    num_iteration: int = 1000
    max_start_iter: int = 30250 # 400*800
    min_start_iter: int = 0
    num_expert: int = 2
    save_step: int = 100
    save_dir: str = None
    p: int = 100
    distill_batch_size: int = 2
    enable_adapter: bool = True
    distill_energy: bool = True
    distill_force: bool = True
    distill_optimizer_type: str = "adam"
    #distill_scheduler_type: str = "step"
    distill_scheduler_type: str = "stepLR"

    distill_scheduler_decay_step: int = 100     # 这里是 epoch的decay

    distill_scheduler_decay_rate: float = 0.96
    distill_lr_adapter: float = 1.0e-4

    adapter_dict: dict = field(default_factory=lambda: {
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 16,
        "target_modules": [
            "atom_embedding",
            "message_layers.0.scalar_message_mlp.0",
            "message_layers.0.scalar_message_mlp.2",
            "message_layers.0.filter_layer",
            "message_layers.1.scalar_message_mlp.0",
            "message_layers.1.scalar_message_mlp.2",
            "message_layers.1.filter_layer",
            "message_layers.2.scalar_message_mlp.0",
            "message_layers.2.scalar_message_mlp.2",
            "message_layers.2.filter_layer",
            "update_layers.0.update_U",
            "update_layers.0.update_V",
            "update_layers.0.update_mlp.0",
            "update_layers.0.update_mlp.2",
            "update_layers.1.update_U",
            "update_layers.1.update_V",
            "update_layers.1.update_mlp.0",
            "update_layers.1.update_mlp.2",
            "update_layers.2.update_U",
            "update_layers.2.update_V",
            "update_layers.2.update_mlp.0",
            "update_layers.2.update_mlp.2",
            "readout_mlp.0",
            "readout_mlp.2",
        ],
        "lora_dropout": 0.1,
        # "bias": None
    })
    