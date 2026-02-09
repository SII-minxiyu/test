import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch_geometric as pyg
from datetime import datetime
from peft import PeftModel
sys.path.append('.')
import torch
from modules.distill.utils.datasets import save_dataset, get_and_split_dataset
from modules.distill.utils.adapters import get_adapter_model
from modules.distill.utils.nets import get_network
#from modules.distill.utils.trainers import Trainer
from modules.distill.utils.train_Kflod import Trainer
from modules.distill.utils.PygMD17 import MD17
os.environ["WANDB_MODE"] = "offline"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def remove_prefix_from_state_dict(state_dict, prefix_to_remove=None, middle_remove="base_layer."):
    new_state_dict = {}
    for k, v in state_dict.items():
        # 先去掉统一的前缀
        if prefix_to_remove and k.startswith(prefix_to_remove):
            new_k = k[len(prefix_to_remove):]
        else:
            new_k = k
        # 再去掉中间的 "base_layer."
        if middle_remove in new_k:
            new_k = new_k.replace(middle_remove, "", 1)  # 只替换一次
        new_state_dict[new_k] = v
    return new_state_dict


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
        '--config',
        type=str,
        help='Path to the YAML configuration file'
    )
    parser.add_argument(
        '--origin_dir',
        type=str,
        help='origin dir'
    )
    parser.add_argument(
        '--run_pattern',
        type=str,
        help='pattern for training runs'
    )

    parser.add_argument(
        '--dataset_name',
        type=str,
        help='Dataset name to override config'
    )

    parser.add_argument(
        '--name',
        type=str,
        help='Dataset subset name to override config'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        help='Save directory'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        help='Train epoch'
    )

    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed'
    )

    parser.add_argument(
        '--wandb_run_id',
        type=str,
        help='wandb run id for resume'
    )

    parser.add_argument(
        '--distill_dir',
        type=str,
        default='/data1_ssd/xh/dis/distill2/test8/distill',
        help='Distill directory 蒸馏模型路径'
    )

    parser.add_argument(
        '--ckpt_id',
        type=str,
        default='999',
        help='Checkpoint id for distill data adapter 保存点'
        # required=True
    )
    return parser.parse_args()


def load_config(config_path, args):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if args.origin_dir:
            config['train_cfg']['origin_dir'] = args.origin_dir
        if args.save_dir:
            config['train_cfg'][
                'save_dir'] = args.save_dir  # #save_dir='/data1_ssd/xh/dis/distill2/test9/expert_trajectory/0
        else:
            config['train_cfg']['save_dir'] = os.path.join(
                '.log',
                'expert_trajectory',
                config['network_cfg']["name"],
                config['data_cfg']["dataset_name"],
                '' if config['data_cfg']["name"] is None else config['data_cfg']["name"],
                datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            )
            os.makedirs(config['train_cfg']['save_dir'])
        cfg_save_path = os.path.join(config['train_cfg']['save_dir'], 'config.yaml')
        with open(cfg_save_path, 'w') as f:
            yaml.safe_dump(config, f)
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading config file: {str(e)}")


def main():
    args = parse_args()
    config = load_config(args.config, args)
    distill_cfg, data_cfg, network_cfg, train_cfg = config['distill_cfg'], config['data_cfg'], config['network_cfg'], \
        config['train_cfg']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(data_cfg['seed'])
    train_dataset, valid_dataset, test_dataset, finetune_dataset, pretrain_dataset = get_and_split_dataset(**data_cfg)

    # dataset = MD17(root='/data2_hdd/xh/MDdata', name='aspirin')
    # split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=200, seed=42)
    # train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    # test_dataset = test_dataset[:1000]
    network = get_network(**network_cfg)  # 标准模型
    network.requires_grad_(True)

    if args.run_pattern == 'pretrain' and train_cfg['merge_adapter'] is True:
        adapter_dir = os.path.join(args.origin_dir, 'distill', f'{args.ckpt_id}_adapter')
        assert os.path.exists(adapter_dir), f"{adapter_dir} is not exist."
        network = PeftModel.from_pretrained(network, adapter_dir)
        for name, param in network.named_parameters():
            if 'lora_' in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
        for name, param in network.named_parameters():
            print(f'{name} requires grad: {param.requires_grad}')
    elif args.run_pattern == 'finetune':
        # 从 预训练的到的best_valid_checkpoint.pt 中加载schnet 参数（剔除 adapter参数）
        ckpt_path = os.path.join(args.origin_dir, 'pretrain/0', 'best_valid_checkpoint.pt')
        #ckpt_path = '/data2_hdd/xh/dis/distill2/fuxian/pretrain/0/checkpoint_epoch_800.pt'
        assert os.path.exists(ckpt_path), f"{ckpt_path} not found."
        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # ckpt 可能是直接 state_dict，也可能是 dict 包含 'model_state_dict'
        state_dict = ckpt.get('model_state_dict', ckpt)

        """
        下面是finetune加上adapter 的代码
        """
        # adapter_dict = distill_cfg['adapter_dict']
        # network = get_adapter_model(adapter_dict, network)
        # network.load_state_dict(state_dict)
        # for name, param in network.named_parameters():
        #     if 'lora_' in name:
        #         param.requires_grad_(False)
        #     else:
        #         param.requires_grad_(True)
        # for name, param in network.named_parameters():
        #     print(f'{name} requires grad: {param.requires_grad}')

        """
        下面是 finetune 不加adapter 的代码
        """
        #找到当前的 base schnet module（兼容 PeftModel / 直接模型）

        if hasattr(network, 'base_model') and hasattr(network.base_model, 'model'):
            base_model = network.base_model.model  # SchNet 实例
        else:
            # 网络不是 PeftModel 的情况（直接是 SchNet）
            base_model = network
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if 'lora_' not in k
        }
        # 去掉多余前缀
        filtered_state_dict = remove_prefix_from_state_dict(filtered_state_dict, "base_model.model.")
        # 严格加载，缺失的 LoRA 参数忽略
        missing_keys, unexpected_keys = base_model.load_state_dict(
            filtered_state_dict, strict=False
        )
        print(f"[Finetune] Loaded SchNet weights from {ckpt_path}")
        if missing_keys:
            print(f"  Missing keys (ignored): {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys (ignored): {unexpected_keys}")
        network = base_model

    cfg = train_cfg.copy()
    cfg.update(config['network_cfg'])
    cfg.update(config['data_cfg'])

    trainer = Trainer()
    trainer.train(
        device=device,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        finetune_dataset=finetune_dataset,
        pretrain_dataset=pretrain_dataset,
        model=network,
        wandb_run_id=args.wandb_run_id,
        **cfg
    )


if __name__ == "__main__":
    main()
