import argparse

import yaml

from modules.distill.utils.nets import get_network
import random
import numpy as np
import torch
import torch_geometric as pyg
from modules.distill.utils.datasets import get_dataset, split_dataset, save_dataset, DistillDataset, \
    get_and_split_dataset
from modules.distill.utils.trainers import Trainer
import shutil
import copy
from peft import PeftModel
import os
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate distill data with YAML configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-d',
        '--distill_dir',
        type=str,
        default='/data1_ssd/xh/dis/distill2/test2/distill1',
        help='Distill directory 蒸馏模型路径'
    )

    parser.add_argument(
        '-c',
        '--ckpt_id',
        type=str,
        default='499',
        help='Checkpoint id for distill data adapter 保存点'
        # required=True
    )

    parser.add_argument(
        '--config',
        type=str,
        help='specific config file 可以不用'
    )

    parser.add_argument(
        '-f',
        '--finetune_epochs',
        type=int,
        help='Epoch for finetuning distill data 微调轮次',
        default=200
    )

    parser.add_argument(
        '--enable_adapter',
        action="store_true",
        help='use distill adapter 微调模型是否加载 adapter ',
    )

    parser.add_argument(
        '-s',
        '--save_dir',
        type=str,
        help='save directory of model checkpoints',
        default=None
    )

    parser.add_argument(
        '--pretrain_mole_size',
        type=int,
        help='pretrain molecule size 截取部分训练数据（暂定）',
        default=500
    )

    parser.add_argument(
        '--no_scheduler',
        type=bool,
        help='no scheduler',
        default=False
    )

    parser.add_argument(
        '--extras',
        required=False,
        type=str,
        nargs='+',
    )

    parser.add_argument(
        '--merge_adapter',
        action="store_true",
        help='merge adapter weight  是否融合 adapter 参数'
    )

    parser.add_argument(
        '--finetune_dataset_path',
        type=str,
        help='path to dataset for finetuning',
        default=None
    )

    parser.add_argument(
        '--project_name',
        type=str,
        help='name of project to wandb',
        default='MoleculeDynamics-Eval-Distilled'
    )

    return parser.parse_args()


def load_config(config_path, args):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if args.finetune_epochs:
            config['train_cfg']['finetune_epochs'] = args.finetune_epochs

        if args.merge_adapter:
            config['train_cfg']['merge_adapter'] = args.merge_adapter

        if args.finetune_dataset_path:
            config['data_cfg']['finetune_dataset_path'] = args.finetune_dataset_path
        else:
            config['data_cfg']['finetune_dataset_path'] = config['data_cfg']['train_dataset_path']  # 微调数据集和生成专家轨迹的数据集一致
        if args.project_name:
            config['train_cfg']['project_name'] = args.project_name
        if args.save_dir is None:
            config['train_cfg']['save_dir'] = os.path.join(args.distill_dir, f'{args.ckpt_id}_eval') if args.save_dir is None else args.save_dir
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading config file: {str(e)}")


def main():
    args = parse_args()
    #save_dir = os.path.join(args.distill_dir, f'{args.ckpt_id}_eval') if args.save_dir is None else args.save_dir
    finetune_epochs = args.finetune_epochs
    print(f'finetune_epoch: {finetune_epochs}')
    if args.config is None or not os.path.exists(args.config):
        config_path = os.path.join(args.distill_dir, 'config.yaml')
    else:
        config_path = args.config
    config = load_config(config_path, args)
    # only_reg_force = True if args.only_reg_force else False
    # shuffle = False if args.cont else True
    distill_cfg, data_cfg, network_cfg, train_cfg = config['distill_cfg'], config['data_cfg'], config['network_cfg'], \
        config['train_cfg']
    save_dir = train_cfg['save_dir']
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Parse configurations

    enable_adapter = distill_cfg['enable_adapter']
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # if data_cfg.get('seed') is not None:
    set_seed(data_cfg['seed'])
    pretrain_data_cfg = copy.deepcopy(data_cfg)
    # dataset = get_dataset(**data_cfg)
    train_dataset, valid_dataset, test_dataset, finetune_dataset = get_and_split_dataset(**data_cfg)
    # train_dataset, valid_dataset, test_dataset = split_dataset(dataset=dataset, **data_cfg)
    coarse_train_dataset = DistillDataset.from_dataset(train_dataset, distill_cfg['use_coarse_label'])
    if args.pretrain_mole_size:
        pretrain_data_cfg['train_size'] = args.pretrain_mole_size
        # coarse_train_dataset = DistillDataset.from_pt(data_path=data_cfg['train_dataset_path'], use_coarse_label=distill_cfg['use_coarse_label'])
        # coarse_train_dataset, _, _ = split_dataset(dataset=dataset, **pretrain_data_cfg)
        # coarse_train_dataset = DistillDataset.from_dataset(coarse_train_dataset, distill_cfg['use_coarse_label'])
        coarse_train_dataset = DistillDataset.sample_subset(coarse_train_dataset, pretrain_data_cfg['train_size'])
        finetune_dataset = DistillDataset.sample_subset(finetune_dataset, pretrain_data_cfg['train_size'])
        print("数据压缩率：", pretrain_data_cfg['train_size']/len(train_dataset))

    if train_dataset is not None:
        save_dataset(train_dataset, os.path.join(save_dir, 'train_dataset.pt'))
    if valid_dataset is not None:
        save_dataset(valid_dataset, os.path.join(save_dir, 'valid_dataset.pt'))
    if test_dataset is not None:
        save_dataset(test_dataset, os.path.join(save_dir, 'test_dataset.pt'))
    if coarse_train_dataset is not None:
        save_dataset(coarse_train_dataset, os.path.join(save_dir, 'coarse_train_dataset.pt'))
    if finetune_dataset is not None:
        save_dataset(finetune_dataset, os.path.join(save_dir, 'finetune_dataset.pt'))

    model = get_network(**network_cfg)
    model.requires_grad_(True)
    if enable_adapter:
        adapter_dir = os.path.join(distill_cfg['save_dir'], f'{args.ckpt_id}_adapter')   # distill_cfg['save_dir'] = '/data1_ssd/xh/dis/distill2/test4/distill4'
        assert os.path.exists(adapter_dir), f"{adapter_dir} is not exist."
        model = PeftModel.from_pretrained(model, adapter_dir)
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

        for name, param in model.named_parameters():
            print(f'{name} requires grad: {param.requires_grad}')
    cfg = train_cfg.copy()
    cfg.update(config['network_cfg'])
    cfg.update(config['data_cfg'])

    trainer = Trainer()
    trainer.train(
        device=device,
        train_dataset=coarse_train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        finetune_dataset=finetune_dataset,
        model=model,
        **cfg
    )


if __name__ == "__main__":
    main()