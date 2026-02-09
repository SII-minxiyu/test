# train.py
import torch
import torch.nn as nn
from PygMD17 import MD17
from PaiNN import PainnModel as PaiNN
from runn import run
from dig.threedgraph.evaluation import ThreeDEvaluator
import numpy as np

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set the root directory, data name and checkpoint directory
root = '/data2_hdd/xh/dis/distill2/lmxcode'
data_name = 'aspirin'

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device("cpu")

# Load the MD17 dataset
dataset = MD17(root='/data2_hdd/xh/MDdata', name=data_name)

# Split the dataset into train, validation, and test sets
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=200, seed=42)
train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
test_dataset = test_dataset[:1000]

model = PaiNN(num_interactions=3,     
              hidden_state_size=128,   
              cutoff=5.0,
              pdb=False) 

torch.manual_seed(42)

loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

# Create an instance of the run class
run3d = run()


run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
          epochs=1500, batch_size=10, vt_batch_size=32, lr=0.001, 
          lr_decay_factor=0.8, lr_decay_step_size=100, weight_decay=0, 
          energy_and_force=True, p=100, save_dir='best',log_dir='best')
