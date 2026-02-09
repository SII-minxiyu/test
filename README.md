# Adapter for Noisy Label Learning in Molecular Dynamics

## Distillation
This section details the parameters for `Distiller` class and related utilities.

---

### Usage

#### Non-blocking distillation for noisy label in Molecular Dynamics
Example code
```python
import time
import threading
from modules.distill import Distiller, DataConf, TrainConf, DistillConf, NetworkConf

if __name__ == '__main__':
    data_cfg = DataConf()
    data_cfg.train_dataset_path = 'modules/distill/test/benzene_train.pt'
    train_cfg = TrainConf()
    distill_cfg = DistillConf()
    network_cfg = NetworkConf()
    distiller = Distiller(data_cfg, train_cfg, distill_cfg, network_cfg, save_dir='.log/distill/test')

    distill_done = threading.Event()

    def on_pipeline_done():
        print('Distiller pipeline finished!')
        distill_done.set()

    distiller(
        non_blocking=True,
        num_expert_trajectory=1,
        gpu_ids_list=[0],
        done_callback=on_pipeline_done
    )
    print('Executing non-blocking distillation...')

    while not distill_done.is_set():
        print("Main thread: Waiting for distillation to finish...")
        time.sleep(5)

    print("Main thread: Detected distillation completion!")

```
---

### `Distiller.__call__` Parameters

| Parameter              | Type                    | Default   | Description                                                                                                                                   |
|------------------------|-------------------------|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `non_blocking`         | `bool`                  | `True`    | If `True`, runs the entire pipeline asynchronously in a background thread. If `False`, runs synchronously and blocks until completion.        |
| `num_expert_trajectory`| `int`                   | `1`       | Number of expert trajectories to generate in parallel. Each trajectory will use a different random seed and save output in a separate folder. |
| `gpu_ids_list`         | `list of int` or `None` | `None`    | List of GPU device IDs to assign for expert trajectory generation. If provided, experts are assigned to GPUs in round-robin. If `None`, runs on CPU. |
| `done_callback`        | `callable` or `None`    | `None`    | A function to be called when the full pipeline (expert trajectory generation + distillation) finishes (only used when `non_blocking=True`).   |

---

#### Example explained

- `non_blocking=True`: The pipeline runs in the background, allowing the main thread to perform other tasks.
- `num_expert_trajectory=1`: Generates 1 expert trajectory.
- `gpu_ids_list=[0]`: Uses GPU 0 for the expert trajectory generation.
- `done_callback=on_pipeline_done`: When the full pipeline is finished, the `on_pipeline_done` function is called, which here sets a `threading.Event` so the main thread can detect completion.
---

### Configuration Classes

This project manages settings using Python `dataclass` configuration objects. The four primary configuration classes are:

- `DataConf`: Dataset and loading options  
- `NetworkConf`: Model architecture parameters  
- `TrainConf`: Expert trajectory preparation parameters
- `DistillConf`: Adapter distillation settings  

Each class can be instantiated and customized as needed for your experiments.

---
#### 1. `DataConf`: Data Configuration

| Parameter           | Type    | Default      | Description                                                      |
|---------------------|---------|--------------|------------------------------------------------------------------|
| `train_dataset_path`| str     | None         | Path to the training dataset (for custom datasets). **Mutually exclusive with `dataset_name`: if this is set, `dataset_name` is ignored.** |
| `val_dataset_path`  | str     | None         | Path to the validation dataset (for custom datasets, **not necessary**).            |
| `test_dataset_path` | str     | None         | Path to the test dataset (for custom datasets, **not necessary**).                  |
| `dataset_name`      | str     | `"MD17"`     | Name of the dataset (used for built-in PyG datasets). **Ignored if `train_dataset_path` is specified.** |
| `name`              | str     | `"benzene"`  | Molecule or task name.                                           |
| `root`              | str     | `"data/MD17"`| Root directory for the dataset.                                  |
| `seed`              | int     | 47           | Random seed for reproducibility.                                 |
| `shuffle`           | bool    | True         | Whether to shuffle the data during loading.                      |
| `test_size`         | int     | 1000         | Number of samples in the test set.                               |
| `train_size`        | int     | 1000         | Number of samples in the training set.                           |
| `valid_size`        | int     | 1000         | Number of samples in the validation set.                         |

**Note:**  
- `train_dataset_path` and `dataset_name` are mutually exclusive.  
- If `train_dataset_path` is specified, the custom dataset will be used and `dataset_name` will be ignored.  
- If `train_dataset_path` is not specified, the built-in PyG dataset indicated by `dataset_name` will be used.

#### About the format of `train_dataset_path` File:
---
The file specified by `train_dataset_path` should be a PyTorch `.pt` file, saved using `torch.save`, as well as `val_dataset_path` and `test_dataset_path`. 

It must be a dictionary containing the following tensors:

| Key    | Shape (example)         | Description                                                                          |
|--------|------------------------|--------------------------------------------------------------------------------------|
| `z`    | `[num_samples, num_atoms]`        | Atomic numbers for each atom in every molecule/sample.                              |
| `pos`  | `[num_samples, num_atoms, 3]`     | 3D coordinates (positions) of each atom in every molecule/sample.                   |
| `y`    | `[num_samples, 1]` or `[num_samples]` | Target property for each sample (e.g. energy).                                  |
| `force`| `[num_samples, num_atoms, 3]`     | Force vectors for each atom in every molecule/sample.                               |

**Example:**

```python
{
    'z':      torch.LongTensor of shape [num_samples, num_atoms],
    'pos':    torch.FloatTensor of shape [num_samples, num_atoms, 3],
    'y':      torch.FloatTensor of shape [num_samples, 1] or [num_samples],
    'force':  torch.FloatTensor of shape [num_samples, num_atoms, 3],
}
```
Please refer to the `save_dataset` function in `modules/distill/utils/datasets.py` for more details.

---

#### 2. `NetworkConf`: Model Architecture Configuration

| Parameter           | Type    | Default      | Description                                                             |
|---------------------|---------|--------------|-------------------------------------------------------------------------|
| `name`              | str     | `"schnet"`   | Name of the model architecture (e.g. "schnet").                         |
| `cutoff`            | float   | 5.0          | Cutoff radius for neighbor search (used in schnet).                     |
| `hidden_channels`   | int     | 256          | Number of hidden channels.                                              |
| `num_filters`       | int     | 128          | Number of filters in the network.                                       |
| `num_gaussians`     | int     | 50           | Number of Gaussian basis functions.                                     |
| `num_interactions`  | int     | 3            | Number of interaction blocks/layers.                                    |

*Note: To support other architectures, add parameters as needed.*

---

#### 3. `TrainConf`: Training Expert Trajectory Configuration

| Parameter                | Type    | Default                              | Description                                            |
|--------------------------|---------|--------------------------------------|--------------------------------------------------------|
| `batch_size`             | int     | 1                                    | Training batch size.                                   |
| `energy_and_force`       | bool    | True                                 | Whether to predict both energy and forces.             |
| `epochs`                 | int     | 200                                  | Number of training epochs.                             |
| `save_epochs`            | int     | 100                                  | Save the model every `save_epochs` epochs.             |
| `save_iters`             | int     | 250                                  | Save the model every `save_iters` iterations.          |
| `loss_func`              | str     | `"l2"`                               | Loss function type (`"l1"` or `"l2"`).                 |
| `lr`                     | float   | 0.0003                               | Initial learning rate.                                 |
| `lr_decay_factor`        | float   | 0.95                                 | Learning rate decay factor.                            |
| `lr_decay_step_size`     | int     | 30000                                | Learning rate decay step size (in iterations).         |
| `optimizer_name`         | str     | `"Adam"`                             | Name of the optimizer (e.g. "Adam", "SGD").            |
| `p`                      | float   | 100                                  | Reciprocal weight for energy loss function.                          |
| `project_name`           | str     | `"MoleculeDynamics-Expert-Trajectory"`| Project name for logging and saving.                   |
| `save_dir`               | str     | None                                 | Directory to save logs and checkpoints.                |
| `scheduler_name`         | str     | `"ExpDecayLR"`                       | Learning rate scheduler type.                          |
| `test_step`              | int     | 1                                    | Evaluation frequency on the test set (in epochs).      |
| `val_step`               | int     | 1                                    | Evaluation frequency on the validation set (in epochs).|
| `vt_batch_size`          | int     | 128                                  | Batch size for validation/testing.                     |
| `weight_decay`           | float   | 0                                    | Weight decay (L2 regularization).                      |

---

#### 4. `DistillConf`: Adapter Distillation Configuration

| Parameter                       | Type   | Default                       | Description                                             |
|----------------------------------|--------|-------------------------------|---------------------------------------------------------|
| `project_name`                   | str    | `"MoleculeDynamics-Adapter-Distill"` | Project name for the distillation run.           |
| `algorithm`                      | str    | `"adapter_mtt"`               | Name of the distillation algorithm.                     |
| `use_coarse_label`               | bool   | True                          | Whether to use coarse labels in distillation, only support `EMT` coarse label temporally.           |
| `num_iteration`                  | int    | 500                           | Number of distillation iterations.                      |
| `max_start_iter`                 | int    | 200000                        | Maximum starting iteration an expert trajectory.            |
| `min_start_iter`                 | int    | 0                             | Minimum starting iteration of an expert trajectory.                             |
| `num_expert`                     | int    | 1                             | Length of a fragment expert trajectory.                                |
| `save_step`                      | int    | 100                           | Save the model every `save_step` iterations.            |
| `p`                              | int    | 100                           | Energy loss reciprocal weight used in distillation.                       |
| `distill_batch_size`             | int    | 1                             | Batch size for distillation.                            |
| `enable_adapter`                 | bool   | True                          | Whether to enable the adapter module.                   |
| `distill_energy`                 | bool   | False                         | Whether to use energy loss in distillation.                      |
| `distill_force`                  | bool   | True                          | Whether to use energy loss in distillation.                       |
| `distill_optimizer_type`         | str    | "adam"                        | Optimizer for the distillation process.                 |
| `distill_scheduler_type`         | str    | "step"                        | Scheduler type for distillation.                        |
| `distill_scheduler_decay_step`   | int    | 400                           | Scheduler step size for decay.                          |
| `distill_scheduler_decay_rate`   | float  | 0.5                           | Scheduler decay rate.                                   |
| `distill_lr_adapter`             | float  | 1.0e-4                        | Learning rate for the adapter module.                   |
| `adapter_dict`                   | dict   | see below                     | Dictionary of adapter configuration options.            |

##### Example `adapter_dict`:

```python
adapter_dict = {
    "peft_type": "LORA", 
    "r": 8, 
    "lora_alpha": 8, 
    "target_modules": [
        "embedding", 
        "interactions.0.conv.lin1", 
        "interactions.0.conv.lin2", 
    ], 
    "lora_dropout": 0.01, 
    "bias": None
}
```

## Use noisy label adapter

The following example demonstrates how to apply the noisy label adapter workflow in molecular dynamics tasks:

1. **Run Distillation:**  
   Use the `Distiller` class to asynchronously perform expert trajectory generation and distillation. You can specify GPU usage and the number of expert trajectories.

2. **Wrap Base Model with Adapter:**  
   After distillation, load the base model and wrap it with an adapter (e.g., LoRA) using the `PeftModel` from the [PEFT library](https://github.com/huggingface/peft). Parameters related to the adapter should be frozen.

3. **Pre-train with Noisy Labels:**  
   Pre-train the adapter-augmented model using your noisy label data.

4. **Finetune with Clean Labels:**  
   If you want to fine-tune on clean labels, you can remove the adapter and unfreeze the base model parameters.

Below is a complete example:
```python

import time
import threading
from modules.distill import Distiller, DataConf, TrainConf, DistillConf, NetworkConf

if __name__ == '__main__':
    data_cfg = DataConf()
    data_cfg.train_dataset_path = 'modules/distill/test/benzene_train.pt'
    train_cfg = TrainConf()
    distill_cfg = DistillConf()
    network_cfg = NetworkConf()
    """
        Default network config:
            'name': 'schnet', 
            'hidden_channels': 256,
            'num_filters': 128,
            'cutoff': 5.0,
            'num_interactions': 3, 
            'num_gaussians': 50
    """
    distiller = Distiller(data_cfg, train_cfg, distill_cfg, network_cfg, save_dir='.log/distill/test')

    distill_done = threading.Event()

    def on_pipeline_done():
        print('Distiller pipeline finished!')
        distill_done.set()

    distiller(
        non_blocking=True,
        num_expert_trajectory=1,
        gpu_ids_list=[0],
        done_callback=on_pipeline_done
    )
    print('Executing non-blocking distillation...')

    #
    # Do other things
    # ...

    # Check distillation completion
    while not distill_done.is_set():
        print("Main thread: Waiting for distillation to finish...")
        time.sleep(5)

    print("Main thread: Detected distillation completion!")
    
    # Wrap base model with adapter, and pre-train model with noisy label
    from modules.distill.nets import get_network
    from peft import PeftModel
    base_model = get_network(**network_cfg.to_dict())
    adapter_dir = os.path.join(distiller.distill_dir, f'{distill_cfg.num_iteration}_adapter')
    assert os.path.exists(adapter_dir), f"{adapter_dir} is not exist."
    model = PeftModel.from_pretrained(model, adapter_dir)
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
    
    for name, param in model.named_parameters():
        print(f'{name} requires grad: {param.requires_grad}')

    #
    # Pre-train
    # ...

    # To fine-tune model with fine label, you can remove adapters.
    if isinstance(model, PeftModel):
        model = model.unload()
        model.requires_grad_(True)

    # 
    # Fine-tune
    # ...
    
```
---