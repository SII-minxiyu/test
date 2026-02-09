import torch
from torch_geometric.data import DataLoader
from torch.autograd import grad
from tqdm import tqdm
from packaging import version
import torch_geometric
from dig.threedgraph.evaluation import ThreeDEvaluator

class Infer:
    """
    Trainer class for 3DGN methods with wandb logging support
    """
    def __init__(self):
        """Initialize trainer"""
        self.best_valid = float('inf')
        self.best_test = float('inf')
        
    def inference(self, device, test_dataset, 
              model, loss_func=None, evaluation=None, batch_size=32,
              energy_and_force=False):
        """
        Main training loop with wandb integration
        
        Args:
            device (torch.device): Device for computation
            test_dataset: Test data
            model: 3DGN model (SchNet, SphereNet etc.)
            evaluation (function): Evaluation function
            batch_size (int): Batch size 
            energy_and_force (bool): Whether to predict energy and forces
            p (int): Force weight in joint loss
            save_dir (str): Directory to save model checkpoints
            project_name (str): Name for wandb project
            save_step (int): Save checkpoint every N epochs
        """
        model = model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f'#Params: {num_params}')
        
        # Initialize scheduler
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
            
        if evaluation is None:
            evaluation = ThreeDEvaluator()
        
        if loss_func is None:    
            loss_func = torch.nn.L1Loss()
            
        # Testing  
        print('\n\nTesting...', flush=True)
        test_mae = self._evaluate(model, test_loader, energy_and_force,
                                evaluation, device)

        metrics = {
            'test_mae': test_mae,
        }
        
        print(f"\nMetrics: {metrics}")
        return metrics

    def _evaluate(self, model, data_loader, energy_and_force, 
                evaluation, device):
        """Evaluation step"""
        model.eval()
        preds = torch.Tensor([])
        targets = torch.Tensor([])

        if energy_and_force:
            preds_force = torch.Tensor([])
            targets_force = torch.Tensor([])
        
        for batch_data in tqdm(data_loader):
            batch_data = batch_data.to(device)
            if energy_and_force:
                batch_data.pos.requires_grad_(True)
            
            if batch_data.y is None:
                setattr(batch_data, 'y', batch_data.energy)
                
            pyg_version = version.parse(torch_geometric.__version__)

            if pyg_version < version.parse("1.6.0"):
                out = model(batch_data)
            else:
                out = model(batch_data.z, batch_data.pos, batch=batch_data.batch)
            
            if energy_and_force:
                force = -grad(outputs=out, inputs=batch_data.pos,
                            grad_outputs=torch.ones_like(out),
                            create_graph=True, retain_graph=False)[0]
                preds_force = torch.cat([preds_force, force.detach().cpu()], dim=0)
                targets_force = torch.cat([targets_force, batch_data.force.detach().cpu()], dim=0)
                del force
                torch.cuda.empty_cache()
            preds = torch.cat([preds, out.detach().cpu()], dim=0)
            targets = torch.cat([targets, batch_data.y.unsqueeze(1).detach().cpu()], dim=0)

            del batch_data, out
            torch.cuda.empty_cache()

        input_dict = {"y_true": targets, "y_pred": preds}

        if energy_and_force:
            input_dict_force = {"y_true": targets_force, "y_pred": preds_force}
            energy_mae = evaluation.eval(input_dict)['mae']
            force_mae = evaluation.eval(input_dict_force)['mae']
            return energy_mae, force_mae

        return evaluation.eval(input_dict)['mae']