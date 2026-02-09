import torch

def get_dynamic_optimizer(optimizer_type, params, **kwargs):
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "sgd":
        return DyncSGD(params=params, **kwargs)
    elif optimizer_type == "adam":
        return DyncAdam(params=params, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Supported types: 'sgd', 'adam'.")
    

class DyncSGD:
    def __init__(self, params, momentum=0.0, **kwargs):
        if not isinstance(params, torch.Tensor):
            raise TypeError("params must be a single torch.Tensor.")

        self.params = params
        self.momentum = momentum

        self.velocity = torch.zeros_like(params) if momentum > 0 else None

    def step(self, params: torch.Tensor, grad: torch.Tensor, lr: torch.Tensor):
        if grad is None or not isinstance(grad, torch.Tensor):
            raise ValueError(f"grad must be a torch.Tensor. But grad is {type(grad)}.")
        if lr is None or not (isinstance(lr, float) or isinstance(lr, torch.Tensor)):
            raise TypeError(f"lr must be a float or torch.Tensor. But lr is {type(lr)}.")

        # updated_params = params.clone()
        if self.momentum > 0:
            self.velocity = self.momentum * self.velocity - lr * grad
            # updated_params -= self.velocity
            return params - self.velocity
        else:
            return params - lr * grad


class DyncAdam:
    def __init__(self, params, betas=(0.9, 0.999), eps=1e-8, lr=None, t=None, m=None, v=None, v_hat_eps=1e-44, **kwargs):
        if not isinstance(params, torch.Tensor):
            raise TypeError("params must be a single torch.Tensor")

        self.params = params
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.v_hat_eps = v_hat_eps
        self.m = torch.zeros_like(params) if m is None or not isinstance(m, torch.Tensor) else m
        self.v = torch.zeros_like(params) if v is None or not isinstance(v, torch.Tensor) else v
        self.t = torch.tensor(0) if t is None or not isinstance(t, torch.Tensor) else t
        self.lr = lr

    def step(self, params: torch.Tensor, grad: torch.Tensor, lr: torch.Tensor=None):
        if grad is None or not isinstance(grad, torch.Tensor):
            raise ValueError(f"grad must be a torch.Tensor. But grad is {type(grad)}.")
        
        if lr is None or not (isinstance(lr, float) or isinstance(lr, torch.Tensor)):
            if self.lr is not None:
                lr = self.lr
            else:
                raise TypeError(f"lr must be a float or torch.Tensor. But lr is {type(lr)}.")

        device = params.device

        self.t = self.t.to(device)
        self.m = self.m.to(device)
        self.v = self.v.to(device)

        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        v_hat = torch.clamp(v_hat, min=self.v_hat_eps)
        return params - lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    
    def get_lr(self):
        return self.lr
    
    def set_lr(self, lr):
        self.lr = lr