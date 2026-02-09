import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from modules.distill.utils.optimizers import get_dynamic_optimizer
from modules.distill.utils.schedulers import get_dynamic_scheduler
from modules.distill.utils.adapters import get_adapter_model
from modules.distill.utils.reparam_module import ReparamModule


def restore_module_parameters(reparam_model, flat_params):
    reparam_model._unflatten_param(flat_params)

    for mn, n in reparam_model._param_infos:
        module = reparam_model._get_module_from_name(mn)

        tensor = getattr(module, n)
        if hasattr(module, n):
            delattr(module, n)

        module.register_parameter(n, torch.nn.Parameter(tensor))

    for mn, n, shared_mn, shared_n in reparam_model._shared_param_infos:
        module = reparam_model._get_module_from_name(mn)
        shared_module = reparam_model._get_module_from_name(shared_mn)
        shared_param = getattr(shared_module, shared_n)

        if hasattr(module, n):
            delattr(module, n)
        module.register_parameter(n, shared_param)

    return reparam_model
class DyncAdam:
    def __init__(self, params, betas=(0.9, 0.999), eps=1e-8, lr=None, t=None, m=None, v=None, v_hat_eps=1e-44,
                 **kwargs):
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

    def step(self, params: torch.Tensor, grad: torch.Tensor, lr: torch.Tensor = None):
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

class DyncExpDecayScheduler:
    def __init__(
            self,
            step: int,
            lr_decay_factor: float,
            lr_decay_step_size: float,
            lr: float = None
    ):
        self.lr = lr
        self.step = step
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_step_size = lr_decay_step_size

    def step_lr(self, lr: float = None):
        if lr is None:
            lr = self.lr

        lr = lr * self.lr_decay_factor ** (self.step / self.lr_decay_step_size)
        self.step += 1
        return lr
class SequentialMLP(nn.Module):
    def __init__(self, input_size, input_dim=3, hidden_size=128, output_size=1):
        super(SequentialMLP, self).__init__()
        self.flatten = nn.Flatten()
        # 首先将5个点的坐标展平或聚合
        self.network = nn.Sequential(
            nn.Linear(input_size * input_dim, hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # 展平为 [batch_size, 15]
        x = self.network(x)
        return x.squeeze(-1)  # 移除最后的维度


adapter_dict = {
    "peft_type": "LORA",
    "r": 4,
    "lora_alpha": 4,
    "target_modules": [
        "network.0",
        "network.1"
    ]
}
num_samples = 3
data_points = 5  # 每个data有5个点
feature_dim = 3  # 每个点是3维坐标
torch.manual_seed(42)
# 生成50个样本，每个样本是5×3的张量
data = torch.randn(num_samples, data_points, feature_dim)
# 生成50个对应的标量标签
labels = torch.randn(num_samples)
dataset = TensorDataset(data, labels)
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

testdate = torch.randn(num_samples, data_points, feature_dim)
testlabels = torch.randn(num_samples)
testdataset = TensorDataset(testdate, testlabels)
testdataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)

if __name__ == '__main__':
    torch.manual_seed(42)  # 设置PyTorch随机种子
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    SModel = SequentialMLP(input_size=5).to(device)
    torch.manual_seed(123)  # 使用不同的种子
    Tmodel = SequentialMLP(input_size=5).to(device)

    target_expert_params = []
    for param in Tmodel.parameters():
        target_expert_params.append(param.data.reshape(-1))
        print(param.data.reshape(-1))
    # start_params = []
    # for param in SModel.parameters():
    #     start_params.append(param.data.reshape(-1))
    #     print(param.data.reshape(-1))
    #
    # start_params = torch.cat(start_params, dim=0).to(device)
    target_expert_params = torch.cat(target_expert_params, dim=0).to(device)

    torch.manual_seed(42)
    student_net = get_adapter_model(adapter_dict, SModel).to(device)
    for name, param in student_net.named_parameters():
        print(name, param.shape)

    student_net.train()
    student_net_params = []
    base_model_params_mask = []
    for name, param in student_net.named_parameters():
        if 'lora_' not in name.lower():
            base_model_params_mask.append(torch.ones_like(param.reshape(-1), dtype=torch.bool))
        else:
            base_model_params_mask.append(torch.zeros_like(param.reshape(-1), dtype=torch.bool))
        student_net_params.append(param.reshape(-1).requires_grad_(True))

    student_net_params = torch.nn.Parameter(torch.cat(student_net_params, dim=0))
    base_model_params_mask = torch.cat(base_model_params_mask, dim=0)
    adapter_params = torch.nn.Parameter(student_net_params[~base_model_params_mask])
    base_model_params = torch.nn.Parameter(student_net_params[base_model_params_mask])

    student_net_params = torch.zeros_like(student_net_params)
    student_net_params.scatter_(0, torch.nonzero(~base_model_params_mask).squeeze(), adapter_params)
    student_net_params.scatter_(0, torch.nonzero(base_model_params_mask).squeeze(), base_model_params)

    student_params_list = [student_net_params]

    student_net = ReparamModule(student_net)



    for name, param in student_net.named_parameters():
        param.requires_grad = True
        print(name, param.requires_grad, param.shape)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    init_lr = 0.001
    scheduler = DyncExpDecayScheduler(
        step=0,
        lr_decay_factor=0.9,  # 每个 step 衰减 0.9^(step/lr_decay_step_size)
        lr_decay_step_size=10,  # 衰减的步长
        lr=init_lr
    )
    optimizer = DyncAdam(params=base_model_params, lr=0.001)
    #optimizer = get_dynamic_optimizer("adam", student_net_params)
    #optimizer = optim.SGD(trainable_params, lr=0.001, momentum=0.9)
    optimizer2 = optim.SGD([adapter_params], lr=0.001, momentum=0.9)
    #schedule = get_dynamic_scheduler("expdecaylr", )
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.1)

    # 设定损失函数
    criterion = nn.MSELoss()
    k = 0

    for epoch in range(0, 100):
        optimizer2.zero_grad()
        # start_expert_params = torch.zeros_like(student_net_params)
        # start_expert_params.scatter_(0, torch.nonzero(~base_model_params_mask).squeeze(), adapter_params)
        # start_expert_params.scatter_(0, torch.nonzero(base_model_params_mask).squeeze(), base_model_params)

        student_net.train()
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            batch_data.requires_grad = True
            #student_params_list[-1].scatter_(0, torch.nonzero(~base_model_params_mask).squeeze(), adapter_params)
            # optimizer.zero_grad()  # 这一步对optimizer 更新过的 参数进行梯度清零
            outputs = student_net(batch_data, flat_param=student_params_list[-1])
            # outputs_2 = SModel(batch_data)
            loss = criterion(outputs, batch_labels)
            # loss.backward(retain_graph=True)
            grad = torch.autograd.grad(loss, student_params_list[-1], create_graph=True)[0]
            next_student_params = optimizer.step(params=student_params_list[-1][base_model_params_mask],
                                                 grad=grad[base_model_params_mask],
                                                 lr=scheduler.step_lr(optimizer.get_lr()))

            student_params_list.append(torch.zeros_like(student_params_list[-1]))
            student_params_list[-1].scatter_(0, torch.nonzero(~base_model_params_mask).squeeze(), adapter_params)
            student_params_list[-1].scatter_(0, torch.nonzero(base_model_params_mask).squeeze(),
                                             next_student_params)
        final_flat = student_params_list[-1]  # 不要 .detach()
        zeros_full = torch.zeros_like(final_flat)
        outer_flat = torch.where(base_model_params_mask, final_flat, zeros_full)
        student_net.eval()
        loss_sum = torch.tensor(0.0, device=device)
        n_samples = 0
        for xb, _ in testdataloader:
            xb = xb.to(device)
            with torch.no_grad():
                teacher_out = Tmodel(xb)
            student_out = student_net(xb, flat_param=outer_flat)
            loss_sum = loss_sum + torch.nn.functional.mse_loss(student_out, teacher_out, reduction='sum')
            n_samples += xb.size(0)
        outer_loss = loss_sum / max(1, n_samples)
        outer_loss.backward()
        # param_loss = torch.tensor(0.0).to(device)
        # align_loss_value = torch.nn.functional.mse_loss(student_params_list[-1][base_model_params_mask], target_expert_params, reduction="sum")
        # align_loss_value.backward(retain_graph=True)

        optimizer2.step()
        scheduler2.step()
    print("训练结束")
