def get_dynamic_scheduler(scheduler_type, **scheduler_kwargs):
    scheduler_type = scheduler_type.lower()

    if scheduler_type == "expdecaylr":
        return DyncExpDecayScheduler(**scheduler_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {scheduler_type}. Supported types: 'expdecay'.")


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
