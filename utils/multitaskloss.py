import torch.nn as nn
import torch


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super().__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros(task_num))

    def forward(self, *losses):
        if len(losses) != self.task_num:
            raise ValueError(f"Expected {self.task_num} losses, got {len(losses)}")

        total = 0
        for log_var, loss in zip(self.log_vars, losses):
            precision = torch.exp(-log_var)
            total = total + precision * loss + log_var
        return total
