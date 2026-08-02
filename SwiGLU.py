import torch
import torch.nn as nn

class SILU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x*torch.sigmoid(x)
class SwiGLU(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.gate_proj=nn.Linear(cfg["hidden_size"],cfg["intermediate_size"])
        self.up_proj=nn.Linear(cfg["hidden_size"],cfg["intermediate_size"])
        self.down_proj=nn.Linear(cfg["intermediate_size"],cfg["hidden_size"])
        self.silu=SILU()
    def forward(self,x):
        gate_silu=self.silu(self.gate_proj(x))
        up=self.up_proj(x)
        y=self.down_proj(gate_silu*up)
        return y