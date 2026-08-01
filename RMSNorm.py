import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.hidden_dim=cfg["hidden_size"]
        self.eps=cfg["eps"]
        self.weight=nn.Parameter(torch.ones(self.hidden_dim))
    def forward(self,x):
        rms=torch.sqrt(torch.mean(torch.square(x),dim=-1,keepdim=True))
        # print("Rms Shape",rms.shape)
        normalized = x / (rms + self.eps)
        # print(normalized.shape)
        return normalized * self.weight