import torch
import torch.nn as nn
class ROPE(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.hidden_size=cfg["hidden_size"]
        self.num_heads=cfg["num_attention_heads"]
        assert self.hidden_size%self.num_heads==0
        self.head_dim = self.hidden_size // self.num_heads
        assert self.head_dim % 2 == 0
        i=torch.arange(self.head_dim//2)
        self.theta=cfg["rope_theta"]
        inv_freq=1/torch.pow(torch.tensor(self.theta),2*i/self.head_dim)
        self.register_buffer("inv_freq",inv_freq)
    def forward(self,x):
        seq_len=x.shape[1]
        pos=torch.arange(seq_len,device=x.device)
        theta=pos.unsqueeze(1)*self.inv_freq.unsqueeze(0)
        cos=torch.cos(theta).unsqueeze(0).unsqueeze(2)
        sin=torch.sin(theta).unsqueeze(0).unsqueeze(2)
        x_even=x[...,::2]
        x_odd=x[...,1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        rot=torch.stack((rot_even,rot_odd),dim=-1).flatten(start_dim=-2)
        return rot