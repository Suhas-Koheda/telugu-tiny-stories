import torch
import torch.nn as nn
from .TransformerBlock import TransformerBlock
from .Embedding import Embedding
from .RMSNorm import RMSNorm

class Llama(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.embedding=Embedding(cfg)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["num_hidden_layers"])]
        )
        self.rms=RMSNorm(cfg)
        self.linear=nn.Linear(cfg["hidden_size"],cfg["vocab_size"],bias=False)
        self.linear.weight=self.embedding.embedding
    def forward(self,x):
        x=self.embedding(x)
        for layer in self.layers:
            x=layer(x)
        x=self.rms(x)
        x=self.linear(x)
        return x