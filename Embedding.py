import torch
import torch.nn as nn
class Embedding(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.vocab_size=cfg["vocab_size"]
        self.embedding_dim=cfg["hidden_size"]
        #self.embedding=nn.Parameter(torch.randn(self.vocab_size,self.embedding_dim))
        self.embedding=nn.Parameter(torch.empty(self.vocab_size,self.embedding_dim))
        nn.init.normal_(self.embedding,mean=0.0,std=0.02)
    def forward(self,x):
        return self.embedding[x]