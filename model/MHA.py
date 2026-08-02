import math
import torch
import torch.nn as nn
from .ROPE import ROPE

class MHA(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.hidden_size=cfg["hidden_size"]
        self.num_head=cfg["num_attention_heads"]
        assert self.hidden_size%self.num_head==0
        self.head_dim=self.hidden_size//self.num_head
        self.w_q=nn.Linear(self.hidden_size,self.hidden_size,bias=cfg["bias"])
        self.w_k=nn.Linear(self.hidden_size,self.hidden_size,bias=cfg["bias"])
        self.w_v=nn.Linear(self.hidden_size,self.hidden_size,bias=cfg["bias"])
        self.rope=ROPE(cfg)
        self.o_proj=nn.Linear(self.hidden_size,self.hidden_size,bias=cfg["bias"])
        mask=torch.tril(torch.ones(cfg["max_position_embeddings"],cfg["max_position_embeddings"]))
        mask = mask.unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask",mask)
    def forward(self,x):
        b,ctx,d_out=x.shape
        
        query=self.w_q(x)
        keys=self.w_k(x)
        values=self.w_v(x)
        
        query=query.view(b,ctx,self.num_head,self.head_dim)
        assert query.shape == (b, ctx, self.num_head, self.head_dim)
        keys=keys.view(b,ctx,self.num_head,self.head_dim)
        values=values.view(b,ctx,self.num_head,self.head_dim)
        
        query=self.rope(query)
        keys=self.rope(keys)
        
        query=query.transpose(1,2)
        keys=keys.transpose(1,2)
        values=values.transpose(1,2)
        scores = query @ keys.transpose(-2, -1)
        assert scores.shape == (b, self.num_head, ctx, ctx)
        scores = scores / math.sqrt(self.head_dim)
        scores=scores.masked_fill_(self.mask[:,:,:ctx,:ctx]==0,float("-inf"))
        weights=torch.softmax(scores,dim=-1)
        attn=weights@values
        attn=attn.transpose(1,2)
        attn=attn.contiguous().view(b,ctx,self.hidden_size)
        assert attn.shape == (b, ctx, self.hidden_size)
        out=self.o_proj(attn)
        return out