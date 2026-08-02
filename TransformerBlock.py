from RMSNorm import RMSNorm
from MHA import MHA
from SwiGLU import SwiGLU

import torch 
import torch.nn as nn
class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.rms1=RMSNorm(cfg)
        self.attention=MHA(cfg)
        self.rms2=RMSNorm(cfg)
        self.ffn=SwiGLU(cfg)
    def forward(self,x):
        rms1=self.rms1(x)
        attn=self.attention(rms1)
        attn_op=attn+x
        rms2=self.rms2(attn_op)
        swiglu=self.ffn(rms2)
        return attn_op+swiglu