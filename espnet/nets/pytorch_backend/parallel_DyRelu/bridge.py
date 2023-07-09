from turtle import forward
from torch import nn
import torch
import math


class Local2Global(nn.Module):

    def __init__(self, dim, heads, dropout=0.):
        super(Local2Global, self).__init__()
        assert dim % heads == 0
        # We assume d_v always equals d_k
        self.d_k = dim // heads
        self.heads = heads
        self.to_q = nn.Linear(dim, dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = dim ** -0.5
        self.linear_out = nn.Linear(dim, dim)
        # self.to_out = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.Dropout(dropout)
        # )

    def forward(self, L, G):
        n_batch = G.size(0)
        q = self.to_q(G).view(n_batch, -1, self.heads, self.d_k)
        k = L.view(n_batch, -1, self.heads, self.d_k)
        v = L.view(n_batch, -1, self.heads, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        dots = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  * self.scale
        attn = self.attend(dots) # (batch, head, time1, time2)
        x = torch.matmul(attn, v) # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.heads * self.d_k)
        )  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)


class Global2Local(nn.Module):
    def __init__(self, dim, heads, dropout=0.):
        super(Global2Local, self).__init__()
        assert dim % heads == 0
        # We assume d_v always equals d_k
        self.d_k = dim // heads
        self.heads = heads
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = dim ** -0.5
        self.linear_out = nn.Linear(dim, dim)

    def forward(self, L, G):
        n_batch = G.size(0)
        q = L.view(n_batch, -1, self.heads, self.d_k)
        k = self.to_k(G).view(n_batch, -1, self.heads, self.d_k)
        v = self.to_v(G).view(n_batch, -1, self.heads, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        dots = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  * self.scale
        attn = self.attend(dots) # (batch, head, time1, time2)
        x = torch.matmul(attn, v) # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.heads * self.d_k)
        )  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)
