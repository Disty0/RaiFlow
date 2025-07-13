from typing import Union
import numbers
import torch
from torch import nn


class DynamicTanh(nn.Module):
    def __init__(self, dim: Union[int, torch.Size], init_alpha: float = 0.2, elementwise_affine: bool = True, bias: bool = True):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.elementwise_affine = elementwise_affine

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)
        self.weight = None
        self.bias = None

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim) / init_alpha)
        else:
            self.weight = nn.Parameter(torch.ones(1) / init_alpha)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.dim))

    def forward(self, hidden_states):
        hidden_states = torch.tanh(torch.mul(hidden_states, self.alpha))
        if self.bias is not None:
            hidden_states = torch.addcmul(self.bias, hidden_states, self.weight)
        else:
            hidden_states = torch.mul(hidden_states, self.weight)
        return hidden_states


class RaiFlowFeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner_dim = int(dim * ff_mult)

        self.ff_gate = nn.Sequential(
            nn.Linear(dim, inner_dim, bias=True),
            nn.GELU(approximate="none"),
            nn.Dropout(dropout),
        )

        self.ff_proj = nn.Linear(dim, inner_dim, bias=True)
        self.ff_out = nn.Linear(inner_dim, dim_out, bias=True)
        self.bias = nn.Parameter(torch.zeros(inner_dim))

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        return self.ff_out(torch.addcmul(self.bias, self.ff_gate(hidden_states), self.ff_proj(hidden_states)))
