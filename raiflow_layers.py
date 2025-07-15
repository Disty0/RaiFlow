from typing import Union
import torch
from torch import nn


class RaiFlowDynamicTanh(nn.Module):
    def __init__(self, dim: Union[int, torch.Size], init_scale: float = 0.2):
        super().__init__()
        if isinstance(dim, int):
            dim = (dim,)
        dim = torch.Size(dim)

        self.scale_in = nn.Parameter(torch.ones(dim) * init_scale)
        self.scale_out = nn.Parameter(torch.ones(dim) / init_scale)
        self.shift_in = nn.Parameter(torch.zeros(dim))
        self.shift_out = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states):
        hidden_states = torch.tanh(torch.addcmul(self.shift_in, hidden_states, self.scale_in))
        hidden_states = torch.addcmul(self.shift_out, hidden_states, self.scale_out)
        return hidden_states


class RaiFlowFeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner_dim = int(max(dim, dim_out) * ff_mult)

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
