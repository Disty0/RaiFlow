from typing import Optional, Union
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
            self.weight = nn.Parameter(torch.ones(self.dim))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.dim))

    def forward(self, hidden_states):
        hidden_states = torch.tanh(torch.mul(hidden_states, self.alpha))
        if self.weight is not None:
            if self.bias is not None:
                hidden_states = torch.addcmul(self.bias, hidden_states, self.weight)
            else:
                hidden_states = torch.mul(hidden_states, self.weight)
        return hidden_states


class RaiFlowFeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int, num_attention_heads: int, attention_head_dim: int, heads_per_group: int = 2, router_mult: int = 2, ff_mult: int = 2, dropout: float = 0.1, is_2d: bool = False):
        super().__init__()
        self.is_2d = is_2d
        self.num_groups = num_attention_heads // heads_per_group
        self.inner_dim = int(self.num_groups * attention_head_dim * heads_per_group * router_mult)
        self.ff_dim = int(self.inner_dim * ff_mult)

        self.router = nn.Sequential(
            nn.Linear(dim, self.inner_dim, bias=True),
            DynamicTanh(dim=self.inner_dim, init_alpha=0.2, elementwise_affine=True, bias=True),
            nn.Dropout(dropout),
        )

        if self.is_2d:
            self.conv = nn.Sequential(
                nn.Conv2d(self.inner_dim, self.ff_dim, 3, padding=1, groups=self.num_groups),
                DynamicTanh(dim=(1, self.ff_dim, 1, 1), init_alpha=0.2, elementwise_affine=True, bias=True),
                nn.Dropout(dropout),
                nn.Conv2d(self.ff_dim, self.inner_dim, 3, padding=1, groups=self.num_groups),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(self.inner_dim, self.ff_dim, 3, padding=1, groups=self.num_groups),
                DynamicTanh(dim=(1, self.ff_dim, 1), init_alpha=0.2, elementwise_affine=True, bias=True),
                nn.Dropout(dropout),
                nn.Conv1d(self.ff_dim, self.inner_dim, 3, padding=1, groups=self.num_groups),
            )

        self.proj_out = nn.Sequential(
            DynamicTanh(dim=self.inner_dim, init_alpha=0.2, elementwise_affine=True, bias=True),
            nn.Dropout(dropout),
            nn.Linear(self.inner_dim, dim_out, bias=True),
        )

    def forward(self, hidden_states: torch.FloatTensor, height: Optional[int] = None, width: Optional[int] = None) -> torch.FloatTensor:
        batch_size, seq_len, inner_dim = hidden_states.shape

        router_outputs = self.router(hidden_states)

        ff_hidden_states = router_outputs.transpose(1,2)
        if self.is_2d:
            ff_hidden_states = ff_hidden_states.view(batch_size, self.inner_dim, height, width)

        ff_hidden_states = self.conv(ff_hidden_states)

        if self.is_2d:
            ff_hidden_states = ff_hidden_states.view(batch_size, self.inner_dim, seq_len)
        ff_hidden_states = ff_hidden_states.transpose(1,2)

        ff_hidden_states = ff_hidden_states + router_outputs
        ff_hidden_states = self.proj_out(ff_hidden_states)

        return ff_hidden_states
