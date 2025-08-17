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


# this is to support int8 matmul, otherwise it is the same thing as normal Conv1d
class LinearConv1d(nn.Module):
    def __init__(self, dim: int, dim_out: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        if padding_mode == "zeros":
            self.padding_mode = "constant"
        else:
            self.padding_mode = padding_mode
        if isinstance(padding, int):
            self.padding = (0, 0, padding, padding)
        else:
            self.padding = (0,0) + padding

        self.linear = nn.Linear((dim * kernel_size), dim_out, bias=bias)

    def forward(self, hidden_states):
        effective_kernel_size = ((self.kernel_size - 1) * self.dilation + 1) if self.dilation > 1 else self.kernel_size
        hidden_states = torch.nn.functional.pad(hidden_states, self.padding, mode=self.padding_mode).unfold(1, effective_kernel_size, self.stride)
        if self.dilation > 1:
            hidden_states_stride = hidden_states.stride()
            hidden_states = hidden_states.as_strided((*hidden_states.shape[:-1], self.kernel_size), (*hidden_states_stride[:-1], (hidden_states_stride[-1] * self.dilation)))
        hidden_states = hidden_states.flatten(-2,-1)
        hidden_states = self.linear(hidden_states)
        return hidden_states


class RaiFlowFeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner_dim = int(max(dim, dim_out) * ff_mult)

        self.ff_gate = nn.Sequential(
            LinearConv1d(dim, inner_dim, kernel_size=3, padding=1, bias=True),
            nn.GELU(approximate="none"),
            nn.Dropout(dropout),
        )

        self.ff_proj = nn.Linear(dim, inner_dim, bias=True)
        self.ff_out = nn.Linear(inner_dim, dim_out, bias=True)
        self.bias = nn.Parameter(torch.zeros(inner_dim))

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        return self.ff_out(torch.addcmul(self.bias, self.ff_gate(hidden_states), self.ff_proj(hidden_states)))
