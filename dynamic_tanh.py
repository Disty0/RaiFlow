import numbers
import torch
from torch import nn

class DynamicTanh(nn.Module):
    def __init__(self, dim: int, init_alpha: float = 0.5, elementwise_affine: bool = True, bias: bool = True,):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.elementwise_affine = elementwise_affine

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)
        self.weight = None
        self.bias = None

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            if bias:
                self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states):
        hidden_states = torch.tanh(torch.mul(hidden_states, self.alpha))
        if self.weight is not None:
            if self.bias is not None:
                hidden_states = torch.addcmul(self.bias, hidden_states, self.weight)
            else:
                hidden_states = torch.mul(hidden_states, self.weight)
        return hidden_states
