from typing import Union
import torch
from torch import nn


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

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        return self.ff_out(self.ff_proj(hidden_states) * self.ff_gate(hidden_states))


class RaiFlowConv1dForward(nn.Module):
    def __init__(self, dim: int, dim_out: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner_dim = int(max(dim, dim_out) * ff_mult)

        self.ff_gate = nn.Sequential(
            nn.Conv1d(dim, inner_dim, 3, padding=1, bias=True),
            nn.GELU(approximate="none"),
            nn.Dropout(dropout),
        )

        self.ff_proj = nn.Linear(dim, inner_dim, bias=True)
        self.ff_out = nn.Linear(inner_dim, dim_out, bias=True)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        return self.ff_out(
            torch.mul(
                self.ff_proj(hidden_states),
                self.ff_gate(hidden_states.transpose(-1,-2)).transpose(-1,-2),
            )
        )


class RaiFlowConv2dForward(nn.Module):
    def __init__(self, dim: int, dim_out: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner_dim = int(max(dim, dim_out) * ff_mult)

        self.ff_gate = nn.Sequential(
            nn.Conv2d(dim, inner_dim, 3, padding=1, bias=True),
            nn.GELU(approximate="none"),
            nn.Dropout(dropout),
        )

        self.ff_proj = nn.Linear(dim, inner_dim, bias=True)
        self.ff_out = nn.Linear(inner_dim, dim_out, bias=True)

    def forward(self, hidden_states: torch.FloatTensor, height: int, width: int) -> torch.FloatTensor:
        return self.ff_out(
            torch.mul(
                self.ff_proj(hidden_states),
                self.ff_gate(hidden_states.transpose(-1,-2).unflatten(-1, (height, width))).flatten(-2,-1).transpose(-1,-2),
            )
        )
