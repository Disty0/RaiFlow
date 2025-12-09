import torch
from torch import nn
from torch.nn import functional as F


class RaiFlowRMSNorm(nn.RMSNorm):
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        return F.rms_norm(
            hidden_states.clamp(-65504, 65504).to(dtype=torch.float32),
            self.normalized_shape,
            self.weight,
            self.eps,
        ).to(dtype=hidden_states.dtype)


class RaiFlowFeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int, ff_mult: int = 4, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        inner_dim = int(max(dim, dim_out) * ff_mult)

        self.ff_gate = nn.Sequential(
            nn.Linear(dim, inner_dim, bias=bias),
            nn.GELU(approximate="none"),
            nn.Dropout(dropout),
        )

        self.ff_proj = nn.Linear(dim, inner_dim, bias=bias)
        self.ff_out = nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        return self.ff_out(self.ff_proj(hidden_states) * self.ff_gate(hidden_states))
