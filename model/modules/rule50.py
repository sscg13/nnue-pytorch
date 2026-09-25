import torch
from torch import nn

RULE50_ROWS = 12
RULE50_MODES = {"none": 0, "ft": 1, "hidden1": 2, "hidden2": 3}


def rule50_hash(mode: str) -> int:
    # Must match nnue_architecture.h. Keep the baseline format unchanged.
    return 0 if mode == "none" else 0x52355400 ^ (RULE50_MODES[mode] << 16) ^ RULE50_ROWS


def rule50_bucket(clock: torch.Tensor) -> torch.Tensor:
    # Stockfish's TT key leaves clocks 0..13 together, then groups eight plies.
    # Clock 100 and rare larger training clocks share the 94+ row.
    return torch.where(clock < 14, 0, 1 + (clock - 14) // 8).clamp(0, RULE50_ROWS - 1).long()


class Rule50Embedding(nn.Module):
    def __init__(self, width: int, scale: float, stacks: int = 1, ft: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(stacks, RULE50_ROWS, width))
        self.scale = scale
        # Conservative FT headroom; hidden tables are int32 preactivation biases.
        self.limit = 127 if ft else 1 << 20

    def quantized_weight(self, fake_quantize: bool = True):
        value = self.weight.clamp(-self.limit / self.scale, self.limit / self.scale)
        if fake_quantize:
            hard = (value * self.scale).round() / self.scale
            value = value + (hard - value).detach()
        return value

    def forward(self, clock, stacks, fake_quantize=True):
        if clock is None:
            raise ValueError("The selected rule50 embedding requires halfmove clocks")
        indices = rule50_bucket(clock).view(-1)
        return self.quantized_weight(fake_quantize)[stacks, indices]

    @torch.no_grad()
    def export(self, dtype):
        return (self.quantized_weight() * self.scale).round().to(dtype)

    @torch.no_grad()
    def load_export(self, value):
        self.weight.copy_(value / self.scale)
