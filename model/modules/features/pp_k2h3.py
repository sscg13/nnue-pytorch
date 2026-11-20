import torch
from torch import nn

from .pp_3wide import PP3Wide


class PPK2H3(PP3Wide):
    HASH = 0x46F5CB78
    FEATURE_NAME = "PP_K2H3^"
    INPUT_FEATURE_NAME = "PP_K2H3"

    NUM_PLANES = 2163
    NUM_BUCKETS = 2 * 3
    NUM_INPUTS = NUM_PLANES * NUM_BUCKETS
    NUM_INPUTS_VIRTUAL = NUM_PLANES
    NUM_REAL_FEATURES = NUM_INPUTS

    def __init__(self, num_outputs: int):
        super().__init__(num_outputs)
        self.virtual_weight = nn.Parameter(
            self.weight.new_zeros((self.NUM_PLANES, num_outputs))
        )

    def merged_weight(self) -> torch.Tensor:
        return (
            self.weight.view(self.NUM_BUCKETS, self.NUM_PLANES, -1) + self.virtual_weight
        ).flatten(0, 1)

    @torch.no_grad()
    def coalesce(self) -> None:
        self.weight.view(self.NUM_BUCKETS, self.NUM_PLANES, -1).add_(self.virtual_weight)
        self.zero_virtual_weights()

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        self.virtual_weight.zero_()

    @torch.no_grad()
    def init_weights(self) -> None:
        self.zero_virtual_weights()

    @torch.no_grad()
    def clip_weights(self, quantization) -> None:
        min_weight, max_weight = quantization.get_ft_weight_bounds(
            self.EXPORT_WEIGHT_DTYPE
        )
        weights = self.weight.view(self.NUM_BUCKETS, self.NUM_PLANES, -1)
        weights.add_(self.virtual_weight)
        weights.clamp_(min_weight, max_weight)
        weights.sub_(self.virtual_weight)

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.merged_weight()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.copy_(export_weight)
        self.zero_virtual_weights()
