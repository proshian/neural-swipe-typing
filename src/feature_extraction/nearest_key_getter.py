from torch import Tensor
import torch.nn as nn

from .distance_getter import DistanceGetter


class NearestKeyGetter(nn.Module):
    def __init__(self,
                 grid: dict,
                 tokenizer,
                 ) -> None:
        super().__init__()
        self.distance_getter = DistanceGetter(
            grid, tokenizer, missing_distance_val=float('inf'))

    def forward(self, coords: Tensor) -> Tensor:
        return self.distance_getter(coords).argmin(dim=-1)
