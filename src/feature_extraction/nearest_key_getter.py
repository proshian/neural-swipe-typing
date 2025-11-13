from torch import Tensor

from .distance_getter import DistanceGetter


class NearestKeyGetter:
    def __init__(self,
                 grid: dict,
                 tokenizer,
                 ) -> None:
        self.distance_getter = DistanceGetter(
            grid, tokenizer, missing_distance_val=float('inf'))

    def __call__(self, coords: Tensor) -> Tensor:
        return self.distance_getter(coords).argmin(dim=1)
