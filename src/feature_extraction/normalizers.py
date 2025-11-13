import torch


def identity_function(x: torch.Tensor) -> torch.Tensor:
    return x

class MeanStdNormalizer:
    def __init__(self, mean: float, std: float):
        if std == 0:
            raise ValueError("Standard deviation must be greater than 0. " \
                             "Note that std = 0 means all values are the same. " \
                             "There's something wrong with the data.")
        self.std = std
        self.mean = mean


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class MinMaxNormalizer:
    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.range = max_val - min_val if (max_val - min_val) > 1e-6 else 1.0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.min_val) / self.range
