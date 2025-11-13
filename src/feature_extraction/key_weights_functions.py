from typing import Protocol

import torch
from torch import Tensor


class WeightsFunction(Protocol):
    def __call__(self, distances: Tensor) -> Tensor:
        """
        Arguments:
        ----------
        distances: Tensor
            A 2d tensor where i-th element is a vector of distances 
            for i-th swipe point and each of keyboard keys. It's supposed
            that distances are measured in half_key_diagonals 
            (distances = raw_distances / half_key_diagonal)
        """
        ...


class WeightsFnV1:
    def __init__(self, bias = 4, scale = 1.8) -> None:
        self.bias = bias
        self.scale = scale

    def __call__(self, distances: Tensor) -> Tensor:
        r"""
        Arguments:
        ----------
        distances: Tensor
            A 2d tensor where i-th element is a vector of distances 
            for i-th swipe point and each of keyboard keys. It's supposed
            that distances are measured in half_key_diagonals 
            (distances = raw_distances / half_key_diagonal)

        $$f(x) = \frac{1}{1+e^{\frac{s \cdot x}{key\_radius} - b}}$$
        b = bias = 4
        s = scale = 1.8
        """

        # return 1 / (1 + torch.exp(1.8 * distances - 4))
        sigmoid_input = distances * (-self.scale) + self.bias
        return torch.nn.functional.sigmoid(sigmoid_input)        


class WeightsFunctionV1Softmax:
    def __init__(self, bias = 4, scale = 1.8) -> None:
        self.bias = bias
        self.scale = scale    

    def __call__(self, distances: Tensor) -> Tensor:
        """
        Arguments:
        ----------
        distances: Tensor
            A 2d tensor where i-th element is a vector of distances 
            for i-th swipe point and each of keyboard keys. It's supposed
            that distances are measured in half_key_diagonals 
            (distances = raw_distances / half_key_diagonal)
        """
        mask = torch.isinf(distances)
        sigmoid_input = distances * (-self.scale) + self.bias
        weights = torch.nn.functional.sigmoid(sigmoid_input)
        # -inf to zero out unpresent values and have a sum of one 
        weights.masked_fill_(mask, float('-inf'))
        return torch.nn.functional.softmax(weights, dim=1)


class WeightsFunctionSigmoidNormalizedV1:
    def __init__(self, bias = 4, scale = 1.8) -> None:
        self.bias = bias
        self.scale = scale

    def __call__(self, distances: Tensor) -> Tensor:
        """
        Arguments:
        ----------
        distances: Tensor
            A 2d tensor where i-th element is a vector of distances 
            for i-th swipe dot and each of keyboard keys. It's supposed
            that distances are measured in half_key_diagonals 
            (distances = raw_distances / half_key_diagonal)
        """
        #! It may be a good idea to move division by half_key_diag outside
        # this function.  The division is just a scaling of distances
        # so that they are not in pixels but use half_key_diag as a unit. 
        sigmoid_input = distances * (-self.scale) + self.bias
        sigmoidal_weights = torch.nn.functional.sigmoid(sigmoid_input)
        weights = sigmoidal_weights / sigmoidal_weights.sum(dim=1, keepdim=True)
        return weights
