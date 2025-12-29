from typing import Optional, Protocol

import torch
import torch.nn as nn
from torch import Tensor

from modules.positional_encodings import SinusoidalPositionalEncoding


class SwipeFeatureEmbedder(Protocol):    
    def __call__(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *features: Variable number of input tensors.
            The features are provided by a SwipeFeatureExtractor
        Returns:
            Single output tensor with the embedded representation
        """
        ...



class WeightedSumEmbedding(nn.Module):
    """
    Computes embedding as a weighted sum of embeddings

    Is used as a swipe dot embedding: the embedding is
    a weighted sum of embeddings of all key on keyboard
    """
    def __init__(self, n_elements: int, dim: int) -> None:
        """
        Arguments:
        ----------
        n_elements: int
            Number of categorical elements to embed. The elements are
            supposed to be indices of keys on a keyboard.
        dim: int
            Dimension of the embedding
        """
        super().__init__()
        # Storing ebeddings and computing weighted sum of them is same as
        # performing a linear transformation:
        # `weights @ embedding_matrix = result_embedding` 
        # where `embedding_matrix` is of shape (n_keys, dim)
        # `weights` is of shape (seq_len, batch_size, n_keys)
        # result_embedding is of shape (seq_len, batch_size, dim)
        # We can think embedding_matrix as a concatenation of `n_keys` embeddings of shape (1, dim),
        # and i-th element of result_embedding is a weighted sum of i-th elements of these embeddings
        self.embeddings = nn.Linear(n_elements, dim)

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
        ----------
        weights: torch.Tensor, shape (seq_len, batch_size, n_elements) or (batch_size, seq_len, n_elements)
            weights[:, :, i] is weight of i-th embedding (embedding of i-th key)
        """
        return self.embeddings(weights)


class WeightsSumEmbeddingWithPos(WeightedSumEmbedding):
    def __init__(self, n_elements, dim, max_len, device) -> None:
        super().__init__(n_elements, dim)
        self.pos_encoder = SinusoidalPositionalEncoding(dim, max_len, device)

    def forward(self, weights):
        emb = super().forward(weights)
        emb = self.pos_encoder(emb)
        return emb




class NearestEmbeddingWithPos(nn.Module):
    """
    Parameters:
    -----------
    n_elements: int
        Number of tokenized keyboard keys
    """
    def __init__(self, n_elements, dim, max_len, device, dropout) -> None:
        super().__init__()
        self.key_emb = nn.Embedding(n_elements, dim)
        self.pos_encoder = SinusoidalPositionalEncoding(dim, max_len, device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, kb_ids_seq: torch.Tensor):
        kb_k_emb = self.key_emb(kb_ids_seq)
        kb_k_emb = self.pos_encoder(kb_k_emb)
        kb_k_emb = self.dropout(kb_k_emb)   
        return kb_k_emb


class SeparateTrajAndWEightedEmbeddingWithPos(nn.Module):
    # Separate in a sense that we don't apply a linear layer to mix the layers
    def __init__(self, n_keys, key_emb_size, max_len, device, dropout = 0.1) -> None:
        super().__init__()
        self.weighted_sum_emb = WeightsSumEmbeddingWithPos(n_keys, key_emb_size, max_len, device)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, traj_feats: Tensor, kb_key_weights: Tensor) -> Tensor: 
        kb_k_emb = self.weighted_sum_emb(kb_key_weights)
        kb_k_emb = self.dropout(kb_k_emb)
        x = torch.cat((traj_feats, kb_k_emb), dim = -1)
        return x
    

class SeparateTrajAndNearestEmbeddingWithPos(nn.Module):
    # Separate in a sense that we don't apply a linear layer to mix the layers
    def __init__(self, n_keys, key_emb_size, max_len, device, dropout = 0.1) -> None:
        super().__init__()
        self.key_emb = NearestEmbeddingWithPos(
            n_keys, key_emb_size, max_len, device, dropout)
    
    def forward(self, traj_feats: Tensor, nearest_kb_key_ids: Tensor) -> Tensor:
        kb_k_emb = self.key_emb(nearest_kb_key_ids)
        x = torch.cat((traj_feats, kb_k_emb), dim = -1)
        return x
    

class VectorizedKeyboardDistributions(nn.Module):
    """
    A module that holds trainable multivariate normal distributions (MVNDs).
    The MVNDs are used as weights for weighted sum embedding of keyboard keys.
    The MVNDs probs can be interpreted as certainty that a point "belongs to
    a segment of the swipe corresponding to a given key".
    """
    def __init__(self, n_keys, key_centers: torch.Tensor):
        super().__init__()

        IGNORE_COORDINATE_VALUE = (-1.0, -1.0)

        self.n_keys = n_keys
        
        assert key_centers.shape == (n_keys, 2)
        
        # Identify keys that should be ignored (centers at -1, -1)
        is_active = ~(
            (key_centers[:, 0] == IGNORE_COORDINATE_VALUE[0]) 
            & (key_centers[:, 1] == IGNORE_COORDINATE_VALUE[1])
        )
        active_idx = torch.nonzero(is_active, as_tuple=False).squeeze(1)

        self.register_buffer("active_keys_mask", is_active)      # (n_keys,)
        self.register_buffer("active_keys_idx", active_idx)      # (n_active,)

        self.n_active_keys = int(active_idx.numel())

        assert self.n_active_keys > 0, "No active keys found in key_centers!"

        self.means = nn.Parameter(key_centers[is_active].clone())  # (n_active, 2)

        # Cholesky factors (vectorized lower-triangular values)
        self.L_params = nn.Parameter(torch.randn(self.n_active_keys, 3) * 0.1)  # (n_active, 3)
        self.eps = 1e-6
    

    def get_covariances(self):
        # Vectorized construction of L for active keys only
        L = torch.zeros(
            self.n_active_keys, 2, 2,
            device=self.L_params.device,
            dtype=self.L_params.dtype,
        )
        tril_indices = torch.tril_indices(2, 2, device=self.L_params.device)
        L[:, tril_indices[0], tril_indices[1]] = self.L_params

        # Softplus on diagonals for stability
        diag_idx = torch.arange(2, device=self.L_params.device)
        L[:, diag_idx, diag_idx] = torch.nn.functional.softplus(L[:, diag_idx, diag_idx]) + self.eps

        return L @ L.transpose(-1, -2)  # (n_active, 2, 2)


    def forward(self, x):
        # x: (seq_len, batch_size, 2)
        out = torch.zeros(
            *x.shape[:-1], self.n_keys,
            device=x.device,
            dtype=x.dtype
        )

        covs = self.get_covariances()
        dist = torch.distributions.MultivariateNormal(self.means, covs)

        # x: (seq_len, batch_size, 1, 2) -> log_prob: (seq_len, batch_size, n_active)
        log_probs_active = dist.log_prob(x.unsqueeze(-2))

        # Scatter active results back into full key dimension
        out.index_copy_(-1, self.active_keys_idx, log_probs_active)
        return out


class SeparateTrajAndTrainableWeightedEmbeddingWithPosV2(nn.Module):
    # Separate in a sense that we don't apply a linear layer to mix the layers
    def __init__(self, n_keys, key_emb_size, max_len, device, dropout = 0.1,
                 key_centers: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.weights_getter = VectorizedKeyboardDistributions(n_keys, key_centers)
        self.weighted_sum_emb = WeightsSumEmbeddingWithPos(n_keys, key_emb_size, max_len, device)
        self.dropout = nn.Dropout(dropout)


    def forward(self, traj_feats: Tensor) -> Tensor:
        xy = traj_feats[...,:2]
        kb_key_weights = self.weights_getter(xy)
        kb_k_emb = self.weighted_sum_emb(kb_key_weights)
        kb_k_emb = self.dropout(kb_k_emb)
        x = torch.cat((traj_feats, kb_k_emb), dim = -1)
        return x






# class PSDSymmetricMatrix(nn.Module):
#     """
#     A trainable symmetric positive semi-definite matrix.
#     """
#     def __init__(self, N) -> None:
#         super().__init__()
#         self.N = N
#         self.num_params = (N * (N + 1)) // 2
#         self.trainable_params = nn.Parameter(torch.randn(self.num_params))

#     @property
#     def psd_sym_matrix(self):
#         A = torch.zeros(self.N, self.N, device=self.trainable_params.device)
#         i, j = torch.triu_indices(self.N, self.N)
#         A[i, j] = self.trainable_params
#         A = A + A.T - torch.diag(A.diag())
#         A = torch.relu(A)
#         return A

#     def forward(self):
#         return self.psd_sym_matrix


class PSDSymmetricMatrix(nn.Module):
    """
    A trainable symmetric positive semi-definite matrix.
    """
    def __init__(self, N) -> None:
        super().__init__()
        self.matrix = nn.parameter.Parameter(torch.randn(N, N), requires_grad=True)

    @property
    def psd_sym_matrix(self):
        return self.matrix @ self.matrix.T

    def forward(self):
        return self.psd_sym_matrix
    

    
class TrainableMultivariateNormal2d(nn.Module):
    def __init__(self, mean: Optional[torch.Tensor] = None) -> None:
        """
        mean: torch.Tensor of shape (2,)
            If present is supposed to be a center of a corresponding key on a keyboard.

        """
        super().__init__()
        if mean is None:
            mean = torch.randn(2)
        self.mean = nn.parameter.Parameter(mean, requires_grad=True)
        # covariance is a semi-positive definite symmetric matrix
        self._psd_sym_matrix = PSDSymmetricMatrix(2)
    
    @property
    def covariance(self):
        return self._psd_sym_matrix.psd_sym_matrix

    def forward(self, x):
        # x is of shape (seq_len, batch_size, 2)
        return torch.distributions.MultivariateNormal(self.mean, self.covariance).log_prob(x)


class KeyboardKeyNormalDistributions(nn.Module):
    def __init__(self, n_keys, key_centers: Optional[torch.Tensor] = None) -> None:
        """
        Arguments:
        ----------
        n_keys: int
            Number of keys on a keyboard.
        key_centers: torch.Tensor of shape (n_keys, 2) or None
            Optional key centers tensor that is used to initialize the means 
            of the distributions. If None, the means are initialized randomly.
            The missing keys are supposed to have a center (-1, -1); 
            they won't have a distribution.
        """

        super().__init__()
        if key_centers is None:
            key_centers = torch.randn(n_keys, 2)

        assert key_centers.shape == (n_keys, 2)
        self.unpresent_keys = []
        if key_centers is not None:
            self.unpresent_keys = [i for i in range(n_keys) 
                                   if key_centers[i][0] == -1 and key_centers[i][1] == -1]
        
        self.distributions = nn.ModuleList([None if i in self.unpresent_keys
                                            else TrainableMultivariateNormal2d(mean) 
                                            for i, mean in enumerate(key_centers)])
        
    def forward(self, coords):
        # coords.shape = (seq_len, batch_size, 2)
        # returns shape = (seq_len, batch_size, n_keys)      
        weights_lst = [
            torch.zeros(coords.shape[:-1], device = coords.device) if i in self.unpresent_keys 
                else dist(coords) 
            for i, dist in enumerate(self.distributions)
        ]
        return torch.stack(weights_lst, dim = -1)


class SeparateTrajAndTrainableWeightedEmbeddingWithPos(nn.Module):
    # Separate in a sense that we don't apply a linear layer to mix the layers
    def __init__(self, n_keys, key_emb_size, max_len, device, dropout = 0.1,
                 key_centers: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.weights_getter = KeyboardKeyNormalDistributions(n_keys, key_centers)
        self.weighted_sum_emb = WeightsSumEmbeddingWithPos(n_keys, key_emb_size, max_len, device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, traj_feats: Tensor) -> Tensor:
        xy = traj_feats[..., :2]
        kb_key_weights = self.weights_getter(xy)
        kb_k_emb = self.weighted_sum_emb(kb_key_weights)
        kb_k_emb = self.dropout(kb_k_emb)
        x = torch.cat((traj_feats, kb_k_emb), dim = -1)
        return x
