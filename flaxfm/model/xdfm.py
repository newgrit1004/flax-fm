from flax import linen as nn
import jax
from flaxfm.layer import  FeaturesEmbeddingFlax, MultiLayerPerceptronFlax, FeaturesLinearFlax, CompressedInteractionNetworkFlax
import numpy as np
from typing import Sequence

class ExtremeDeepFactorizationMachineModelFlax(nn.Module):
    field_dims : np.ndarray
    embed_dim : int
    cross_layer_sizes : Sequence[int]
    split_half : bool
    mlp_dims : Sequence[int]
    dropout : float = 0.2

    def setup(self):
        self.embedding = FeaturesEmbeddingFlax(self.field_dims, self.embed_dim)
        self.embed_output_dim = len(self.field_dims) * self.embed_dim
        self.cin = CompressedInteractionNetworkFlax(len(self.field_dims), self.cross_layer_sizes, self.split_half)
        self.linear = FeaturesLinearFlax(self.field_dims)
        self.mlp = MultiLayerPerceptronFlax(self.mlp_dims, self.dropout)


    @nn.compact
    def __call__(self, x, training:bool=True):
        embed_x = self.embedding(x)
        x = self.linear(x) + self.cin(embed_x) + self.mlp(embed_x.reshape(-1, self.embed_output_dim), training)
        return jax.nn.sigmoid(x.squeeze(1))