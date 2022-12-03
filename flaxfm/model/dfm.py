from flax import linen as nn
import jax
from flaxfm.layer import  FactorizationMachineFlax, FeaturesEmbeddingFlax, MultiLayerPerceptronFlax, FeaturesLinearFlax
import numpy as np
from typing import Sequence
from flaxfm.utils import Sequential

class DeepFactorizationMachineModelFlax(nn.Module):
    field_dims : np.ndarray
    embed_dim : int
    mlp_dims : Sequence[int]
    dropout : float = 0.2

    def setup(self):
        self.linear = FeaturesLinearFlax(self.field_dims)
        self.embedding = FeaturesEmbeddingFlax(self.field_dims, self.embed_dim)
        self.embed_output_dim = len(self.field_dims) * self.embed_dim
        self.fm = FactorizationMachineFlax(reduce_sum=True)
        self.mlp = MultiLayerPerceptronFlax(self.mlp_dims, self.dropout)


    @nn.compact
    def __call__(self, x, training:bool=True):
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.reshape(-1, self.embed_output_dim), training)
        return jax.nn.sigmoid(x.squeeze(1))