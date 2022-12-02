from flax import linen as nn
import jax
from flaxfm.layer import FeaturesLinearFlax, FeaturesEmbeddingFlax, FactorizationMachineFlax
import numpy as np

class FactorizationMachineModelFlax(nn.Module):
    field_dims : np.ndarray
    embed_dim : int

    def setup(self):
        self.embedding = FeaturesEmbeddingFlax(self.field_dims, self.embed_dim)
        self.linear = FeaturesLinearFlax(self.field_dims)
        self.fm = FactorizationMachineFlax(reduce_sum=True)
    @nn.compact
    def __call__(self, x):
        x = self.linear(x) + self.fm(self.embedding(x))
        return jax.nn.sigmoid(x.squeeze(1))