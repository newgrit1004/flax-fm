from flax import linen as nn
import jax
from flaxfm.layer import  FactorizationMachineFlax, FeaturesEmbeddingFlax, MultiLayerPerceptronFlax, FeaturesLinearFlax
import numpy as np
from typing import Sequence
from flaxfm.utils import Sequential

class FactorizationSupportedNeuralNetworkModelFlax(nn.Module):
    """
    Note:
    You can only assign Module attributes to self inside Module.setup().
    Outside of that method, the Module instance is frozen (i.e., immutable).
    This behavior is similar to frozen Python dataclasses.
    (https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.SetAttributeFrozenModuleError)
    """
    field_dims : np.ndarray
    embed_dim : int
    mlp_dims : Sequence[int]
    dropout : float = 0.2

    def setup(self):
        self.embedding = FeaturesEmbeddingFlax(self.field_dims, self.embed_dim)
        self.linear = FeaturesLinearFlax(self.field_dims)
        self.mlp = MultiLayerPerceptronFlax(self.mlp_dims, self.dropout)
        self.fm = FactorizationMachineFlax(reduce_sum=False)

    @nn.compact
    def __call__(self, x, training:bool=True):
        fm = Sequential([self.fm,
                            nn.BatchNorm(use_running_average=not training),
                            nn.Dropout(rate=self.dropout, deterministic=not training)
                            ])
        cross_term = fm(self.embedding(x))
        x = self.linear(x) + self.mlp(cross_term, training)
        return jax.nn.sigmoid(x.squeeze(1))