from flax import linen as nn
import jax
from flaxfm.layer import MultiLayerPerceptronFlax, FeaturesLinearFlax, FieldAwareFactorizationMachineFlax
import numpy as np
from typing import Sequence
from flaxfm.utils import Sequential

class FieldAwareNeuralFactorizationMachineModelFlax(nn.Module):
    field_dims : np.ndarray
    embed_dim : int
    mlp_dims : Sequence[int]
    dropout : float = 0.2

    def setup(self):
        self.linear = FeaturesLinearFlax(self.field_dims)
        self.ffm = FieldAwareFactorizationMachineFlax(self.field_dims, self.embed_dim)
        self.ffm_output_dim = len(self.field_dims)*(len(self.field_dims)-1)//2 * self.embed_dim
        self.mlp = MultiLayerPerceptronFlax(self.mlp_dims, self.dropout)

    @nn.compact
    def __call__(self, x, training:bool=True):
        cross_term = self.ffm(x).reshape(-1, self.ffm_output_dim)
        bn_dropout = Sequential([nn.BatchNorm(use_running_average=not training),
                            nn.Dropout(rate=self.dropout, deterministic=not training)
                            ])

        cross_term = bn_dropout(cross_term)
        x = self.linear(x) + self.mlp(cross_term, training)
        return jax.nn.sigmoid(x.squeeze(1))