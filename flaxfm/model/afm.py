import numpy as np
import jax
from flax import linen as nn
from flaxfm.layer import FeaturesLinearFlax, FeaturesEmbeddingFlax, AttentionalFactorizationMachineFlax

class AttentionalFactorizationMachineModelFlax(nn.Module):
    field_dims : np.ndarray
    embed_dim : int
    attn_size : int
    dropout : float = 0.2

    def setup(self):
        self.num_fields = len(self.field_dims)
        self.embedding = FeaturesEmbeddingFlax(self.field_dims, self.embed_dim)
        self.linear = FeaturesLinearFlax(self.field_dims)
        self.afm = AttentionalFactorizationMachineFlax(self.attn_size, self.dropout)

    def __call__(self, x, training:bool=True):
        x = self.linear(x) + self.afm(self.embedding(x), training)
        return jax.nn.sigmoid(x.squeeze(1))