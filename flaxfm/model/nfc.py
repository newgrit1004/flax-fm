from flax import linen as nn
import jax.numpy as jnp
import jax
from flaxfm.layer import FeaturesEmbeddingFlax, MultiLayerPerceptronFlax, FeaturesLinearFlax
from typing import Sequence
import numpy as np

class NeuralCollaborativeFilteringFlax(nn.Module):
    field_dims: np.ndarray
    embed_dim: int
    user_field_idx : np.ndarray
    mlp_dims : Sequence[int]
    item_field_idx : np.ndarray
    dropout : float = 0.2


    def setup(self):
        self.embedding = FeaturesEmbeddingFlax(self.field_dims, self.embed_dim)
        self.embed_output_dim = len(self.field_dims) * self.embed_dim
        self.fc = nn.Dense(1)
        self.mlp = MultiLayerPerceptronFlax(self.mlp_dims, self.dropout)

    def __call__(self, x, training:bool=True):
        x = self.embedding(x)
        user_x = x[:, self.user_field_idx].squeeze(1)
        item_x = x[:, self.item_field_idx].squeeze(1)
        x = self.mlp(x.reshape(-1, self.embed_output_dim), training, output_layer=False)
        gmf = user_x * item_x
        x = jnp.concatenate((gmf, x), axis=1)
        x = self.fc(x).squeeze(1)
        return jax.nn.sigmoid(x)