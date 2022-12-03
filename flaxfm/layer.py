from flax import linen as nn
import numpy as np
import jax
import jax.numpy as jnp
from flaxfm.utils import slicing
from typing import Sequence

class FeaturesLinearFlax(nn.Module):
    field_dims : np.ndarray
    output_dim : int = 1

    def setup(self):
        self.fc = nn.Embed(sum(self.field_dims), self.output_dim)
        self.offsets = np.array((0, *np.cumsum(self.field_dims)[:-1]), dtype=np.compat.long)
    @nn.compact
    def __call__(self, x):
        x = x+self.offsets
        return np.sum(self.fc(x), axis=1)


class FeaturesEmbeddingFlax(nn.Module):
    field_dims : np.ndarray
    embed_dim : int = 16

    def setup(self):
        self.embedding = nn.Embed(sum(self.field_dims), self.embed_dim, embedding_init=jax.nn.initializers.glorot_uniform())
        self.offsets = np.array((0, *np.cumsum(self.field_dims)[:-1]), dtype=np.compat.long)

    @nn.compact
    def __call__(self, x):
        x = x+self.offsets
        return self.embedding(x)


class FactorizationMachineFlax(nn.Module):
    reduce_sum : bool = True

    @nn.compact
    def __call__(self, x):
        square_of_sum = np.sum(x, axis=1)**2
        sum_of_square = np.sum(x**2, axis=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = np.sum(ix, axis=1, keepdims=True)
        return 0.5*ix


class FieldAwareFactorizationMachineFlax(nn.Module):
    field_dims : np.ndarray
    embed_dim : int = 16

    def setup(self):
        self.num_fields = len(self.field_dims)
        self.embeddings = [nn.Embed(sum(self.field_dims), self.embed_dim, embedding_init=jax.nn.initializers.glorot_uniform()) for _ in range(self.num_fields)]
        self.offsets = np.array((0, *np.cumsum(self.field_dims)[:-1]), dtype=np.compat.long)


    @nn.compact
    def __call__(self, x):
        x = x+self.offsets
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(slicing(xs[j],i,1) * slicing(xs[i],j,1))
        ix = jnp.stack(ix, axis=1)
        return jnp.array(ix)


class MultiLayerPerceptronFlax(nn.Module):
    embed_dims : Sequence[int]
    dropout : float

    @nn.compact
    def __call__(self, x, training:bool=True, output_layer=True):
        for embed_dim in self.embed_dims:
            x = nn.Dense(embed_dim)(x)
            x = nn.relu(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)

        if output_layer:
            x = nn.Dense(1)(x)

        return x

class AttentionalFactorizationMachineFlax(nn.Module):
    attn_size : int
    dropout : float = 0.2

    def setup(self):
        self.attention = nn.Dense(self.attn_size)
        self.projection = nn.Dense(1)
        self.fc = nn.Dense(1)

    @nn.compact
    def __call__(self, x, training:bool=True):
        num_fields = x.shape[1]
        row, col = [], []
        for i in range(num_fields -1):
            for j in range(i+1, num_fields):
                row.append(i), col.append(j)
        p, q = jnp.expand_dims(slicing(x, row[0], 1), axis=1), jnp.expand_dims(slicing(x, col[0], 1), axis=1)
        inner_product = p*q
        attn_scores = nn.relu(self.attention(inner_product))
        attn_scores = nn.softmax(self.projection(attn_scores), axis=1)
        attn_scores = nn.Dropout(rate=self.dropout, deterministic=not training)(attn_scores)
        attn_output = np.sum(attn_scores * inner_product, axis=1)
        attn_output = nn.Dropout(rate=self.dropout, deterministic=not training)(attn_output)
        return self.fc(attn_output)