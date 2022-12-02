from flax import linen as nn
import numpy as np
import jax
import jax.numpy as jnp
from flaxfm.utils import slicing

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