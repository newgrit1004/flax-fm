from flax import linen as nn
from jax import lax
import numpy as np
import jax
import jax.numpy as jnp
from flaxfm.utils import slicing
from typing import Sequence, Any

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



class CompressedInteractionNetworkFlax(nn.Module):
    """
    #TODO(23/01/07): this network is working well, but not the same to original code.(https://github.com/rixwew/pytorch-fm/blob/master/torchfm/model/xdfm.py)
                    The shape of output of conv_layers are different from the original code.
                    Fix the code when I know how to make linen.Conv layer as torch.nn.Conv1d layer.
    """
    input_dim : int
    cross_layer_sizes : Sequence[int]
    split_half : bool

    def setup(self):
        self.num_layers = len(self.cross_layer_sizes)
        self.fc = nn.Dense(1)

    @nn.compact
    def __call__(self, x):
        prev_dim, fc_input_dim = self.input_dim, 0
        conv_layers = list()
        for i in range(self.num_layers):
            cross_layer_size = self.cross_layer_sizes[i]
            conv_layers.append(nn.Conv(cross_layer_size, kernel_size=(1,), strides=1))
            if self.split_half and i != self.num_layers-1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim


        xs = list()
        x0, h = jnp.expand_dims(x, axis=2), x
        for i in range(self.num_layers):
            x = x0 * jnp.expand_dims(h, axis=1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.reshape(batch_size, f0_dim*fin_dim, embed_dim)
            x = nn.relu(conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(jnp.sum(jnp.concatenate(xs, axis=1), axis=2))


class FlaxConv1D(nn.Module):
    """
    reference : https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/gpt2/modeling_flax_gpt2.html
    """
    features: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = None

    @nn.compact
    def __call__(self, inputs):
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param("kernel", jax.nn.initializers.normal(stddev=0.02), (self.features, inputs.shape[-1]))
        kernel = jnp.asarray(kernel.transpose(), self.dtype)
        y = lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())), precision=self.precision)
        if self.use_bias:
            bias = self.param("bias", jax.nn.initializers.zeros, (self.features,))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias
        return y