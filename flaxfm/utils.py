import flax
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxlib.xla_extension import DeviceArray
from functools import partial
from typing import Sequence

class Sequential(nn.Module):
    layers: Sequence[nn.Module]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

@flax.struct.dataclass
class Config:
    seed : int = 42
    epochs : int = 10
    learning_rate :float = 0.001
    batch_size : int = 256
    pass

config = Config()


@partial(jax.jit, static_argnums=2)
def slicing(x:DeviceArray, index:int, axis:int):
    return jnp.asarray(x).take(index, axis)