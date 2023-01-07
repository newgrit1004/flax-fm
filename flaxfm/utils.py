import flax
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxlib.xla_extension import DeviceArray
from functools import partial
from typing import Sequence, Callable
import time

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


class time_measure:
    """실행하는 함수의 정보 및 측정시간을 알 수 있는 클래스 데코레이터.
    #TODO : print 대신 logging 모듈로 수정
    """
    def __init__(self, func:Callable):
        self.start_time = time.time()
        self.func = func
    def __call__(self, *args, **kwargs) :
        print(f"{self.func.__name__}{args} started at {time.strftime('%X')}")
        self.func(*args, **kwargs)
        finish_time = time.time()
        print(f"{self.func.__name__}{args} finish at {time.strftime('%X')}, total:{finish_time-self.start_time} sec(s)")