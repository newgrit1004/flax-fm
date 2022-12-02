from flax import linen as nn
import jax
from flaxfm.layer import FeaturesLinearFlax
import numpy as np

class LogisticRegressionModelFlax(nn.Module):
    field_dims : np.ndarray

    def setup(self):
        self.linear = FeaturesLinearFlax(self.field_dims)

    @nn.compact
    def __call__(self, x):
        return jax.nn.sigmoid(self.linear(x).squeeze(1))