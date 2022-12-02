from flaxfm.layer import FeaturesLinearFlax,FieldAwareFactorizationMachineFlax
from flax import linen as nn
import numpy as np


class FieldAwareFactorizationMachineModelFlax(nn.Module):
    field_dims : np.ndarray
    embed_dim : int = 16
    output_dim : int = 1

    def setup(self):
        self.linear = FeaturesLinearFlax(self.field_dims)
        self.ffm = FieldAwareFactorizationMachineFlax(self.field_dims, self.embed_dim)


    @nn.compact
    def __call__(self, x):
        ffm_term = np.sum(np.sum(self.ffm(x), axis=1), axis=1, keepdims=True)
        x = self.linear(x) + ffm_term
        return nn.sigmoid(x.squeeze(1))