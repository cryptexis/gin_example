import gin
import numpy as np
import torch.nn as nn


@gin.configurable
class Generator1(nn.Module):
    def __init__(self, latent_dim, img_shape, base_depth=128):
        super().__init__()
        self.img_shape = img_shape
        self.base_depth = base_depth
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, self.base_depth , normalize=False),
            *block(self.base_depth , self.base_depth*2),
            *block(self.base_depth*2, self.base_depth*4),
            *block(self.base_depth*4, self.base_depth*8),
            nn.Linear(self.base_depth*8, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

 
@gin.configurable
class Generator2(nn.Module):
    def __init__(self, latent_dim, img_shape, base_depth=128):
        super().__init__()
        self.img_shape = img_shape
        self.base_depth = base_depth
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, self.base_depth , normalize=False),
            *block(self.base_depth , self.base_depth*2),
            *block(self.base_depth*2, self.base_depth*4),
            *block(self.base_depth*4, self.base_depth*8),
            nn.Linear(self.base_depth*8, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img