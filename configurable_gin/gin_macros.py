import gin


@gin.configurable
def latent_dim(value):
    return value