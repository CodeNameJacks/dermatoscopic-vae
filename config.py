class Config:
    # CNN
    kernels=4
    strides=2
    # VAE architecture
    latent_dim = 256
    filters = [32, 64, 128, 256, 256]
    input_shape = (256, 256, 3)
    last_conv_dim = int(input_shape[0] / (2 ** len(filters)))
    b_norm = 3
    # Batch Norm
    epsilon = 1e-5

