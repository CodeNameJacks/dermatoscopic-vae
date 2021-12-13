class Config:
    # VAE architecture
    kernels = 4
    strides = 2
    latent_dim = 128
    filters = [32, 64, 128, 256, 512]
    input_shape = (224, 224, 3)
    last_conv_dim = int(input_shape[0] / (2 ** len(filters)))
    b_norm = 3
    # Batch Norm
    epsilon = 1e-5
    # Spatial Classifier
    num_classes = 10

