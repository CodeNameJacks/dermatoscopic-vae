import argparse
from operator import itemgetter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_weights', type=str, help="Checkpoint to load the weights from. ex: checkpoints/vae/vae.data-00000-of-00001 then "
                                                        "input checkpoints/vae/vae")
    parser.add_argument('-hdd', '--ham_dataset_dir', type=str, required=True, help='Dataset to test VAE on (only 1 class) ex: ham/mel')
    parser.add_argument('-sd', '--save_dir', type=str, default='gen_data', help='Directory to store the generated data')
    parser.add_argument('-e', '--epsilon', type=float, default=0.2, help='Epsilon value for generating noise on latent space from standard normal')
    parser.add_argument('-nm', '--num_images', type=int, default=800, help='Number of images to reconstruct')
    args = vars(parser.parse_args())

from pathlib import Path
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.image import imsave

from model import ConvolutionalVAE


def imshow(x):
    plt.clf()
    plt.figure()
    plt.imshow(x)
    plt.show()
    plt.close()


def main(ar):
    model_weights, ham_dataset_dir, save_dir, epsilon, num_images = itemgetter('model_weights', 'ham_dataset_dir', 'save_dir', 'epsilon',
                                                                               'num_images')(ar)
    os.makedirs(save_dir, exist_ok=True)

    seed = 42

    ds = keras.preprocessing.image_dataset_from_directory(ham_dataset_dir, color_mode='rgb',
                                                          labels=None, shuffle=True, image_size=(224, 224),
                                                          batch_size=1)

    rescale = keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 255)
    ds = ds.map(lambda x: rescale(x))
    ds = ds.prefetch(tf.data.AUTOTUNE)

    autoencoder = ConvolutionalVAE()
    autoencoder.build(input_shape=(None, 224, 224, 3))

    if Path(f'{model_weights}.index').is_file():
        autoencoder.load_weights(f'./{model_weights}')

    i = 0
    for image in ds.take(num_images):
        _, _, z_images = autoencoder.encoder(image)
        z_images = z_images.numpy()
        z_noise = np.random.standard_normal(z_images.shape)
        z_altered = z_images + epsilon * z_noise
        synthetic_image = autoencoder.decoder(z_altered).numpy()[0]
        imsave(f'{save_dir}/{i}.jpg', synthetic_image)
        i += 1


if __name__ == '__main__':
    main(args)
