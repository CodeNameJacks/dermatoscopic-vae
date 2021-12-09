import argparse
from operator import itemgetter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_weights', type=str, help="Checkpoint to load the weights from. ex: checkpoints/vae/vae.data-00000-of-00001 then "
                                                        "input checkpoints/vae/vae")
    parser.add_argument('-hdd', '--ham_dataset_dir', type=str, required=True, help='Dataset to test VAE on (only 1 class) ex: ham/mel')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='Number of images to reconstruct')
    args = vars(parser.parse_args())

import tensorflow as tf
import tensorflow.keras as keras
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from model import ConvolutionalVAE


def main(args):
    model_weights, ham_dataset_dir, batch_size = itemgetter('model_weights', 'ham_dataset_dir', 'batch_size')(args)

    seed = 42

    val_ds = keras.preprocessing.image_dataset_from_directory(ham_dataset_dir, validation_split=0.2, color_mode='rgb',
                                                              labels=None, shuffle=True, subset='validation', image_size=(224, 224),
                                                              batch_size=batch_size, seed=seed)

    rescale = keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 255)
    val_ds = val_ds.map(lambda x: rescale(x))
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    autoencoder = ConvolutionalVAE()
    autoencoder.build(input_shape=(None, 224, 224, 3))

    if Path(f'{model_weights}.index').is_file():
        autoencoder.load_weights(f'./{model_weights}')

    for images in val_ds.take(1):
        plt.figure()
        reconstructions = autoencoder(images).numpy()
        f, axarr = plt.subplots(3, len(images), figsize=(2 * len(images), 3))
        # First display original
        for i, image in enumerate(images):
            axarr[0, i].imshow(image)
        # Then display reconstructions of original
        for i, reconstructed_image in enumerate(reconstructions):
            axarr[1, i].imshow(reconstructed_image)
        # Finally display images generated from 100-dimension vector z ~ N(0,1) * eps + z_original, where eps is 0.25
        _, _, z_images = autoencoder.encoder(images)
        z_images = z_images.numpy()
        z_noise = np.random.standard_normal(z_images.shape)
        z_altered = z_images + 0.3 * z_noise
        sampled_images = autoencoder.decoder(z_altered).numpy()
        for i, sampled_image in enumerate(sampled_images):
            axarr[2, i].imshow(sampled_image)

        plt.show()


if __name__ == '__main__':
    main(args)
