import argparse
import multiprocessing
from operator import itemgetter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ham_dataset_dir', type=str,
                        help='Path to ham dataset\'s category ex: `ham/bkl`.')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size for the model (default - 32).')
    parser.add_argument('-nw', '--num_workers', type=int, default=multiprocessing.cpu_count(),
                        help='Maximum number of processes to spin up when using process-based threading (default - number of cores [multiprocessing.cpu_count()]).')
    parser.add_argument('-c', '--cache', dest='cache', action='store_true', help='Caching of training data is enabled.')
    parser.add_argument('-nc', '--no_cache', dest='cache', action='store_false',
                        help='Caching of training data is disabled (default).')
    parser.add_argument('-cf', '--cache_file', nargs='?', const='', default='',
                        help='File location for where to cache. Ex: /tmp/cache. If caching is enabled but directory is not provided, will cache in memory (default).')
    parser.set_defaults(cache=True)
    args = vars(parser.parse_args())

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from sklearn.model_selection import train_test_split
from clr_callback import CyclicLR
from model import ConvolutionalVAE

seed = 42


def imshow(x):
    plt.clf()
    plt.figure()
    plt.imshow(x)
    plt.show()
    plt.close()


def plot_latent_space(vae, n=6, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 256
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n, 3))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size, 3)
            figure[
            i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure)
    plt.show()


def initialize_training_env():
    # Set initial seed
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # SET MEMORY GROWTH
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    # policy = keras.mixed_precision.Policy('mixed_float16')
    # keras.mixed_precision.set_global_policy(policy)


def plot_history_metric(metrics, history: keras.callbacks.History, clr=None, savefiles=None):
    for metric in metrics:
        val = f'val_{metric}'
        plt.clf()
        plt.figure()
        plt.plot(history.history[metric])
        if val in history.history.keys():
            plt.plot(history.history[val])
        plt.title(f'{metric.capitalize()}')
        plt.ylabel(f'{metric.capitalize()}')
        plt.xlabel('Epoch')
        if val in history.history.keys():
            plt.legend(['train', 'val'], loc='upper left')
        if not savefiles:
            plt.show()
    if clr is not None:
        plt.clf()
        plt.figure()
        plt.plot(clr.history['lr'])
        plt.title('Learning Rate')
        plt.ylabel('LR')
        plt.xlabel('Iteration')
        if not savefiles:
            plt.show()


def main(args):
    ham_dataset_dir, batch_size, cache, cache_file, num_workers = itemgetter('ham_dataset_dir',
                                                                                 'batch_size', 'cache', 'cache_file',
                                                                                 'num_workers')(args)
    print(f'Ham Directory: {ham_dataset_dir}')
    # Setup env
    initialize_training_env()

    if cache and len(cache_file.strip()) != 0:
        import os
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    checkpoint_path = 'training/cp{epoch:02d}-{val_loss:.2f}.ckpt'
    # Get speech commands

    train_ds = keras.preprocessing.image_dataset_from_directory(ham_dataset_dir, validation_split=0.2, color_mode='rgb',
                                                                labels=None, shuffle=False, subset='training', image_size=(256, 256),
                                                                batch_size=batch_size)

    val_ds = keras.preprocessing.image_dataset_from_directory(ham_dataset_dir, validation_split=0.2, color_mode='rgb',
                                                              labels=None, shuffle=False, subset='validation', image_size=(256, 256),
                                                              batch_size=batch_size)
    rescale = keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 255)
    train_ds = train_ds.map(lambda x: rescale(x))
    val_ds = val_ds.map(lambda x: rescale(x))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    for image in train_ds.take(1):
        image = image[0]
        input_shape = image.shape
        print('Input shape:', input_shape)

    strategy = tf.distribute.MirroredStrategy()
    clr = CyclicLR(base_lr=0.0005, max_lr=0.005, step_size=150., mode='triangular2')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)

    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    if strategy.num_replicas_in_sync > 1:
        with strategy.scope():
            autoencoder = ConvolutionalVAE()
            autoencoder.compile(optimizer=keras.optimizers.Adam())
    else:
        autoencoder = ConvolutionalVAE()
        autoencoder.compile(optimizer=keras.optimizers.Adam(), run_eagerly=False)
    history = autoencoder.fit(train_ds, validation_data=val_ds, epochs=100, workers=num_workers,
                              max_queue_size=30, callbacks=[early_stopping])
    plot_history_metric(['loss', 'reconstruction_loss', 'kl_loss'], history)
    for image in val_ds.take(1):
        image = image[0]
        imshow(image.numpy())
        reconstruction = autoencoder(image[tf.newaxis, ...])
        imshow(reconstruction[0].numpy())


if __name__ == "__main__":
    main(args)
