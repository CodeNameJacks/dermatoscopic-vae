import argparse
import multiprocessing
from operator import itemgetter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('speech_commands_dir', type=str, help='Path to speech commands dataset\'s category ex: `speech_commands/cat`.')
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
import os
from os import path
from sklearn.model_selection import train_test_split

seed = 42


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_logvar) to sample z the vector encoding a patch"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        a = tf.exp(0.5 * z_log_var)
        # Casting only required when using mixed_precision
        # Without it, it results in a float16 and float32 type mismatch
        epsilon = tf.cast(tf.keras.backend.random_normal(shape=(batch, dim)), dtype=a.dtype)
        return z_mean + a * epsilon


class ConvolutionalVAE(keras.Model):
    def __init__(self, code_dim_size, target_image_dims, **kwargs):
        super(ConvolutionalVAE, self).__init__(**kwargs)
        self.code_dim_size = code_dim_size
        self.target_image_dims = target_image_dims

        encoder_inputs = keras.Input(shape=target_image_dims)  # (124, 128, 1)
        num_conv_layers = 2
        kernels = 3
        strides = 2
        x = keras.layers.Conv2D(filters=16, kernel_size=kernels, strides=strides, padding='same', activation='relu')(encoder_inputs)  # (62, 64, 32)
        x = keras.layers.Conv2D(filters=32, kernel_size=kernels, strides=strides, padding='same', activation='relu')(x)  # (31, 32, 64)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(16, activation='relu')(x)
        z_mean = keras.layers.Dense(code_dim_size, name='z_mean')(x)
        z_logvar = keras.layers.Dense(code_dim_size, name='z_logvar')(x)
        z = Sampling()([z_mean, z_logvar])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_logvar, z], name='encoder')
        self.encoder.summary()

        latent_input = keras.Input(shape=(code_dim_size,))
        decode_image_x_shape = int(target_image_dims[0] / (num_conv_layers * strides))
        decode_image_y_shape = int(target_image_dims[1] / (num_conv_layers * strides))

        x = keras.layers.Dense(decode_image_x_shape * decode_image_y_shape * 32, activation='relu')(latent_input)
        x = keras.layers.Reshape((decode_image_x_shape, decode_image_y_shape, 32))(x)
        x = keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        decoder_output = keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')(x)
        decoder_output = keras.layers.Activation('sigmoid', dtype='float32')(decoder_output)
        self.decoder = keras.Model(latent_input, decoder_output, name='decoder')
        self.decoder.summary()

        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.mse_tracker
        ]

    def call(self, inputs, training=None, mask=None):
        z_mean, z_logvar, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.binary_crossentropy(inputs, reconstruction)))
        kl_loss = -0.5 * (1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss))
        # Casting only required when using mixed_precision
        # Without it, it results in a float16 and float32 type mismatch
        total_loss = reconstruction_loss + tf.cast(kl_loss, dtype=reconstruction_loss.dtype)
        mse_loss = tf.reduce_mean(keras.losses.mean_squared_error(inputs, reconstruction))

        self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        self.add_metric(total_loss, name='total_loss', aggregation='mean')
        self.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')
        self.add_metric(mse_loss, name='mse', aggregation='mean')
        return reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_logvar, z = self.encoder(data[0])
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data[0], reconstruction), axis=(1, 2)))
            kl_loss = -0.5 * (1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            # Casting only required when using mixed_precision
            # Without it, it results in a float16 and float32 type mismatch
            total_loss = reconstruction_loss + tf.cast(kl_loss, dtype=reconstruction_loss.dtype)
            mse_loss = tf.reduce_mean(keras.losses.mean_squared_error(data[0], reconstruction), axis=(1, 2))

        gradient = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradient, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.mse_tracker.update_state(mse_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "mse": self.mse_tracker.result(),
        }


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
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]


def get_waveform_and_label(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    spectrogram = get_spectrogram(waveform)
    return spectrogram


def get_spectrogram(waveform):
    # zero-padding for an audio waveform with less than 16,000 samples
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([input_len] - tf.shape(waveform), dtype=tf.float32)
    # cast the waveform tensors to float32
    waveform = tf.cast(waveform, dtype=tf.float32)
    # concatenate the waveform with zero_padding, which ensures all
    # audio clips are of the same length.decode_audio
    equal_length = tf.concat([waveform, zero_padding], axis=0)
    # convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    # Reduce frames by 1 if it's odd (convolutional needs even inputs or else it gets messed up)
    if spectrogram.shape[-1] % 2 == 1:
        spectrogram = spectrogram[:, :-1]
    # obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)

    # add a channels dimension, so that the spectrogram can be used
    # as image-like input data with conv layers (which expect
    # shape (batch_size, height, width, channels)).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    # output_ds = files_ds.map(map_func=get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    output_ds = files_ds.map(lambda x: tf.py_function(get_waveform_and_label, [x], [tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
    return output_ds


def generate_and_save_images(model: ConvolutionalVAE, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + i)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


def main(args):
    speech_commands_dir, batch_size, cache, cache_file, num_workers = itemgetter('speech_commands_dir',
                                                                                 'batch_size', 'cache', 'cache_file',
                                                                                 'num_workers')(args)
    print(f'Speech Commands Directory: {speech_commands_dir}')
    # Setup env
    initialize_training_env()

    if cache and len(cache_file.strip()) != 0:
        import os
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    checkpoint_path = 'training/cp{epoch:02d}-{val_loss:.2f}.ckpt'
    # Get speech commands
    code_dim_size = 2
    # Get file paths of audio files
    filenames = tf.io.gfile.glob(speech_commands_dir + '/*')
    filenames = tf.random.shuffle(filenames).numpy()
    num_samples = len(filenames)

    print('Number of total examples:', num_samples)
    # Create train/val/test split 70-20-10
    train_files, val_files = train_test_split(filenames, test_size=0.3, random_state=seed)
    val_files, test_files = train_test_split(val_files, test_size=0.33, random_state=seed)
    print('Training set size:', len(train_files))
    print('Validation set size:', len(val_files))
    print('Test set size:', len(test_files))

    train_ds: tf.data.Dataset = preprocess_dataset(train_files).batch(batch_size)
    val_ds: tf.data.Dataset = preprocess_dataset(val_files).batch(batch_size)
    test_ds: tf.data.Dataset = preprocess_dataset(test_files)

    if cache:
        train_ds = train_ds.cache(cache_file)
        val_ds = val_ds.cache(cache_file)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    for spec, in test_ds.take(1):
        input_shape = spec.shape
    print('Input shape:', input_shape)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    if strategy.num_replicas_in_sync > 1:
        with strategy.scope():
            autoencoder = ConvolutionalVAE(code_dim_size=code_dim_size, target_image_dims=input_shape)
            autoencoder.compile(optimizer=keras.optimizers.Adam())
    else:
        autoencoder = ConvolutionalVAE(code_dim_size=code_dim_size, target_image_dims=input_shape)
        autoencoder.compile(optimizer=keras.optimizers.Adam(clipnorm=1.0))
    autoencoder.fit(train_ds, validation_data=val_ds, epochs=30, workers=num_workers, max_queue_size=30)


if __name__ == "__main__":
    main(args)
