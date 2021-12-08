import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from config import Config as c

ham_dataset_dir = 'ham'
batch_size = 48
seed = 42

train_ds = keras.preprocessing.image_dataset_from_directory(ham_dataset_dir, validation_split=0.2, color_mode='rgb',
                                                            labels='inferred', shuffle=True, subset='training', image_size=(256, 256),
                                                            batch_size=batch_size, seed=seed)

val_ds = keras.preprocessing.image_dataset_from_directory(ham_dataset_dir, validation_split=0.2, color_mode='rgb',
                                                          labels='inferred', shuffle=True, subset='validation', image_size=(256, 256),
                                                          batch_size=batch_size, seed=seed)

classes = train_ds.class_names

train_ds_single_batch = keras.preprocessing.image_dataset_from_directory(ham_dataset_dir, validation_split=0.2, color_mode='rgb',
                                                                         labels='inferred', shuffle=True, subset='training', image_size=(256, 256),
                                                                         batch_size=1, seed=seed)

y = np.array([label.numpy()[0] for _, label in train_ds_single_batch])
class_weights_list = compute_class_weight('balanced', classes=np.unique(y), y=y)
print(class_weights_list)
class_weights = {}

for i in range(len(class_weights_list)):
    class_weights[i] = class_weights_list[i]

rescale = keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 255)

train_ds = train_ds.map(lambda x, y: (rescale(x), y))
val_ds = val_ds.map(lambda x, y: (rescale(x), y))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

model = Sequential()
for conv_filter in c.filters:
    model.add(keras.layers.Conv2D(conv_filter, 3, strides=2, padding='same', input_shape=c.input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(c.num_classes, activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=50, class_weight=class_weights)
