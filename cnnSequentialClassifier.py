import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


#defined VAE architecture paramteres
kernels = 2
strides = 2
latent_dim = 28
filters = [32, 64, 128, 256, 512]
input_shape = (224, 224, 3)
last_conv_dim = int(input_shape[0] / (2 ** len(filters)))
b_norm = 3
#Batch Norm
epsilon = 1e-5
#spatial Classifier
num_classes = 4
#dataset
ham_dataset_dir = 'ham'
batch_size = 48
seed = 42
#define early stopping paramters
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)

#load and split datatset
train_ds = keras.preprocessing.image_dataset_from_directory(ham_dataset_dir, validation_split=0.3, color_mode='rgb',
                                                          labels='inferred', shuffle=True, subset='validation', image_size=(224, 224),
                                                          batch_size=batch_size, seed=seed)

val_ds = keras.preprocessing.image_dataset_from_directory(ham_dataset_dir, validation_split=0.3, color_mode='rgb',
                                                          labels='inferred', shuffle=True, subset='validation', image_size=(224, 224),
                                                          batch_size=batch_size, seed=seed)

classes = train_ds.class_names

train_ds_single_batch = keras.preprocessing.image_dataset_from_directory(ham_dataset_dir, validation_split=0.3, color_mode='rgb',
                                                                         labels='inferred', shuffle=True, subset='training', image_size=(224, 224),
                                                                         batch_size=1, seed=seed)

#define labels and class weights
y = np.array([label.numpy()[0] for _, label in train_ds_single_batch])
class_weights_list = compute_class_weight('balanced', classes=np.unique(y), y=y)
print(class_weights_list)
class_weights = {}

for i in range(len(class_weights_list)):
    class_weights[i] = class_weights_list[i]

#scale immages
rescale = keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 255)
train_ds = train_ds.map(lambda x, y: (rescale(x), y))
val_ds = val_ds.map(lambda x, y: (rescale(x), y))

#autotune dataset
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

#define the neural network
model = Sequential()
for conv_filter in filters:
    model.add(keras.layers.Conv2D(conv_filter, 3, strides=2, padding='same', input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5), activation='relu'))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

#print summary of the model
model.summary()

#optimize the model with using Adam
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

#fit the training and validation data to the model
history = model.fit(train_ds, validation_data=val_ds, epochs=50, class_weight=class_weights)
