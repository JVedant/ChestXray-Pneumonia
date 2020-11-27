from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from kerastuner.engine.hyperparameters import HyperParameter, HyperParameters
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
#from tensorflow.keras.applications.resnet_v2 import ResNet152V2
#from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.applications.xception import Xception 
import os
from config.config import LEARNING_RATE, IMAGE_SIZE


def build_model(hp):
    model = Sequential()

    model.add(Conv2D(filters=hp.Int('input_layer_1', min_value=64, max_value=512, step=64), kernel_size=(3, 3), input_shape = IMAGE_SIZE + [3]))
    model.add(Activation(LeakyReLU(0.3)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=hp.Int('input_layer_2', min_value=64, max_value=256, step=32), kernel_size=(3, 3), input_shape = IMAGE_SIZE + [3]))
    model.add(Activation(LeakyReLU(0.3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    for i in range(hp.Int('n_layers', 1, 10)):

        model.add(Conv2D(filters=hp.Int(f'conv_{i}_1', min_value=64, max_value=512, step=64), kernel_size=(2,2)))
        model.add(Activation(LeakyReLU(0.3)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=hp.Int(f'conv_{i}_2', min_value=64, max_value=512, step=64), kernel_size=(2,2)))
        model.add(Activation(LeakyReLU(0.3)))
        model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(units=hp.Int('Fully_connected_1', min_value=256, max_value=1024, step=32)))
    model.add(Activation(tf.nn.relu))
    model.add(Dropout(0.15))
    model.add(Dense(1))
    model.add(Activation(tf.nn.sigmoid))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
        optimizer=Adam(learning_rate=LEARNING_RATE))

    return model


def vgg_16(hp):
    base_model = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)

    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    for i in range(hp.Int('n_connected_layers', 1, 10)):
        x = Dense(units=hp.Int(f'Dense_{i}', min_value=64, max_value=512, step=64), activation=LeakyReLU(hp.Choice('leaky_relu', values = [0.2, 0.3, 0.1])))(x)
        x = Dropout(0.2)(x)
    x = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs = base_model.input, outputs = x)

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer = Adam(lr=hp.Choice('learning_rate', values = LEARNING_RATE)))
    return model


def vgg_19(hp):
    base_model = VGG19(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)

    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    for i in range(hp.Int('n_connected_layers', 1, 10)):
        x = Dense(units=hp.Int(f'Dense_{i}', min_value=64, max_value=512, step=64), activation=LeakyReLU(hp.Choice('leaky_relu', values = [0.2, 0.3, 0.1])))(x)
        x = Dropout(0.2)(x)
    x = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs = base_model.input, outputs = x)

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer = Adam(lr=hp.Choice('learning_rate', values = LEARNING_RATE)))
    return model


def resnet_50(hp):
    base_model = ResNet50(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)

    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    for i in range(hp.Int('n_connected_layers', 1, 10)):
        x = Dense(units=hp.Int(f'Dense_{i}', min_value=64, max_value=512, step=64), activation=LeakyReLU(hp.Choice('leaky_relu', values = [0.2, 0.3, 0.1])))(x)
        x = Dropout(0.2)(x)
    x = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs = base_model.input, outputs = x)

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer = Adam(lr=hp.Choice('learning_rate', values = LEARNING_RATE)))
    return model