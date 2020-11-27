import numpy as np
import os
import tensorflow as tf
from config.config import TRAIN_DIR, TEST_DIR, BATCH_SIZE, EPOCHS, MODEL, LOGS
from model import build_model, vgg_16, vgg_19, resnet_50
from time import time
from DataGenerator import data_generator

from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from kerastuner.tuners import RandomSearch
from tensorflow.keras.preprocessing.image import ImageDataGenerator


'''def train_CustomModel():

    # generating data from existing images
    train_datagenerator, test_datagenerator = data_generator()

    # getting the model and callback from model.py
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.001)

    tuner = RandomSearch(project_name=os.path.join(LOGS, 'trial_2/custom'), max_trials=3, executions_per_trial=5, hypermodel=build_model, objective='val_accuracy')
    tuner.search(train_datagenerator, epochs=10, callbacks=[lr_reduction], validation_data=test_datagenerator)
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

    model = tuner.hypermodel.build(best_hps)
    model.fit_generator(train_datagenerator, epochs = EPOCHS, validation_data =test_datagenerator)

    return model'''


def train_vgg16():

    # generating data from existing images
    train_datagenerator, test_datagenerator = data_generator()

    # getting the model and callback from model.py
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-5)

    tuner = RandomSearch(project_name=os.path.join(LOGS, 'trial_2/vgg_16'), max_trials=3, executions_per_trial=5, hypermodel=vgg_16, objective='val_accuracy')
    tuner.search(train_datagenerator, epochs=10, callbacks=[lr_reduction], validation_data=test_datagenerator)
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

    model = tuner.hypermodel.build(best_hps)
    model.fit_generator(train_datagenerator, epochs = EPOCHS, validation_data =test_datagenerator)

    return model


def train_vgg19():

    # generating data from existing images
    train_datagenerator, test_datagenerator = data_generator()

    # getting the model and callback from model.py
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-5)

    tuner = RandomSearch(project_name=os.path.join(LOGS, 'trial_2/vgg_19'), max_trials=3, executions_per_trial=5, hypermodel=vgg_16, objective='val_accuracy')
    tuner.search(train_datagenerator, epochs=10, callbacks=[lr_reduction], validation_data=test_datagenerator)
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

    model = tuner.hypermodel.build(best_hps)
    model.fit_generator(train_datagenerator, epochs = EPOCHS, validation_data =test_datagenerator)

    return model


def train_resnet50():

    # generating data from existing images
    train_datagenerator, test_datagenerator = data_generator()

    # getting the model and callback from model.py
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-5)

    tuner = RandomSearch(project_name=os.path.join(LOGS, 'trial_2/resnet_50'), max_trials=3, executions_per_trial=5, hypermodel=vgg_16, objective='val_accuracy')
    tuner.search(train_datagenerator, epochs=10, callbacks=[lr_reduction], validation_data=test_datagenerator)
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

    model = tuner.hypermodel.build(best_hps)
    model.fit_generator(train_datagenerator, epochs = EPOCHS, validation_data =test_datagenerator)

    return model


if __name__ == '__main__':

#    custom_model = train_CustomModel()
    vgg_16_model = train_vgg16()
    vgg_19_model = train_vgg19()
    resnet_50_model = train_resnet50()

    ''' save_model(model=custom_model,
        filepath=os.path.join(MODEL + '/h5', 'custom_model.h5'),
        save_format='h5')'''

    save_model(model=vgg_16_model,
        filepath=os.path.join(MODEL + '/h5', 'vgg_16_model.h5'),
        save_format='h5')

    save_model(model=vgg_19_model,
        filepath=os.path.join(MODEL + '/h5', 'vgg_19_model.h5'),
        save_format='h5')
    
    save_model(model=resnet_50_model,
        filepath=os.path.join(MODEL + '/h5', 'resnet_50_model.h5'),
        save_format='h5')