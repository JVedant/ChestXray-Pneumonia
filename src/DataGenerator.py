import numpy as np
import os
import tensorflow as tf
from config.config import TRAIN_DIR, TEST_DIR, BATCH_SIZE, IMAGE_SIZE
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def data_generator():
    train_datagen = ImageDataGenerator(rescale=1/255.0, width_shift_range=0.2, height_shift_range=0.2, rotation_range=40, zoom_range=0.2, shear_range=0.2, samplewise_center=True)
    test_datagen = ImageDataGenerator(rescale=1/255.0)

    # generating data from existing images
    train_datagenerator = train_datagen.flow_from_directory(TRAIN_DIR,  batch_size=BATCH_SIZE, shuffle=True, class_mode='binary',target_size=IMAGE_SIZE)
    test_datagenerator = test_datagen.flow_from_directory(TEST_DIR, batch_size=BATCH_SIZE, shuffle=True, class_mode='binary', target_size=IMAGE_SIZE)

    return train_datagenerator, test_datagenerator