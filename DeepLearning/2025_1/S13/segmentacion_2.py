#importar librerias
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight

#Dataset (Oxford Pets con segmentaciones)
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
train = dataset['train']
test = dataset['test']

def preprocess(sample):
    image = tf.image.resize(sample['image'], (128, 128))
    mask = tf.image.resize(sample['segmentation_mask'], (128, 128))
    mask = tf.cast(mask, tf.uint8) - 1  # Quitar clase de fondo extra
    return image / 255.0, mask

train = train.map(preprocess).batch(16).prefetch(tf.data.AUTOTUNE)
test = test.map(preprocess).batch(16).prefetch(tf.data.AUTOTUNE)

#unet
def unet_model(output_channels):
    inputs = tf.keras.Input(shape=(128, 128, 3))
    # Downsampling
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    # Bottleneck
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    # Upsampling
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    outputs = layers.Conv2D(output_channels, 1, activation='softmax')(x)
    return models.Model(inputs, outputs)
