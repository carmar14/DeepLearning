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

# Para simplificar, computamos las frecuencias globales de clases
def compute_class_distribution(dataset):
    total = []
    for _, mask in dataset.unbatch():
        flat_mask = tf.reshape(mask, [-1])
        total.extend(flat_mask.numpy())
    unique, counts = np.unique(total, return_counts=True)
    class_freq = dict(zip(unique, counts))
    return class_freq

class_freq = compute_class_distribution(train)
total_pixels = sum(class_freq.values())
class_weights = {k: total_pixels / (len(class_freq) * v) for k, v in class_freq.items()}
print("Class frequencies:", class_freq)
print("Class weights:", class_weights)

#metricas de evaluacion
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def pixel_accuracy(y_true, y_pred):
    y_pred_labels = tf.argmax(y_pred, axis=-1)
    correct = tf.equal(tf.cast(y_true, tf.int64), y_pred_labels)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

def mean_pixel_accuracy(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.reshape(y_pred, [-1])
    num_classes = tf.reduce_max(y_true) + 1
    mpa = []
    for c in range(num_classes):
        mask = tf.equal(y_true, c)
        correct = tf.equal(tf.boolean_mask(y_pred, mask), c)
        acc = tf.reduce_mean(tf.cast(correct, tf.float32))
        mpa.append(acc)
    return tf.reduce_mean(tf.stack(mpa))

def frequency_weighted_iou(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.reshape(y_pred, [-1])
    num_classes = tf.reduce_max(y_true) + 1
    total_pixels = tf.size(y_true)
    fwiou = 0
    for c in range(num_classes):
        true_c = tf.equal(y_true, c)
        pred_c = tf.equal(y_pred, c)
        inter = tf.reduce_sum(tf.cast(true_c & pred_c, tf.float32))
        union = tf.reduce_sum(tf.cast(true_c | pred_c, tf.float32))
        freq = tf.reduce_sum(tf.cast(true_c, tf.float32)) / tf.cast(total_pixels, tf.float32)
        fwiou += freq * (inter / (union + 1e-6))
    return fwiou

#train

model = unet_model(output_channels=3)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train, validation_data=test, epochs=10)

# Obtener una predicción sobre todo el test set
for images, masks in test.take(1):
    preds = model.predict(images)

    print("IoU:", iou_metric(masks, preds).numpy())
    print("Dice Coef:", dice_coef(masks, preds).numpy())
    print("Pixel Acc:", pixel_accuracy(masks, preds).numpy())
    print("Mean Pixel Acc:", mean_pixel_accuracy(masks, preds).numpy())
    print("Frequency Weighted IoU:", frequency_weighted_iou(masks, preds).numpy())
