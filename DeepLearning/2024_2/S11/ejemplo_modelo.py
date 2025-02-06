from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, BatchNormalization, Dense, Dropout

model = Sequential([
    # Primera capa convolucional
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),  # MaxPooling con tamaño 2x2

    # Segunda capa convolucional con stride y padding
    Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
    AveragePooling2D((2, 2)),  # AveragePooling con tamaño 2x2
    BatchNormalization(),  # Normalización por lotes

    # Tercera capa convolucional
    Conv2D(128, (3, 3), activation='relu'),
    GlobalMaxPooling2D(),  # Pooling global máximo
    Dropout(0.5),  # Dropout para reducir el sobreajuste

    # Capa convolucional adicional con GlobalAveragePooling
    Conv2D(128, (3, 3), activation='relu'),
    GlobalAveragePooling2D(),  # Pooling global promedio

    # Capa totalmente conectada
    Dense(128, activation='relu'),
    Dropout(0.5),

    # Capa de salida para clasificación de 10 clases
    Dense(10, activation='softmax')
])