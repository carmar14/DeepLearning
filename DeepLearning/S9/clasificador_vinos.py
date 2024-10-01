import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Cargar el dataset Wine
data = load_wine()
X = data.data
y = data.target

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar las características (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encoding de las etiquetas
y_train_ohe = to_categorical(y_train, num_classes=3)
y_test_ohe = to_categorical(y_test, num_classes=3)

# Crear el modelo de red neuronal multicapa (MLP)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Capa oculta con 64 neuronas
    Dense(64, activation='relu'),  # Otra capa oculta con 64 neuronas
    Dense(3, activation='softmax')  # Capa de salida para clasificación multiclase (3 clases)
])

# Compilar el modelo
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train_ohe,
                    epochs=50,
                    batch_size=16,
                    validation_data=(X_test, y_test_ohe),
                    verbose=1)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(X_test, y_test_ohe, verbose=0)
print(f"Exactitud en el conjunto de prueba: {test_acc:.2f}")

# Graficar la precisión y pérdida de entrenamiento/validación
epochs = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(12, 5))

# Gráfico de la precisión
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(epochs, history.history['val_accuracy'], label='Precisión Validación')
plt.title('Precisión durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Gráfico de la pérdida
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], label='Pérdida Entrenamiento')
plt.plot(epochs, history.history['val_loss'], label='Pérdida Validación')
plt.title('Pérdida durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()
