import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Cargar el dataset de Diabetes
data = load_diabetes()
X = data.data  # Características (10 variables)
y = data.target  # Variable objetivo (progresión de la diabetes)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar las características (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear el modelo de red neuronal multicapa (MLP) para regresión
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Capa oculta con 64 neuronas
    Dense(64, activation='relu'),  # Otra capa oculta con 64 neuronas
    Dense(1)  # Capa de salida para regresión (1 valor continuo)
])

# Compilar el modelo
model.compile(optimizer=Adam(),
              loss='mean_squared_error')  # Pérdida para problemas de regresión

# Entrenar el modelo
history = model.fit(X_train, y_train,
                    epochs=100,  # Entrenar durante 100 épocas
                    batch_size=16,
                    validation_data=(X_test, y_test),
                    verbose=1)

# Evaluar el modelo en el conjunto de prueba
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Pérdida en el conjunto de prueba (MSE): {test_loss:.2f}")

# Graficar la pérdida de entrenamiento/validación
epochs = range(1, len(history.history['loss']) + 1)

plt.figure(figsize=(10, 5))

# Gráfico de la pérdida
plt.plot(epochs, history.history['loss'], label='Pérdida Entrenamiento')
plt.plot(epochs, history.history['val_loss'], label='Pérdida Validación')
plt.title('Pérdida durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (MSE)')
plt.legend()
plt.show()
