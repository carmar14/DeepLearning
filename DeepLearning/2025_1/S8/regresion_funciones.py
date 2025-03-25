import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_percentage_error

# Función que vamos a intentar aprender
def func(x1, x2):
    return x1*x1-x2*x2#np.sin(x1) + np.log(x2 + 1) + (x1 * x2)

# Generar un dataset con valores de x1 y x2
np.random.seed(42)
X1 = np.random.uniform(-5, 5, 1000)  # Variable x1 entre 0 y 10
X2 = np.random.uniform(-5, 5, 1000)  # Variable x2 entre 0 y 10
y = func(X1, X2)  # Valores reales de la función

# Crear el dataset con las variables X1 y X2
X = np.column_stack((X1, X2))

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar las características (scaling)
'''
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
'''

# Crear el modelo de red neuronal multicapa (MLP) para regresión
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Capa oculta con 64 neuronas
    Dense(64, activation='relu'),  # Otra capa oculta con 64 neuronas
    Dense(1)  # Capa de salida para regresión (1 valor continuo)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo en el conjunto de prueba
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Pérdida en el conjunto de prueba (MSE): {test_loss:.2f}")

# Calcular el RMSE (Raíz del Error Cuadrático Medio)
rmse = np.sqrt(test_loss)
print(f"Raíz del error cuadrático medio (RMSE): {rmse:.2f}")

# Calcular MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")

# Graficar los valores reales y predichos para la función
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='red', alpha=0.6, label='Valores Predichos')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', label='Valores Reales (Referencia)')
plt.title('Valores Reales vs Predichos')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.legend()
plt.show()

# Visualizar la superficie de la función y la superficie predicha
fig = plt.figure(figsize=(12, 6))
# Crear un meshgrid para graficar las superficies
x1_range = np.linspace(-10, 10, 100)
x2_range = np.linspace(1, 10, 100)
X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
y_real_surface = func(X1_mesh, X2_mesh)

# Graficar la función real
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X1_mesh, X2_mesh, y_real_surface, cmap='viridis')
ax.set_title('Superficie de la Función Real')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')

# Graficar la función predicha
X1_flat = X1_mesh.flatten()
X2_flat = X2_mesh.flatten()
'''
X_mesh_scaled = scaler.transform(np.column_stack((X1_flat, X2_flat)))
y_pred_surface = model.predict(X_mesh_scaled).reshape(X1_mesh.shape)
'''
y_pred_surface = model.predict(np.column_stack((X1_flat, X2_flat))).reshape(X1_mesh.shape)
ax.scatter(X1_mesh, X2_mesh, y_pred_surface, 'ro')
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X1_mesh, X2_mesh, y_pred_surface, cmap='plasma')
ax2.set_title('Superficie de la Función Predicha')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('f_pred(x1, x2)')

plt.tight_layout()
plt.show()



