import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Definir las entradas y salidas para la compuerta XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Crear el modelo de red neuronal multicapa (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', max_iter=100, random_state=62)

# Entrenar el modelo
mlp.fit(X, y)

# Realizar predicciones
y_pred = mlp.predict(X)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y, y_pred)

# Mostrar los resultados
print("Predicciones:", y_pred)
print("Exactitud del modelo:", accuracy)

# Mostrar pesos y bias de la red
print("Pesos de la capa oculta:", mlp.coefs_[0])
print("Pesos de la capa de salida:", mlp.coefs_[1])
print("Bias de la capa oculta:", mlp.intercepts_[0])
print("Bias de la capa de salida:", mlp.intercepts_[1])

# Crear una malla de puntos para graficar el plano de separación
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predecir las clases para cada punto en la malla
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar el plano de separación
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100, cmap=plt.cm.coolwarm)

plt.title("Plano de separación generado por la MLP para el problema XOR")
plt.xlabel('Entrada 1')
plt.ylabel('Entrada 2')
plt.show()
