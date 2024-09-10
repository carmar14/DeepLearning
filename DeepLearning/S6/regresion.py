import numpy as np
import matplotlib.pyplot as plt


# Función de activación lineal
def linear(x):
    return x


# Derivada de la función lineal
def linear_derivative(x):
    return np.ones_like(x)


# Entrenamiento de la red neuronal para regresión
def train_regression(X, y, epochs=10000, lr=0.01):
    np.random.seed(1)
    weights = np.random.rand(X.shape[1], 1)
    bias = np.random.rand(1)

    for epoch in range(epochs):
        output = linear(np.dot(X, weights) + bias)
        error = y - output
        adjustments = lr * error * linear_derivative(output)
        weights += np.dot(X.T, adjustments)
        bias += np.sum(adjustments)

    return weights, bias


# Generamos los datos
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X) + 0.5 * X

# Entrenamos la red
weights, bias = train_regression(X, y)
print(weights)
# Predicción
y_pred = linear(np.dot(X, weights) + bias)

# Visualización de resultados
plt.plot(X, y, label='Función Real')
plt.plot(X, y_pred, label='Predicción de la Red',color='r')
plt.legend()
plt.show()
