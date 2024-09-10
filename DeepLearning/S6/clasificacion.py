import numpy as np


# Función de activación escalón
def step_function(x):
    return np.where(x >= 0, 1, 0)


# Entrenamiento del perceptrón
def train(X, y, epochs=10000, lr=0.1):
    np.random.seed(1)
    weights = np.random.rand(X.shape[1], y.shape[1])
    bias = np.random.rand(1, y.shape[1])

    for epoch in range(epochs):
        # Calcular la salida
        output = step_function(np.dot(X, weights) + bias)

        # Calcular el error
        error = y - output

        # Actualizar los pesos
        weights += lr * np.dot(X.T, error)
        bias += lr * np.sum(error, axis=0)

    return weights, bias


# Representación binaria de las letras A a Z (simplificada a 10 bits por letra)
alphabet = {
    'A': [0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
    'B': [1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    'C': [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
    'D': [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    'E': [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    'F': [1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
    'G': [0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
    'H': [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
    'I': [0, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    'J': [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
    'K': [1, 1, 0, 1, 0, 0, 0, 1, 1, 0],
    'L': [1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    'M': [1, 1, 1, 0, 1, 0, 1, 0, 0, 0],
    'N': [1, 0, 1, 1, 1, 0, 0, 1, 1, 1],
    'O': [1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
    'P': [1, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    'Q': [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    'R': [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
    'S': [1, 0, 1, 0, 1, 1, 1, 1, 0, 0],
    'T': [1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
    'U': [0, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    'V': [0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
    'W': [1, 1, 1, 1, 0, 1, 0, 0, 1, 1],
    'X': [1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    'Y': [0, 1, 0, 1, 1, 1, 0, 0, 1, 0],
    'Z': [0, 1, 1, 0, 0, 1, 1, 1, 0, 1]
}

# Convertir las letras en su representación binaria
X = np.array(list(alphabet.values()))

# Crear las salidas esperadas (codificación binaria de A a Z)
y = np.array([
    [0, 0, 0, 0, 0],  # A
    [0, 0, 0, 0, 1],  # B
    [0, 0, 0, 1, 0],  # C
    [0, 0, 0, 1, 1],  # D
    [0, 0, 1, 0, 0],  # E
    [0, 0, 1, 0, 1],  # F
    [0, 0, 1, 1, 0],  # G
    [0, 0, 1, 1, 1],  # H
    [0, 1, 0, 0, 0],  # I
    [0, 1, 0, 0, 1],  # J
    [0, 1, 0, 1, 0],  # K
    [0, 1, 0, 1, 1],  # L
    [0, 1, 1, 0, 0],  # M
    [0, 1, 1, 0, 1],  # N
    [0, 1, 1, 1, 0],  # O
    [0, 1, 1, 1, 1],  # P
    [1, 0, 0, 0, 0],  # Q
    [1, 0, 0, 0, 1],  # R
    [1, 0, 0, 1, 0],  # S
    [1, 0, 0, 1, 1],  # T
    [1, 0, 1, 0, 0],  # U
    [1, 0, 1, 0, 1],  # V
    [1, 0, 1, 1, 0],  # W
    [1, 0, 1, 1, 1],  # X
    [1, 1, 0, 0, 0],  # Y
    [1, 1, 0, 0, 1]  # Z
])

# Entrenamos el perceptrón
weights, bias = train(X, y)

# Prueba de predicción con la letra A
letra_A = np.array([0, 1, 1, 1, 0, 1, 0, 0, 0, 1])
prediccion_A = step_function(np.dot(letra_A, weights) + bias)
print("Predicción para A:", prediccion_A)

# Prueba con otras letras...
