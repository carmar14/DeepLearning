#Camilo Roman Y Vivian Echeverri

import numpy as np
import matplotlib.pyplot as plt

#Función para añadir ruido a las letras
def add_noise(letters, noise_level):
    noisy_inputs = letters.copy()
    num_noisy_bits = int(noise_level * letters.size)
    indices = np.random.choice(letters.size, num_noisy_bits, replace=False)
    noisy_inputs.flat[indices] = 1 - noisy_inputs.flat[indices]  # Invertir bits
    print("Letras con Ruido"+str(noisy_inputs))
    return noisy_inputs

# Función para mostrar la letra en formato matriz 5x5
def mostrar_letra_en_matriz(bits):
    matriz = np.array(bits).reshape(5, 5)
    for fila in matriz:
        print(" ".join(str(bit) for bit in fila))
    print("\n")

#Función de activación de escalón
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Perceptrón: Definir una red con una sola capa
class Perceptron:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        # Inicializa pesos aleatorios (con el bias)
        self.weights = np.random.randn(output_size, input_size + 1)
        self.learning_rate = learning_rate
        self.errors = []  # Añadir esta línea para almacenar los errores

    def predict(self, X):
        # Añadir el bias al final de las entradas
        X_bias = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        # Calcular la salida usando la función de activación
        linear_output = np.dot(X_bias, self.weights.T)
        return step_function(linear_output)

    def train(self, X, y, epochs):
        # Añadir bias a las entradas
        X_bias = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        for _ in range(epochs):
            total_error = 0
            for i in range(X.shape[0]):
                # Realizar la predicción
                y_pred = self.predict(X[i].reshape(1, -1))
                # Calcular el error
                error = y[i] - y_pred[0]
                total_error += sum(abs(error))
                # Actualizar los pesos
                self.weights += self.learning_rate * error[:, np.newaxis] * X_bias[i]
            self.errors.append(total_error)  # Almacenar el error total de esta época


#Preparar los datos de entrada y salida Para el Alfabeto
letters = np.array([
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],#A
    [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0],#B
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],#C
    [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0],#D
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],#E
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],#F
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1],#G
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],#H
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],#I
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],#J
    [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],#K
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],#L
    [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],#M
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1],#N
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],#O
    [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],#P
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1],#Q
    [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],#R
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],#S
    [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],#T
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],#U
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],#V
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],#W
    [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],#X
    [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],#Y
    [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1]#Z
])

#Salida esperada para el Alfabeto
outputs = np.array([
    [0, 0, 0, 0, 0],#A
    [0, 0, 0, 0, 1],#B
    [0, 0, 0, 1, 0],#C
    [0, 0, 0, 1, 1],#D
    [0, 0, 1, 0, 0],#E
    [0, 0, 1, 0, 1],#F
    [0, 0, 1, 1, 0],#G
    [0, 0, 1, 1, 1],#H
    [0, 1, 0, 0, 0],#I
    [0, 1, 0, 0, 1],#J
    [0, 1, 0, 1, 0],#K
    [0, 1, 0, 1, 1],#L
    [0, 1, 1, 0, 0],#M
    [0, 1, 1, 0, 1],#N
    [0, 1, 1, 1, 0],#O
    [0, 1, 1, 1, 1],#P
    [1, 0, 0, 0, 0],#Q
    [1, 0, 0, 0, 1],#R
    [1, 0, 0, 1, 0],#S
    [1, 0, 0, 1, 1],#T
    [1, 0, 1, 0, 0],#U
    [1, 0, 1, 0, 1],#V
    [1, 0, 1, 1, 0],#W
    [1, 0, 1, 1, 1],#X
    [1, 1, 0, 0, 0],#Y
    [1, 1, 0, 0, 1]#Z
    ])

#Entrenar la red neuronal Perceptrón
input_size = letters.shape[1] # Entradas = Características
output_size = outputs.shape[1] # Salidas = Clases

#Entrenar la red neuronal Perceptrón Sin Ruido
perceptron = Perceptron(input_size, output_size, learning_rate=0.01)
perceptron.train(letters, outputs, epochs=1000)
print("Pesos Finales Sin Ruido: ", perceptron.weights)
print("Salida", outputs[:,0])

#Entrenar la red neuronal Perceptrón Con Ruido
noise_level=0.1
noisy_letters=add_noise(letters,noise_level)
perceptron_noisy =  Perceptron(input_size, output_size, learning_rate=0.01)
perceptron_noisy.train(noisy_letters, outputs, epochs=1000)
print("Pesos Finales Con Riudo: ", perceptron_noisy.weights)
print("Salida", outputs[:,0])


# Graficar el error global sin Ruido y Con Ruido
plt.figure(figsize=(10, 5))
plt.plot(perceptron.errors, label='Sin Ruido')
plt.plot(perceptron_noisy.errors, label='Con Ruido')
plt.xlabel('Épocas')
plt.ylabel('Error Global')
plt.title('Error Global durante el Entrenamiento')
plt.legend()
plt.grid(True)
plt.show()


# Mostrar las predicciones de todas las letras Sin Ruido y Con Ruido
for i in range(letters.shape[0]):
    letter = letters[i].reshape(1, -1)  # Seleccionar la letra
    prediction = perceptron.predict(letter)
    print(f"Entrada sin Ruido: {chr(65 + i)}\n Salida Esperada: {outputs[i]}\n Predicción: {prediction[0]}\n")
    print(mostrar_letra_en_matriz(letters[i]))

    noisy_letter = noisy_letters[i].reshape(1, -1)  # Seleccionar la letra con ruido
    noisy_prediction = perceptron_noisy.predict(noisy_letter)
    print(f"Entrada con Ruido: {chr(65 + i)}\n Salida Esperada: {outputs[i]}\n Predicción: {noisy_prediction[0]}\n")
