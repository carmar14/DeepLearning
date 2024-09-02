import numpy as np
import matplotlib.pyplot as plt


class AdalineMultiOutput:
    def __init__(self, input_size, output_size, learning_rate=0.001, epochs=1000):
        """
        Inicializa la red ADALINE con múltiples salidas.

        Parameters:
        - input_size: Número de entradas (sin incluir el bias).
        - output_size: Número de salidas (señales a recuperar).
        - learning_rate: Tasa de aprendizaje para la actualización de pesos.
        - epochs: Número de iteraciones sobre el conjunto de entrenamiento.
        """
        # Inicializar pesos: una fila por cada salida, incluyendo el bias
        self.weights = np.zeros((output_size, input_size + 1))
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, x):
        """
        Realiza una predicción para una sola muestra de entrada.

        Parameters:
        - x: Vector de entrada.

        Returns:
        - Vector de salidas predichas.
        """
        x = np.insert(x, 0, 1)  # Insertar el término de bias
        return np.dot(self.weights, x)  # Producto punto para cada salida

    def train(self, X, D):
        """
        Entrena la red ADALINE utilizando el conjunto de datos proporcionado.

        Parameters:
        - X: Matriz de entradas (n_samples, input_size).
        - D: Matriz de salidas deseadas (n_samples, output_size).
        """
        for epoch in range(self.epochs):
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)  # Insertar el término de bias
                y = np.dot(self.weights, x_i)  # Predicción actual
                error = D[i] - y  # Error de predicción
                print("Error:", error.shape)
                print("entradas ", x_i.shape)
                # Actualizar los pesos para cada salida
                self.weights += self.learning_rate * np.outer(error, x_i)

            # Opcional: imprimir el error promedio cada cierto número de épocas
            if (epoch + 1) % 100 == 0:
                y_pred = self.predict_batch(X)
                mse = np.mean((D - y_pred) ** 2)
                print(f"Epoch {epoch + 1}/{self.epochs}, MSE: {mse:.4f}")

    def predict_batch(self, X):
        """
        Realiza predicciones para un conjunto de muestras de entrada.

        Parameters:
        - X: Matriz de entradas (n_samples, input_size).

        Returns:
        - Matriz de salidas predichas (n_samples, output_size).
        """
        X_bias = np.insert(X, 0, 1, axis=1)  # Insertar el término de bias en todas las muestras
        return np.dot(X_bias, self.weights.T)  # Producto punto para todas las salidas


# Parámetros de la señal
n_samples = 1000
t = np.linspace(0, 2 * np.pi, n_samples)

# Crear tres señales senoidales con diferentes frecuencias y amplitudes
signal1 = 1.0 * np.sin(2 * np.pi * 1 * t)  # Amplitud 1.0, Frecuencia 1 Hz
signal2 = 0.5 * np.sin(2 * np.pi * 3 * t)  # Amplitud 0.5, Frecuencia 3 Hz
signal3 = 0.2 * np.sin(2 * np.pi * 5 * t)  # Amplitud 0.2, Frecuencia 5 Hz

# Crear la señal compuesta
composite_signal = signal1 + signal2 + signal3

# Añadir ruido a la señal compuesta
#p.random.seed(42)  # Para reproducibilidad
#noise = np.random.normal(0, 0.1, n_samples)
noisy_composite_signal = composite_signal #+ noise

# Configuración de ADALINE
delay = 100  # Número de retrasos (taps) para la entrada
X = np.array([noisy_composite_signal[i:i + delay] for i in range(n_samples - delay)])
# Señales originales retrasadas para alinear con X
d1 = signal1[delay:]
d2 = signal2[delay:]
d3 = signal3[delay:]
D = np.stack((d1, d2, d3), axis=1)  # Matriz de salidas deseadas (n_samples - delay, 3)

# Crear y entrenar la red ADALINE multisalida
adaline = AdalineMultiOutput(input_size=delay, output_size=3, learning_rate=0.001, epochs=1000)
adaline.train(X, D)

# Recuperar las señales utilizando la red ADALINE entrenada
recovered_signals = adaline.predict_batch(X)  # Matriz de salidas recuperadas (n_samples - delay, 3)
recovered_signal1 = recovered_signals[:, 0]
recovered_signal2 = recovered_signals[:, 1]
recovered_signal3 = recovered_signals[:, 2]

# Graficar las señales originales, la señal compuesta con ruido y las señales recuperadas
plt.figure(figsize=(14, 12))

# Señal compuesta con ruido
plt.subplot(4, 1, 1)
plt.plot(t, noisy_composite_signal, label="Señal Compuesta con Ruido")
plt.title("Señal Compuesta con Ruido")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)

# Señal 1: Original vs Recuperada
plt.subplot(4, 1, 2)
plt.plot(t[delay:], recovered_signal1, label="Señal Recuperada 1 (ADALINE)", color="r")
plt.plot(t, signal1, '--', label="Señal Original 1", color="r")
plt.title("Recuperación de la Señal 1")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)

# Señal 2: Original vs Recuperada
plt.subplot(4, 1, 3)
plt.plot(t[delay:], recovered_signal2, label="Señal Recuperada 2 (ADALINE)", color="g")
plt.plot(t, signal2, '--', label="Señal Original 2", color="g")
plt.title("Recuperación de la Señal 2")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)

# Señal 3: Original vs Recuperada
plt.subplot(4, 1, 4)
plt.plot(t[delay:], recovered_signal3, label="Señal Recuperada 3 (ADALINE)", color="b")
plt.plot(t, signal3, '--', label="Señal Original 3", color="b")
plt.title("Recuperación de la Señal 3")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
