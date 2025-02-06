import numpy as np

class AdalineMultiOutput:
    def __init__(self, input_size, output_size, learning_rate=0.001, epochs=500):
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
        # vector de error
        self.errors = []

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
            total_error = 0
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)
                y = np.dot(self.weights, x_i)
                error = D[i] - y
                total_error += np.sum(error ** 2)
                self.weights += 2 * self.learning_rate * np.outer(error, x_i)

            mse = total_error / (len(X) * self.weights.shape[0])
            self.errors.append(mse)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, MSE: {mse:.5f}")

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