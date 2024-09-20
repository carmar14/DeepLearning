#Nombres: Valentina Beca, Brayan Zamora y Vivian Echeverri.

#importar librerias
import numpy as np
import matplotlib.pyplot as plt

#Definir las funciones de activación y entrenamiento
# Función de activación (lineal)
def linear_function(x):
    return x

# Función para hacer predicciones(output)
def adaline_predict(X, weights):
    return linear_function(np.dot(X, weights[1:]) + 0*weights[0])

# Algoritmo de la red ADALINE
def adaline_train(X, y, learning_rate, epochs):
    # Inicializar los pesos (uno más para el bias)
    weights = np.random.rand(X.shape[1] + 1)
    #vector de error
    errors = []

    # Entrenamiento
    for _ in range(epochs):
        total_error = 0
        for xi, target in zip(X, y):
            # Calcular la salida (predicción)
            output = adaline_predict(xi,weights)
            # Calcular error
            error = (target - output)**2
            total_error += abs(error)
            # Actualizar los pesos
            update = 2*learning_rate * (target - output)
            weights[1:] += update * xi
            weights[0] += update
        errors.append(total_error)
    return weights,errors

# Preparar los datos de entrada y salida
n_samples = 5000
t = np.linspace(0,12,n_samples)
#np.random.normal(0, 0.4, n_samples) #0.4*np.sin(24*t)
x1 = np.sin(t)
x2 = 0.4*np.sin(24*t)
x3 = 0.2 * np.sin(20*t)
y = np.array([x1,x2,x3])
X = x1 + x2 + x3

# Salida esperada: señal sin ruido
for i in range(len(y)):
    plt.plot(t,y[i], label="x"+str(i+1)+" sin ruido")
plt.plot(t,X, label="Señal con ruido")
plt.legend()
plt.title("Señales originales y señal con ruido")
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud de la señal")
plt.grid()
plt.show()

# Crear las entradas y la salida para ADALINE
delay = 15
noisy_signal = np.array([X[i:i+delay] for i in range(n_samples-delay)])
print(noisy_signal.shape)

trainded_weights = []
globalErrors = []
for output in y:
    d = output[delay:]
    # Entrenar el perceptrón
    weights, errors = adaline_train(noisy_signal, d, 0.01, 200)
    trainded_weights.append(weights)
    globalErrors.append(errors)

# Predecir las señales filtradas con los pesos entrenados
filtered_signals = []
for i in range(len(y)):
    predict = ([adaline_predict(xi, trainded_weights[i]) for xi in noisy_signal])
    filtered_signals.append(predict)

#Graficar los errores
plt.figure()
for i in range(len(globalErrors)):
    plt.plot(globalErrors[i], label="Error Red Para x"+str(i+1))
plt.xlabel('Época')
plt.ylabel('Error Red Por Xi')
plt.title('Error de la Red')
plt.grid(True)
plt.legend()
plt.show()

# Graficar las señales originales y filtradas
plt.figure()
for i in range(len(y)):
    plt.plot(t[delay:], y[i][delay:], label="x"+str(i+1)+" original")
    plt.plot(t[delay:], filtered_signals[i],  label="x"+str(i+1)+" filtrada")
    plt.grid()
    plt.xlabel("Tiempo(s)")
    plt.ylabel("Amplitud de la señal")
    plt.legend()
    plt.title("Señal x"+str(i+1)+" original y señal x"+str(i+1)+" filtrada")
    plt.show()