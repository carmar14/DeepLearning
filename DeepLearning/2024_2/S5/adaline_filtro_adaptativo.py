#importar librerias
import numpy as np
import matplotlib.pyplot as plt

#Definir las funciones de activación y entrenamiento
# Función de activación (escalón)
def linear_function(x):
    return x

# Función para hacer predicciones(output)
def adaline_predict(X, weights):
    return linear_function(np.dot(X, weights[1:]) + 0*weights[0])

# Algoritmo del Perceptrón
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
            output = adaline_predict(xi,weights)#step_function(np.dot(xi, weights[1:]) + weights[0])
            # Calcular error absoluto
            error = (target - output)**2
            total_error += abs(error)
            # Actualizar los pesos
            update =learning_rate * (target - output)
            weights[1:] += update * xi
            weights[0] += update
        errors.append(total_error)
    return weights,errors

# Preparar los datos de entrada y salida
# Datos de entrada para el filtro adaptativo
#señal con ruido
n_samples = 5000
t = np.linspace(0,12,n_samples)
noise = 0.4*np.sin(24*t)#np.random.normal(0, 0.4, n_samples) #0.4*np.sin(24*t)
X = np.sin(t)+noise
plt.plot(t,X)
plt.grid()


# Salida esperada: señal sin ruido
y = np.sin(t)
plt.plot(t,y,'r')
plt.legend(["Entrada con ruido","Salida sin ruido"])
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud de la señal")


# Crear las entradas y la salida para ADALINE
delay = 15
noisy_signal = np.array([X[i:i+delay] for i in range(n_samples-delay)])
print(noisy_signal.shape)
d = y[delay:]


# Entrenar el perceptrón
weights,errors = adaline_train(noisy_signal, d, 0.01, 200)
print("Pesos entrenados:", weights)
print("Errores:", errors)

# Graficar el error global en cada época
plt.figure()
plt.plot(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Época')
plt.ylabel('Error Global')
plt.title('Error Global del Perceptrón en cada Época')
plt.grid(True)

#señal filtrada
prediction=np.zeros(noisy_signal.shape[0])
print("tamaño ", prediction.shape)
i = 0
for xi in noisy_signal:
    #print("prediccion ", i, " ",adaline_predict(xi, weights))
    prediction[i] = adaline_predict(xi, weights)
    i +=1

print("tamaños y ", prediction.shape)
# Mostrar la gráfica
plt.figure()
plt.grid(True)
plt.plot(t,X,'b')
plt.plot(t,y,'--r')
plt.plot(t[delay:],prediction,'-.k')
plt.legend(["Entrada con ruido","Salida sin ruido",'Salidad filtrada por la red'])
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud de la señal")
plt.show()
