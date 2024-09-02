#importar librerias
import numpy as np
import matplotlib.pyplot as plt

#Definir las funciones de activación y entrenamiento
# Función de activación (escalón)
def linear_function(x):
    return x

# Función para hacer predicciones(output)
def adaline_predict(X, weights):
    Y = np.array([0, 0, 0])
    #print("tamñooo", y.shape[1])
    for i in range(y.shape[1]):
        Y[i] = linear_function(np.dot(X, weights[i, 1:]) + 0*weights[i, 0])
    return Y

# Algoritmo del Perceptrón
def adaline_train(X, y, learning_rate, epochs):
    # Inicializar los pesos (uno más para el bias)
    weights = np.random.rand(y.shape[1],X.shape[1] + 1)
    #print("pesos", weights.shape)
    #vector de error
    errors = []

    # Entrenamiento
    for _ in range(epochs):
        total_error = 0
        for xi, target in zip(X, y):
            # Calcular la salida (predicción)
            output = adaline_predict(xi,weights)
            # Calcular error absoluto
            error = (target - output)**2
            total_error += sum(error)
            print("error ", target - output)
            # Actualizar los pesos
            update = 2*learning_rate * (target - output)/len(noisy_signal)
            print("Actualizacion de pesos",update.shape)
            #print("entrada", xi.shape[0])
            #print("error", update.shape)
            weights[:,1:] += update.reshape(3,1) * xi.reshape(1,xi.shape[0])
            weights[:,0] += update
        errors.append(total_error)
    return weights,errors

# Preparar los datos de entrada y salida
# Datos de entrada para el filtro adaptativo
#señal con ruido
n_samples = 1000
t = np.linspace(0,2 * np.pi,n_samples)
A1 = 1
A2 = 0.5
A3 = 0.2
w1 = 2 * np.pi * 1
w2 = 2 * np.pi * 3
w3 = 2 * np.pi * 5
x1 = A1*np.sin(w1*t)
x2 = A2*np.sin(w2*t)
x3 = A3*np.sin(w3*t)
X = x1+x2+x3
#normalizacion
#X = 2*(X-min(X))/(max(X)-min(X))-1

plt.plot(t,X)
plt.grid()


# Salida esperada: señal sin ruido
delay = 150
y1 = x1[delay:]
y2 = x2[delay:]
y3 = x3[delay:]

#salidas normalizadas
#y1 = 2*(y1-min(y1))/(max(y1)-min(y1))-1
#y3 = 2*(y2-min(y2))/(max(y2)-min(y2))-1
#y3 = 2*(y3-min(y3))/(max(y3)-min(y3))-1
y= np.array([y1,y2,y3]).T
print("tamaño de la salida", y.shape)



# Crear las entradas y la salida para ADALINE

noisy_signal = np.array([X[i:i+delay] for i in range(n_samples-delay)])
print("tamaño de entradas",noisy_signal.shape)
print("tamaño",len(noisy_signal))
print("tamaño del tiempo",len(noisy_signal)-1)
d = y#y[delay:,:]
print("tamaño de la salida deseada", d.shape)


# Entrenar el perceptrón
weights,errors = adaline_train(noisy_signal, d, 0.001, 1000)
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
prediction=np.zeros((noisy_signal.shape[0],d.shape[1]))
print("tamaño ", prediction.shape)
i = 0
for xi in noisy_signal:
    #print("prediccion ", i, " ",adaline_predict(xi, weights))
    prediction[i,:] = adaline_predict(xi, weights)
    i +=1

print("tamaños y ", prediction.shape)
# Mostrar la gráfica
plt.figure()
plt.subplot(3,1,1)
plt.plot(t,X,'b')
plt.plot(t,x1,'--r')
plt.plot(t[delay:],prediction[:,0],'-.k')
plt.legend(["Entrada con ruido","Salida deseada sin ruido",'Salidad filtrada por la red'])
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud de la señal")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t,X,'b')
plt.plot(t,x2,'--r')
plt.plot(t[delay:],prediction[:,1],'-.k')
plt.legend(["Entrada con ruido","Salida deseada sin ruido",'Salidad filtrada por la red'])
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud de la señal")
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(t,X,'b')
plt.plot(t,x3,'--r')
plt.plot(t[delay:],prediction[:,2],'-.k')
plt.legend(["Entrada con ruido","Salida deseada sin ruido",'Salidad filtrada por la red'])
plt.xlabel("Tiempo(s)")
plt.ylabel("Amplitud de la señal")
plt.grid(True)

plt.show()

