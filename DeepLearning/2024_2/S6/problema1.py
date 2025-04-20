
#Importamos las librearías necesarias
import numpy as np
import matplotlib.pyplot as plt

#Definimos la función de activación. En este caso, la función escalón.
def funcionEscalon(x):
    if x >= 0:
        return 1
    else:
        return 0

#Función de predicción del perceptrón.
def prediccion(entrada, pesos):
    # Se calcula el producto punto de la entrada y los pesos, se suma el sesgo y se aplica la función de activación.
    return funcionEscalon(np.dot(entrada, pesos[1:]) + pesos[0])

#Función de entrenamiento del perceptrón.
def entrenamientoPerceptron(entrada, resultadoEsperado, tasaAprendizaje=0.1, epocas=100):
    # Se inicializan los pesos de manera aleatoria, usando una distribución uniforme entre 0 y 1.
    pesos = np.random.rand(entrada.shape[1] + 1)
    errores = []

# Se realiza el entrenamiento del perceptrón durante el número de épocas especificado.
    for _ in range(epocas):
        totalError = 0
        # Se recorren los datos de entrada y los resultados esperados.
        for xi, objetivo in zip(entrada, resultadoEsperado):
            # Se realiza la predicción y se calcula el error
            salida = prediccion(xi, pesos)
            error = objetivo - salida
            totalError += abs(error)
            actualizacion = tasaAprendizaje * (error)
            # Se actualizan los pesos y el bias del perceptrón.
            pesos[1:] += actualizacion * xi
            pesos[0] += actualizacion
        errores.append(totalError)
    return pesos, errores

#Datos de entrada de la red.
letras = np.array([
    # Letra A
    [
        0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1
    ],

    # Letra B
    [
        1, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 0
    ],

    # Letra C
    [
        0, 1, 1, 1, 1,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0,
        0, 1, 1, 1, 1
    ],

    # Letra D
    [
        1, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 0
    ],

    # Letra E
    [
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 0,
        1, 1, 1, 1, 0,
        1, 0, 0, 0, 0,
        1, 1, 1, 1, 1
    ],

    # Letra F
    [
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 0,
        1, 1, 1, 1, 0,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0
    ],

    # Letra G
    [
        0, 1, 1, 1, 1,
        1, 0, 0, 0, 0,
        1, 0, 1, 1, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 1
    ],

    # Letra H
    [
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1
    ],

    # Letra I
    [
        0, 1, 1, 1, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 1, 1, 1, 0
    ],

    # Letra J
    [
        0, 0, 0, 1, 1,
        0, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 0
    ],

    # Letra K
    [
        1, 0, 0, 0, 1,
        1, 0, 0, 1, 0,
        1, 1, 1, 0, 0,
        1, 0, 0, 1, 0,
        1, 0, 0, 0, 1
    ],

    # Letra L
    [
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0,
        1, 1, 1, 1, 1
    ],

    # Letra M
    [
        1, 0, 0, 0, 1,
        1, 1, 0, 1, 1,
        1, 0, 1, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1
    ],

    # Letra N
    [
        1, 0, 0, 0, 1,
        1, 1, 0, 0, 1,
        1, 0, 1, 0, 1,
        1, 0, 0, 1, 1,
        1, 0, 0, 0, 1
    ],

    # Letra O
    [
        0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 0
    ],

    # Letra P
    [
        1, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 0,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0
    ],

    # Letra Q
    [
        0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 1, 0, 1,
        0, 1, 1, 1, 1
    ],

    # Letra R
    [
        1, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 0,
        1, 0, 1, 0, 0,
        1, 0, 0, 1, 0
    ],

    # Letra S
    [
        0, 1, 1, 1, 1,
        1, 0, 0, 0, 0,
        0, 1, 1, 1, 0,
        0, 0, 0, 0, 1,
        1, 1, 1, 1, 0
    ],

    # Letra T
    [
        1, 1, 1, 1, 1,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0
    ],

    # Letra U
    [
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 0
    ],

    # Letra V
    [
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 0, 1, 0,
        0, 0, 1, 0, 0
    ],

    # Letra W
    [
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 1, 0, 1,
        1, 1, 0, 1, 1,
        1, 0, 0, 0, 1
    ],

    # Letra X
    [
        1, 0, 0, 0, 1,
        0, 1, 0, 1, 0,
        0, 0, 1, 0, 0,
        0, 1, 0, 1, 0,
        1, 0, 0, 0, 1
    ],

    # Letra Y
    [
        1, 0, 0, 0, 1,
        0, 1, 0, 1, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0
    ],

    # Letra Z
    [
        1, 1, 1, 1, 1,
        0, 0, 0, 1, 0,
        0, 0, 1, 0, 0,
        0, 1, 0, 0, 0,
        1, 1, 1, 1, 1
    ]
])

#Salida esperada de la red.
salidaBinaria = np.array([
    [0, 0, 0, 0, 1],  # A
    [0, 0, 0, 1, 0],  # B
    [0, 0, 0, 1, 1],  # C
    [0, 0, 1, 0, 0],  # D
    [0, 0, 1, 0, 1],  # E
    [0, 0, 1, 1, 0],  # F
    [0, 0, 1, 1, 1],  # G
    [0, 1, 0, 0, 0],  # H
    [0, 1, 0, 0, 1],  # I
    [0, 1, 0, 1, 0],  # J
    [0, 1, 0, 1, 1],  # K
    [0, 1, 1, 0, 0],  # L
    [0, 1, 1, 0, 1],  # M
    [0, 1, 1, 1, 0],  # N
    [0, 1, 1, 1, 1],  # O
    [1, 0, 0, 0, 0],  # P
    [1, 0, 0, 0, 1],  # Q
    [1, 0, 0, 1, 0],  # R
    [1, 0, 0, 1, 1],  # S
    [1, 0, 1, 0, 0],  # T
    [1, 0, 1, 0, 1],  # U
    [1, 0, 1, 1, 0],  # V
    [1, 0, 1, 1, 1],  # W
    [1, 1, 0, 0, 0],  # X
    [1, 1, 0, 0, 1],  # Y
    [1, 1, 0, 1, 0]   # Z
])

#Entrenamos el perceptrón con los datos de entrada y salida esperada.
pesosEntrenamiento = []
erroresEntrenamiento = []
tasa_aprendizaje = 0.1
epocas = 80
#Entrenamos una neurona por cada salida binaria esperada.
for i in range(salidaBinaria.shape[1]):
    pesos, errores = entrenamientoPerceptron(letras, salidaBinaria[:,i], tasa_aprendizaje, epocas)
    pesosEntrenamiento.append(pesos)
    erroresEntrenamiento.append(errores)


#Imprimimos los pesos por neurona
for i in range(len(pesosEntrenamiento)):
    print("Pesos Neurona ", i+1, ": ", pesosEntrenamiento[i])

#Función para predecir la letra en 5 bits a partir de una entrada ingresada.
def predecir_letra(x, pesosEntrenados):
    prediccionBinario = np.array([]).astype(int)
    for pesos in pesosEntrenados:
        prediccionBinario = np.append(prediccionBinario, prediccion(x, pesos))
    return prediccionBinario

#Pesos usados para realizar la conclusión en el artículo
'''
pesosEntrenamiento = [
    # Pesos Neurona 1
    [
        0.31904664, 0.11477362, 0.27444266, 0.14749348, -0.12468489, 0.74345121,
        1.92154867, 0.39293363, -0.72325231, 0.37525119, 1.74446745, -1.24749414,
        -0.00475849, -0.91400933, 0.46873471, -0.4454678, -1.1711133, 1.20487836,
        1.59875454, 0.32836433, -0.82154426, -0.62525307, -0.37156407, -0.48388643,
        0.17431909, -0.52395826
    ],
    # Pesos Neurona 2
    [
        0.08027201, -0.29626131, 0.01521048, -0.76436365, -0.39768899, -0.7554599,
        -1.16043219, 0.52940473, 0.94661166, 0.91310591, -0.47516395, -0.39755165,
        0.29413013, 0.83508385, -0.23494195, 1.33481149, 0.46219083, -0.47309773,
        -0.90246734, -0.31208267, -0.37542686, 0.48425734, 0.24582522, 0.19634419,
        0.24571106, 0.30887212
    ],
    # Pesos Neurona 3
    [
        -1.88475579, 1.11165384, 0.51501113, -0.24103871, -0.23580393, 0.30117427,
        0.63839599, 0.62799827, 1.13926381, -0.37397918, -1.21068368, 0.53828337,
        -1.23448734, -0.2212018, -0.02039346, 1.62405659, 0.45665893, -0.47333299,
        -0.74524875, 0.26411121, -0.48352118, 0.65382071, 0.1449208, 0.33099665,
        0.13781007, -0.73545111
    ],
    # Pesos Neurona 4
    [
        -1.8756616, -1.30630652, 0.49529159, -0.02581833, 0.81825294, 1.74008861,
        0.43347677, -1.07146955, -1.24412883, 0.08653381, -0.13039893, 0.5065217,
        -0.32610852, 0.98266279, -0.39206335, -0.72792733, -0.47546496, 0.74597933,
        0.18634144, 1.64599959, 1.34082214, 0.22341229, -0.64695482, -0.43970097,
        1.19207951, -1.24399921
    ],
    # Pesos Neurona 5
    [
        -0.84278605, -1.60843998, 0.52253117, 0.29938436, -0.90687637, 0.50199106,
        1.24933816, -0.26993642, 0.20407513, 1.27916132, -0.82454651, 0.45205041,
        0.01127588, 0.23402782, -0.2064211, 0.19089109, 0.30496303, 0.16074841,
        1.08686828, 0.50391815, 0.72008345, -1.12686786, 0.69735211, 0.06852595,
        -0.24531908, 0.04501357
    ]
]
'''

# Predecimos las letras
for i in range (len(letras)):
    prediccionBinaria = predecir_letra(letras[i], pesosEntrenamiento)
    print("Entrada:", letras[i])
    print("Letra:", chr(65 + i))
    print("Salida esperada:", salidaBinaria[i])
    print("Predicción:", prediccionBinaria)

#Entradas con ruido de dos letras
letrasRuido = np.array([
# Letra A con ruido
[1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
# Letra Z con ruido
[0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0]
])

# Predecimos las letras con ruido
print("Predicciones con ruido:")
for i in range (len(letrasRuido)):
    prediccionBinaria = predecir_letra(letrasRuido[i], pesosEntrenamiento)
    print("Entrada con ruido:", letrasRuido[i])
    print("Salida esperada:", salidaBinaria[i])
    print("Predicción:", prediccionBinaria)

#Graficamos los errores para cada neurona
for i in range(len(erroresEntrenamiento)):
    plt.plot(erroresEntrenamiento[i], label='Neurona ' + str(i+1))
plt.legend(title='Error por Neurona')
plt.xlabel('Épocas')
plt.ylabel('Error')
plt.title('Errores de Entrenamiento')
plt.grid()
plt.show()

#Graficamos los errores de la red en general.
totalEpocas = []
erroresEpoca = []
for i in range(epocas):
    errorEpoca = 0
    for neurona in erroresEntrenamiento:
        errorEpoca += neurona[i]
    erroresEpoca.append(errorEpoca)
    totalEpocas.append(i)
plt.plot(totalEpocas, erroresEpoca)
plt.xlabel('Épocas')
plt.ylabel('Error')
plt.title('Error Global')
plt.grid()
plt.show()

