#Camilo Roman Y Vivian Echeverri

import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr
import sounddevice as sd
import sklearn.preprocessing as skp
from scipy.io.wavfile import write, read as wavfile_read

# Parte de captura de Audio
def capturarAudio(archivoSalida,sample_rate=44100):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"Grabando durante 5 segundos...")
        audio = recognizer.record(source, duration=6)
        print("Grabación completa.")
    # Convertir el audio a un array de NumPy
    audio_data = np.frombuffer(audio.get_wav_data(), dtype=np.int16)
    write(archivoSalida, sample_rate, audio_data)  # Guardar el archivo de audio
    return audio_data

#Cargar archivo de audio de fondo
def cargarArchivoAudio(rutaArchivo, duracion, sample_rate):
    sample_rate, audio_data = wavfile_read(rutaArchivo)
    # Asegurarse de que el audio tenga la duración correcta
    n_samples = duracion * sample_rate
    if len(audio_data) > n_samples:
        # Si el audio es más largo, recortar
        audio_data = audio_data[:n_samples]
    return np.array(audio_data, dtype=np.int16)

#Función para reproducir audio
def reproducirAudio(audio_data, sample_rate):
    sd.play(audio_data, sample_rate)
    sd.wait()

# Función para normalizar los datos
def normalizar_datos(data):
    #Convertir los datos a flotantes
    data = data.astype(np.float32)
    #Verificar el rango de los datos
    print(f"Rango de los datos antes de normalizar:  Min{np.min(data)} - Max{np.max(data)}")
    #Normalizacación de los datos
    max_abs_value = np.max(np.abs(data))
    if max_abs_value == 0:
        print("Los datos tienen valor máximo 0, no se pueden normalizar.")
        return data  # Devuelve los datos originales
    else:
        normalized_data = data / max_abs_value
        print(f"Rango de los datos después de normalizar: Min{np.min(normalized_data)} - Max{np.max(normalized_data)}")
    return normalized_data 


# Función de activación lineal
def linear_function(x):
    return x

# Función de predicción Adaline
def adaline_predict(X, weights):
    return linear_function(np.dot(X, weights[1:]) + weights[0])

# Función de entrenamiento Adaline
def adaline_train(X, y, learning_rate, epochs):
    # Inicializar los pesos (uno más para el bias)
    weights = np.random.rand(X.shape[1] + 1)
    errors = []

    # Entrenamiento
    for _ in range(epochs):
        total_error = 0
        for xi, target in zip(X, y):
            # Calcular la salida (predicción)
            output = adaline_predict(xi, weights)
            # Calcular error absoluto
            error = (target - output) ** 2
            total_error += abs(error)
            # Actualizar los pesos
            update = learning_rate * (target - output)
            weights[1:] += update * xi
            weights[0] += update
        errors.append(total_error)
    return weights, errors

#Preparar los datos de entrada y salida
n_samples = 220500 #10 segundos de audio a 44100 Hz , duración * sample_rate
duration = 5
t = np.linspace(0, duration, n_samples)

#Las señales de audio
voz1 = capturarAudio("voz1.wav")
voz2 = capturarAudio("voz2.wav")
vozInstrumento = cargarArchivoAudio("Sonido-del-Mar.wav", duration, 44100)


# Verificar que todas las señales tengan la misma longitud
if vozInstrumento.ndim > 1:
    vozInstrumento = vozInstrumento[:, 0]  # Asegúrate de que sea unidimensional
voz1 = voz1[:n_samples]
voz2 = voz2[:n_samples]
vozInstrumento = vozInstrumento[:n_samples]

#Normalizar las señales de audio
voz1Normalizada = normalizar_datos(voz1) 
voz2Normalizada = normalizar_datos(voz2)
vozInstrumentoNormalizada = normalizar_datos(vozInstrumento)

X = voz1Normalizada+ voz2Normalizada+ vozInstrumentoNormalizada
X = normalizar_datos(X)

#Graficar la salida esperada
y = np.array([voz1Normalizada, voz2Normalizada, vozInstrumentoNormalizada])
for i in range(len(y)):
    plt.plot(t, y[i], linestyle='--', label=f"Voz {i + 1}"+" sin ruido")
plt.plot(t, X, linestyle='--' ,label="Voz con Ruido")
plt.legend()
plt.title("Señales de Audio Originales y Mezcladas")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud de la señal")
plt.grid()
plt.show()


#Crear las entradas para la red neuronal Adaline
delay = 900
noisy_signal = np.array([X[i:i+delay] for i in range(n_samples - delay)])
print(noisy_signal.shape)

#Entrenar la red neuronal Adaline para separar las señales originales
trained_weights = []
globalErrors = []
for output in y:
    d = output[delay:]
    # Entrenar la red neuronal Adaline
    weights, errors = adaline_train(noisy_signal,d,0.01,50)
    trained_weights.append(weights)
    globalErrors.append(errors)

#Predecir las señales filtradas con los pesos entrenados
filtered_signals = []
for i in range(len(y)):
    predict = ([adaline_predict(xi, trained_weights[i]) for xi in noisy_signal])
    filtered_signals.append(predict)

#Graficar los errores de entrenamiento
plt.figure()
for i in range(len(globalErrors)):
    plt.plot(globalErrors[i], label=f"Error Red para la voz {i + 1}")
plt.xlabel("Época")
plt.ylabel("Error Red Por xi")
plt.title("Error de la Red")
plt.grid(True)
plt.legend()
plt.show()

#Graficar las señales originales y las señales filtradas
plt.figure()
for i in range (len(y)):
    plt.plot(t[delay:], y[i][delay:], linestyle='--',label=f"Voz {i + 1} original")
    plt.plot(t[delay:], filtered_signals[i], linestyle='--', label=f"Voz {i + 1} filtrada")
    plt.grid()
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud de la señal")
    plt.legend()
    plt.title(f"Señal de voz x{i + 1} original vs Predicciones de la red para la voz x{i + 1}")
    plt.show()

#Reproducir los audios originales y filtrados
print("Reproduciendo audio original...")
reproducirAudio(voz1, 44100)
print("Reproduciendo audio filtrado...")
reproducirAudio(filtered_signals[0], 44100) 
print("Reproduciendo audio original...")
reproducirAudio(voz2, 44100)
print("Reproduciendo audio filtrado...")
reproducirAudio(filtered_signals[1], 44100) 
print("Reproduciendo audio original...")
reproducirAudio(vozInstrumento, 44100)
print("Reproduciendo audio filtrado...")
reproducirAudio(filtered_signals[2], 44100) 

# Obtener la longitud mínima entre los arrays
min_length = min(len(voz1Normalizada), len(filtered_signals[0]))

# Recortar los arrays a la longitud mínima
voz1Normalizada = voz1Normalizada[:min_length]
filtered_signals[0] = filtered_signals[0][:min_length]

# Calcular la correlación
correlacionVoz1 = np.corrcoef(voz1Normalizada, filtered_signals[0])[0,1]
print("Correlación Audio 1: ", correlacionVoz1)

# Repetir el proceso para los otros arrays
min_length = min(len(voz2Normalizada), len(filtered_signals[1]))
voz2Normalizada = voz2Normalizada[:min_length]
filtered_signals[1] = filtered_signals[1][:min_length]

correlacionVoz2 = np.corrcoef(voz2Normalizada, filtered_signals[1])[0,1]
print("Correlación Audio 2: ", correlacionVoz2)

min_length = min(len(vozInstrumentoNormalizada), len(filtered_signals[2]))
vozInstrumentoNormalizada = vozInstrumentoNormalizada[:min_length]
filtered_signals[2] = filtered_signals[2][:min_length]

correlacionInstrumento = np.corrcoef(vozInstrumentoNormalizada, filtered_signals[2])[0,1]
print("Correlación Audio Instrumento: ", correlacionInstrumento)