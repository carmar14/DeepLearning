import speech_recognition as sr
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pygame
import soundfile as sf

class FiltroAdaline:
    def __init__(self, tamanoEntrada, tamanoSalida, tasaAprendizaje=0.01, epocas=100):
        # Inicializar pesos: una fila por cada salida, incluyendo el bias
        self.pesos = np.zeros((tamanoSalida, tamanoEntrada + 1))
        print('pesos:', self.pesos.shape)
        self.tasaAprendizaje = tasaAprendizaje
        self.epocas = epocas
        # vector de error
        self.errores = []

    def prediccion(self, entradas):
        entradas = np.insert(entradas, 0, 1)  # Insertar el término de bias
        return np.dot(self.pesos, entradas)  # Producto punto para cada salida

    def entrenamiento(self, entradas, salidasDeseadas):
        for epoca in range(self.epocas):
            # calculo del mse(error cuadratico medio)
            mse_ = 0
            for i in range(len(entradas)):
                entradas_i = np.insert(entradas[i], 0, 1)  # Insertar el término de bias
                y = np.dot(self.pesos, entradas_i)  # Predicción actual
                error = salidasDeseadas[i] - y  # Error de predicción
                #calculo del mse
                mse_ += sum(error ** 2)
                # Actualizar los pesos para cada salida
                self.pesos += 2*self.tasaAprendizaje * np.outer(error, entradas_i)

            self.errores.append(np.mean(mse_))
            # Opcional: imprimir el error promedio cada cierto número de épocas
            if (epoca + 1) % 20 == 0:
                y_pred = self.prediccionLote(entradas)

                mse = np.mean((salidasDeseadas - y_pred) ** 2)
                self.errores.append(mse)
                print(f"epoca {epoca + 1}/{self.epocas}, MSE: {mse:.5f}")

    def prediccionLote(self, entradas):
        entradas_bias = np.insert(entradas, 0, 1, axis=1)  # Insertar el término de bias en todas las muestras
        return np.dot(entradas_bias, self.pesos.T)  # Producto punto para todas las salidas

#Normalizar la señal de audio
def normalizarAudio(audio):
    amplitudMax = np.max(np.abs(audio))
    if amplitudMax > 0:
        audioNormalizado = audio / amplitudMax
    else:
        audioNormalizado = audio
    return audioNormalizado

#Función para reproducir los audios
def reproducir_audio(archivo):
    pygame.mixer.music.load(archivo)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10) # Esperar a que termine de reproducirse

pygame.mixer.init()
#Audio de fondo
audioFondo = 'audioFondo.wav'
#Usar librosa para obtener los datos de la señal de audio
datosAudioFondo, freceuncia = librosa.load(audioFondo, sr=None)

#Escucha lo que dices y lo guarda
def escuchar(nombre):
    entrada = sr.Recognizer()
    with  sr.Microphone() as source:
        print('Di algo:')
        audio = entrada.listen(source, phrase_time_limit=5)
        #Guardar audio en un archivo
        nombre = nombre + '.wav'
        with open(nombre, 'wb') as f:
            f.write(audio.get_wav_data())
        print('Audio guardado exitosamente')
        #Obtener los datos de la señal de audio
        datosAudio, freceuncia = librosa.load(nombre, sr=None)
        return datosAudio, freceuncia

#Capturar las conversaciones
conversacion1, freceuncia1 = escuchar('conversacion1')
conversacion2, freceuncia2 = escuchar('conversacion2')
#Tomar la misma cantidad de muestras de las conversaciones y el audio de fondo
min_len = min(len(conversacion1), len(conversacion2), len(datosAudioFondo))
conversacion1 = conversacion1[:min_len]
conversacion2 = conversacion2[:min_len]
datosAudioFondo = datosAudioFondo[:min_len]

#Sumar las señales de audio
suma = conversacion1 + conversacion2 + datosAudioFondo
#Normalizar la señal de audio
normalizado = normalizarAudio(suma)
print('suma nomarlizada', normalizado)
#Reproducir audio normalizado
sf.write('suma_normalizada.wav', normalizado, freceuncia)

# Configuración de ADALINE
t = np.arange(len(normalizado))  # Vector de tiempo
delay = 900  # Número de retrasos (taps) para la entrada
#Se toman las muestras de la señal normalizada para las entradas
entradas = np.array([normalizado[i:i + delay] for i in range(min_len - delay)])
print('entradas:', entradas.shape)
# Señales originales retrasadas para alinear con entradas
audio1Retrasado = conversacion1[delay:]
audio2Retrasado  = conversacion2[delay:]
audio3Retrasado= datosAudioFondo[delay:]
salidaDeseada = np.stack((audio1Retrasado, audio1Retrasado , audio3Retrasado), axis=1) 
print('salidaDeseada:', salidaDeseada.shape)
print(len(normalizado), len(conversacion1), len(conversacion2), len(datosAudioFondo))
adaline = FiltroAdaline(tamanoEntrada=delay, tamanoSalida=3, tasaAprendizaje=0.0001, epocas=160)
adaline.entrenamiento(entradas, salidaDeseada)

audiosRecuperados = adaline.prediccionLote(entradas)  # Matriz de salidas recuperadas (n_samples - delay, 3)
audioRecuperado1 = audiosRecuperados[:, 0]
audioRecuperado2 = audiosRecuperados[:, 1]
audioRecuperado3 = audiosRecuperados[:, 2]

# Graficar el mse a medida que pasa las epocas                    
plt.figure()
plt.plot(range(1, len(adaline.errores) + 1), adaline.errores, marker='o')
plt.xlabel('Época')
plt.ylabel('Error Global')
plt.title('Error Global de ADALINE')
plt.grid(True)

#Comparar señales originales con las recuperadas
plt.figure(figsize=(16, 14))

#Señal compuesta
reproducir_audio('suma_normalizada.wav')
plt.subplot(4, 1, 1)
plt.plot(t, normalizado, label="Señal Compuesta")
plt.title("Señal Compuesta")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)

#Conversa 1: Original vs Recuperada
#Reproducir audio original y recuperado
reproducir_audio('conversacion1.wav')
sf.write('conversacion1_recuperada.wav', audioRecuperado1, freceuncia)
reproducir_audio('conversacion1_recuperada.wav')
plt.subplot(4, 1, 2)
plt.plot(t[delay:], audioRecuperado1, '.',label="Conversa 1 Recuperada", color="r")
plt.plot(t, conversacion1, '--', label="Conversa 1 Original", color="k")
plt.title("Recuperación de la Conversa 1")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)

#Conversa 2: Original vs Recuperada
#Reproducir audio original y recuperado
reproducir_audio('conversacion2.wav')
sf.write('conversa2_recuperada.wav', audioRecuperado2, freceuncia)
reproducir_audio('conversa2_recuperada.wav')
plt.subplot(4, 1, 3)
plt.plot(t[delay:], audioRecuperado2,'.', label="Conversa 2 Recuperada", color="g")
plt.plot(t, conversacion2, '--', label="Conversa 2 Original", color="k")
plt.title("Recuperación de la Conversa 2")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)


#Audio de fondo: Original vs Recuperada
#Reproducir audio original y recuperado
reproducir_audio('audioFondo.wav')
sf.write('audioFondo_recuperado.wav', audioRecuperado3, freceuncia)
reproducir_audio('audioFondo_recuperado.wav')
plt.subplot(4, 1, 4)
plt.plot(t[delay:], audioRecuperado3,'.', label="Audio de fondo Recuperado", color="b")
plt.plot(t, datosAudioFondo, '--', label="Audio de fondo Original", color="k")
plt.title("Recuperación del Audio de fondo")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)

plt.subplots_adjust(hspace=1.5)

plt.show()