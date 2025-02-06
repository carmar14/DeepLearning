import pandas as pd
import matplotlib.pyplot as plt

# Creación de los datos a partir de la tabla de la imagen.
data = {
    "#": list(range(1, 22)),
    "Age": [59, 59, 62, 61, 52, 56, 51, 51, 74, 53, 55, 76, 54, 72, 57, 59, 52, 62, 62, 60, 56],
    "Gender": ["Female", "Male", "Female", "Female", "Male", "Male", "Female", "Female", "Female", "Female",
               "Male", "Female", "Female", "Female", "Male", "Female", "Female", "Female", "Female", "Female", "Male"],
    "Orientation Score": [2]*21,
    "Orientation Time": [7, 8, 8, 6, 13, 15, 4, 7, 8, 10, 9, 4, 14, 4, 26, 9, 9, 18, 7, 19, 26],
    "Clock Score": [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    "Clock Time": [116, 78, 95, 90, 138, 101, 41, 65, 101, 225, 151, 56, 89, 65, 461, 99, 128, 83, 76, 285, 92],
    "Fixation Score": [1, 1,1,1,1,1,1,1,1,1,1,1,1,1, 0, 1,1,1,1,1,1],
    "Fixation Time": [27, 13, 37, 14, 28, 44, 7, 28, 19, 28, 21, 12, 22, 14, 51, 31, 24, 29, 30, 24, 37],
    "Language Score": [4,4,4,4,4,4,4,2,4,2,4,4,4,4,4,2,2,4,4,2,0],
    "Language Time": [10, 15, 23, 14, 17, 15, 18, 33, 47, 27, 32, 14, 13, 13, 32, 27, 31, 36, 13, 72, 35],
    "Calculation Score": [3, 3, 5, 5, 5, 5, 5, 5, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    "Calculation Time": [32, 62, 50, 36, 52, 87, 59, 66, 78, 215, 47, 64, 81, 76, 152, 61, 81, 95, 82, 169, 102],
    "Memory Score": [3, 3, 3, 3, 1, 3, 3, 2, 1, 2, 3, 2, 1, 3, 2, 2, 3, 2, 1, 1, 3],
    "Memory Time": [11, 29, 19, 172, 22, 31, 18, 14, 77, 75, 34, 77, 89,57, 47, 15, 34, 64, 15, 178, 18],
    "Total Score": [14, 14, 16, 16, 14, 16, 15, 12, 10, 9, 16, 15, 14, 16, 14, 13, 14, 14, 13, 12, 12],
    "Total Time": [203, 205, 232, 332, 270, 293, 147, 213, 330, 580, 294, 227, 308, 229, 769, 242, 307, 325, 223, 747, 310]
}

# Creación del DataFrame
df = pd.DataFrame(data)

print(df)

# Seleccionar columnas relevantes para la visualización
categories = ['Orientation', 'Clock', 'Fixation', 'Language', 'Calculation', 'Memory']
scores = [f'{category} Score' for category in categories]
times = [f'{category} Time' for category in categories]

# Crear un DataFrame con las puntuaciones y los tiempos
df_summary = df[scores + times]

# Graficar las puntuaciones y los tiempos
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

# Puntuaciones por categoría
df_summary[scores].plot(kind='bar', ax=axes[0], stacked=True, colormap='Set2')
axes[0].set_title('Scores by Category',fontsize=10)
axes[0].set_ylabel('Score')
axes[0].set_xticks(range(len(df_summary)))  # Asegurar que el eje X tenga las etiquetas correctas
axes[0].set_xticklabels(range(1, len(df_summary) + 1), fontsize=6,rotation=0)  # Índices desde 1
axes[0].set_yticklabels(axes[0].get_yticks().astype(int), fontsize=6)  # Cambiar fontsize del eje Y

#axes[0].set_xlabel('Índice')

# Tiempos por categoría
df_summary[times].plot(kind='bar', ax=axes[1], stacked=True, colormap='Set3')
axes[1].set_title('Times by Category',fontsize=10)
axes[1].set_ylabel('Time (seconds)')
axes[1].set_xlabel('Users')
axes[1].set_xticks(range(len(df_summary)))
axes[1].set_xticklabels(range(1, len(df_summary) + 1), fontsize=6, rotation=0)  # Índices desde 1
axes[1].set_yticklabels(axes[1].get_yticks().astype(int), fontsize=6)  # Cambiar fontsize del eje Y


# Ajustar el diseño: aumentar el espacio entre subgráficos
plt.subplots_adjust(hspace=2.5)

# Mostrar la figura
plt.tight_layout()
plt.show()
