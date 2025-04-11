import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1, l2
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix

# Cargar el dataset Wine
data = load_wine()
X = data.data
y = data.target

feature_names = data.feature_names  # Nombres de las características

# Convertir a DataFrame para facilitar la manipulación
df = pd.DataFrame(X, columns=feature_names)
df['Clase de Vino'] = y

# Mostrar las primeras filas del DataFrame
print(df.head())

# Graficar las distribuciones de las características para cada clase de vino
plt.figure(figsize=(12, 8))
for i, feature in enumerate(feature_names):
    plt.subplot(4, 4, i+1)
    sns.histplot(data=df, x=feature, hue='Clase de Vino', multiple="stack", palette="Set2", kde=True)
    plt.title(feature)
    plt.tight_layout()

plt.show()

# Crear un mapa de calor para ver la correlación entre las características
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación de las Características del Vino')
plt.show()

# Graficar el par de características más correlacionadas (para ilustrar una relación)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='alcohol', y='flavanoids', hue='Clase de Vino', palette='Set1')
plt.title('Relación entre Alcohol y Flavonoides por Clase de Vino')
plt.show()

# Graficar la distribución de las clases usando un gráfico de barras
plt.figure(figsize=(8, 5))
sns.countplot(x='Clase de Vino', data=df, palette='Set2')
plt.title('Distribución de las Clases en el Wine Dataset')
plt.xlabel('Clase de Vino')
plt.ylabel('Número de Muestras')
plt.show()

plt.show()

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Estandarizar las características (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encoding de las etiquetas
y_train_ohe = to_categorical(y_train, num_classes=3)
y_test_ohe = to_categorical(y_test, num_classes=3)

# Crear el modelo de red neuronal multicapa (MLP)
#64,64
model = Sequential([
    Dense(32, activation='tanh', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)),  # Capa oculta con 64 neuronas
    Dense(16, activation='tanh', kernel_regularizer=l2(0.01)),  # Otra capa oculta con 64 neuronas
    Dense(3, activation='softmax')  # Capa de salida para clasificación multiclase (3 clases)
])

# Compilar el modelo
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
'''
history = model.fit(X_train, y_train_ohe,
                    epochs=100,
                    batch_size=16,
                    validation_data=(X_test, y_test_ohe),
                    verbose=1)
'''

history = model.fit(X_train, y_train_ohe,
                    epochs=100,
                    batch_size=16,
                    validation_split=0.2,
                    verbose=1)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(X_test, y_test_ohe, verbose=0)
print(f"Exactitud en el conjunto de prueba: {test_acc:.2f}")

# Graficar la exactitud y pérdida de entrenamiento/validación
epochs = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(12, 5))

# Gráfico de la exactitud
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], label='Exactitud Entrenamiento')
plt.plot(epochs, history.history['val_accuracy'], label='Exactitud Validación')
plt.title('Exactitud durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Exactitud')
plt.legend()

# Gráfico de la pérdida
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], label='Pérdida Entrenamiento')
plt.plot(epochs, history.history['val_loss'], label='Pérdida Validación')
plt.title('Pérdida durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()

#Predict test data
y_pred = model.predict(X_test)
print("Salidas predichas", y_pred)
#Print actual and predicted value
actual = np.argmax(y_test_ohe,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")

#--------matrix confusion-----

# Compute confusion matrix
print("datos reales", y_test)
print("datos predichos", predicted)
#cm = confusion_matrix(y_test, y_pred)
cm = confusion_matrix(actual, predicted)

print(cm)

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')

#------metricas------
n_classes = 3
y_true = actual
# Binarizar las etiquetas para ROC AUC
y_true_bin = y_test_ohe
# Calcular Precision, Recall, F1-Score y Accuracy para el problema multiclase
precision_macro = precision_score(y_true, predicted, average='macro')
recall_macro = recall_score(y_true, predicted, average='macro')
f1_macro = f1_score(y_true, predicted, average='macro')
accuracy = accuracy_score(y_true, predicted)

# Calcular AUC (Area Under the Curve) para cada clase
auc = roc_auc_score(y_true_bin, y_pred, multi_class="ovr")

# Mostrar las métricas
print(f'Precisión (Macro): {precision_macro:.2f}')
print(f'Recall (Macro): {recall_macro:.2f}')
print(f'F1-Score (Macro): {f1_macro:.2f}')
print(f'Exactitud (Accuracy): {accuracy:.2f}')
print(f'AUC (OvR): {auc:.2f}')

# Calcular métricas clase por clase
for i in range(n_classes):
    print(f"\nMétricas para la clase {i}:")

    # Convertir la clase actual en un problema binario (One vs Rest)
    y_true_binary = (y_true == i).astype(int)  # 1 para la clase actual, 0 para el resto
    y_pred_binary = (predicted == i).astype(int)  # Predicciones convertidas a binario

    # Calcular precisión, recall, f1-score y accuracy para la clase actual
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    accuracy = accuracy_score(y_true_binary, y_pred_binary)

    # Mostrar las métricas
    print(f'Precisión: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-Score: {f1:.2f}')
    print(f'Exactitud (Accuracy): {accuracy:.2f}')
    print(f'AUC: {auc:.2f}')

# Calcular ROC curve para cada clase
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
    roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_pred[:, i])

# Graficar las curvas ROC para cada clase
plt.figure()
colors = ['blue', 'green', 'red']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Clase {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC para Clasificación Multiclase')
plt.legend(loc="lower right")
plt.show()
