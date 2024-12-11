import os
import numpy as np
import wget
import zipfile
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Input, add
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications.inception_v3 import preprocess_input


# ------------------------
# Descargar el Dataset
# ------------------------
def download_dataset():
    dataset_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    captions_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

    if not os.path.exists("Flickr8k_Dataset.zip"):
        print("Descargando imágenes...")
        wget.download(dataset_url, "Flickr8k_Dataset.zip")

    if not os.path.exists("Flickr8k_text.zip"):
        print("\nDescargando descripciones...")
        wget.download(captions_url, "Flickr8k_text.zip")

    print("\nExtrayendo archivos...")
    with zipfile.ZipFile("Flickr8k_Dataset.zip", 'r') as zip_ref:
        zip_ref.extractall("Flickr8k")
    with zipfile.ZipFile("Flickr8k_text.zip", 'r') as zip_ref:
        zip_ref.extractall("Flickr8k")
    print("Dataset preparado.")


# Llamar a la función para descargar y preparar los datos
download_dataset()

# ------------------------
# Rutas del Dataset
# ------------------------
img_dir = 'Flickr8k/Flickr8k_Dataset'
desc_path = 'Flickr8k/Flickr8k_text/Flickr8k.token.txt'

# ------------------------
# Extracción de características
# ------------------------
base_model = InceptionV3(weights='imagenet')
model_incep = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def extract_features(directory):
    features = {}
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        img = load_img(img_path, target_size=(299, 299))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature = model_incep.predict(img, verbose=0)
        features[img_name] = feature
    return features

features = extract_features(img_dir)

# ------------------------
# Procesamiento de descripciones
# ------------------------
def load_descriptions(filepath):
    descriptions = {}
    with open(filepath, 'r') as file:
        for line in file:
            tokens = line.strip().split('\t')
            img_id, caption = tokens[0], tokens[1]
            img_id = img_id.split('.')[0]
            if img_id not in descriptions:
                descriptions[img_id] = []
            descriptions[img_id].append('startseq ' + caption + ' endseq')
    return descriptions

descriptions = load_descriptions(desc_path)

tokenizer = Tokenizer()
all_captions = [caption for captions in descriptions.values() for caption in captions]
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

def create_sequences(tokenizer, max_length, descriptions, photo_features):
    X1, X2, y = [], [], []
    for img_id, captions in descriptions.items():
        for caption in captions:
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(photo_features[img_id][0])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

max_length = max(len(caption.split()) for caption in all_captions)
X1, X2, y = create_sequences(tokenizer, max_length, descriptions, features)

# ------------------------
# Definición del modelo
# ------------------------
inputs1 = Input(shape=(2048,))
img_model = Dropout(0.5)(inputs1)
img_model = Dense(256, activation='relu')(img_model)

inputs2 = Input(shape=(max_length,))
txt_model = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
txt_model = Dropout(0.5)(txt_model)
txt_model = LSTM(256)(txt_model)

decoder = add([img_model, txt_model])
decoder = Dense(256, activation='relu')(decoder)
outputs = Dense(vocab_size, activation='softmax')(decoder)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# ------------------------
# Entrenamiento del modelo
# ------------------------
model.fit([X1, X2], y, epochs=20, batch_size=64)

# ------------------------
# Generar descripciones
# ------------------------
def generate_caption(model, tokenizer, photo_feature, max_length):
    caption = 'startseq'
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        y_pred = model.predict([photo_feature, seq], verbose=0)
        y_pred = np.argmax(y_pred)
        word = tokenizer.index_word.get(y_pred)
        if word == 'endseq':
            break
        caption += ' ' + word
    return caption.replace('startseq ', '').replace(' endseq', '')

# Probar una imagen
img_path = '/path/to/Flickr8k/Images/example.jpg'
img = load_img(img_path, target_size=(299, 299))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)
photo_feature = model_incep.predict(img, verbose=0)
caption = generate_caption(model, tokenizer, photo_feature, max_length)
print("Descripción generada:", caption)
