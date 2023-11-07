import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizer

# Cargar el modelo
model = tf.saved_model.load("exported_model")

# Tokenizador
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

label_mapping = {0: "negative", 1: "neutral", 2: "positive"}

st.title("Clasificación de Texto")

# Agregar un campo de entrada de texto
text_input = st.text_area("Ingresa el texto a clasificar", "")

if st.button("Clasificar"):
    # Preprocesar el texto
    text = [text_input]  # Convertir el texto en una lista (simulando el conjunto de prueba)

    # Tokenizar y padear el texto
    text_encoded = [tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=128) for text in text]
    max_sequence_length = 98
    text_padded = tf.keras.preprocessing.sequence.pad_sequences(text_encoded, maxlen=max_sequence_length, padding='post', truncating='post')

    # Realizar la predicción con el modelo
    raw_prediction = model(text_padded)
    predicted_labels_indices = tf.argmax(raw_prediction, axis=1)  # Obtener los índices de las etiquetas

    # Convertir los índices de las etiquetas a etiquetas originales utilizando el diccionario
    predicted_labels = [label_mapping[index] for index in predicted_labels_indices.numpy()]

    st.write("Resultado de la clasificación:", predicted_labels[0])