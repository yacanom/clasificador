import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizer

# Cargar el modelo
model = tf.saved_model.load("/home/cano/MINE/DL/clasificador/model")

# Tokenizador
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

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
    prediction = model(text_padded)  # Esta es solo una demostración, debes adaptarla a tu modelo

    st.write("Resultado de la clasificación:", prediction)
