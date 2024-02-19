import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import requests
import os
import pickle
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf

def modelo_edad():


    st.title('¿En qué rango de edad te encuentras? :thinking_face:')
    st.subheader('	:arrow_left: Sube una foto en la que aperezca una cara')
# Cargar el modelo desde el archivo pickle
    modelo = load_model("modelo_edad.h5")

    uploaded_file = st.sidebar.file_uploader(label="***:warning: Sube la foto :warning:***", type=["jpg"])

    para_predecir = st.button("Lanzar predicción")

    if uploaded_file is None and para_predecir:
        st.warning("¡No has subido la foto!")


    if para_predecir and uploaded_file is not None:
        # Convertir la imagen cargada en un array numpy
        img1 = image.load_img(uploaded_file, target_size = (200, 200))
        img = image.img_to_array(img1)
        img = np.expand_dims(img, axis=0)
        # Realizar la predicción con el modelo de edad

        edad_predicha = modelo.predict(img)

        # Determinar la edad predicha
        edad_predicha = np.argmax(edad_predicha[0])

        # Mostrar la imagen
        st.image(img1, caption="Imagen cargada")

        # Mostrar la predicción de edad
        if edad_predicha == 0:
            st.write('Es una persona pequeña')
        elif edad_predicha == 1:
            st.write('Es una persona joven')
        elif edad_predicha == 2:
            st.write('Es una persona adulta')
        else:
            st.write('Es una persona mayor')
        # Muestra la imagen y el género predicho


if __name__ == "__main__":
    modelo_edad()