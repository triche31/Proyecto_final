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

def edad():


    st.title('¿Qué edad tienes? :thinking_face:')
    st.subheader(':camera_with_flash: Hazte una foto y ¡Lanza la predicción!')
# Cargar el modelo desde el archivo pickle
    modelo = load_model("modelo_regresion.h5")

    edad = st.number_input("Ingresa tu edad", min_value=0, max_value=120, step=1)
    uploaded_file = st.camera_input("Tomar foto")


    if uploaded_file is not None:
        st.success("La foto se cargo correctamente.")


    para_predecir = st.button("Lanzar predicción")

    if uploaded_file is None and para_predecir:
        st.warning("¡No te has hecho la foto!")


    if para_predecir and uploaded_file is not None:

        img1 = image.load_img(uploaded_file, target_size = (200, 200))
        img = image.img_to_array(img1)
        img = np.expand_dims(img, axis=0)
        # Realizar la predicción con el modelo de edad
        edad_predicha = modelo.predict(img)


        st.image(img1, caption="Imagen cargada")

        st.write(f"La edad predicha es {edad_predicha}")
        st.write(f"La edad real es: {edad}")


if __name__ == "__main__":
     edad()