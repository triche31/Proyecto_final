import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import requests
import os
from keras.models import load_model
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
from keras.preprocessing import image
import tensorflow as tf


def prueba():


    st.title('¿Hombre o mujer? :thinking_face:')
    st.subheader('	:arrow_left: Sube una foto en la que aperezca una cara')
# Cargar el modelo desde el archivo pickle
    modelo = load_model("modelo_female_male_color.h5")

    uploaded_file = st.sidebar.file_uploader(label="***:warning: Sube la foto :warning:***", type=["jpg"])

    para_predecir = st.button("Lanzar predicción")

    if uploaded_file is None and para_predecir:
        st.warning("¡No has subido la foto!")

    if para_predecir and uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(218, 178))
        X = image.img_to_array(img)
        image_expanded = np.expand_dims(X, axis=0)

        predicciones = modelo.predict(image_expanded)

        if (predicciones >= 0.5):
            genero = 'Hombre'
        else:
            genero = 'Mujer'

        # Muestra la imagen y el género predicho
        st.image(img, caption=f'Género: {genero}')

if __name__ == "__main__":
    prueba()