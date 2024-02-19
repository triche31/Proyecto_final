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


def final():
    st.title("¡Ya somos DATA SCIENCE!")



    col1, col2 = st.columns([1, 0.1])

    image2 = Image.open(
        "imagenes/grupo.jpg")
    col1.image(image=image2, width=700)

    image1 = Image.open(
        "imagenes/hack.jpg")
    col2.image(image=image1, width=50)

    # Mostramos la pregunta en la primera columna
    st.title("¿Te han gustado los proyectos :desktop_computer:?")

    # Usamos el slider en la segunda columna para recoger la respuesta del usuario
    respuesta = st.slider(label="", min_value=0, max_value=10, step=1)

    if respuesta == 0:
        st.error(":sob:")
    elif respuesta == 1:
        st.error(":cry::cry:")
    elif respuesta ==2:
        st.error(":weary::weary::weary:")
    elif respuesta == 3:
        st.error(":anguished::anguished::anguished::anguished:")
    elif respuesta == 4:
        st.error(":neutral_face::neutral_face::neutral_face::neutral_face::neutral_face:")
    elif respuesta == 5:
        st.info(":neutral_face::grimacing::grimacing::grimacing::grimacing::grimacing:")
    elif respuesta == 6:
        st.info(":smirk::smirk::smirk::smirk::smirk::smirk::smirk:")
    elif respuesta == 7:
        st.warning(":sunglasses::sunglasses::sunglasses::sunglasses::sunglasses::sunglasses::sunglasses::sunglasses:")
    elif respuesta == 8:
        st.info(":smiley::smiley::smiley::smiley::smiley::smiley::smiley::smiley::smiley:")
    elif respuesta == 9:
        st.info(":wink: :grin::wink: :grin::wink: :grin::wink: :grin::wink: :grin::wink: :grin::wink: :grin:")
    else:
        st.success(":heart_eyes: :heart_eyes: :heart_eyes: :heart_eyes:	:heart_eyes::heart_eyes::heart_eyes::heart_eyes::heart_eyes::heart_eyes::heart_eyes:")






if __name__ == "__main__":
    final()