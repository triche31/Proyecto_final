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



def modelo_page():
    st.title("Convolutional Neural Networks (CNN)")
    st.title("------Female :woman-tipping-hand: Male :man-tipping-hand:------")

    image = Image.open(
        "imagenes/principal.jpg")

    st.image(image=image,
             caption="CNN",
             use_column_width=True)
    st.write(
        "Una Red Neuronal Convolucional **(Convolutional Neural Networks)** tiene una estructura similar a un perceptrón multicapa; están formadas por neuronas que tienen parámetros en forma de pesos y biases.")
    st.write(
        "Las redes neuronales convolucionales están formadas de muchas capas CONVOLUCIONALES (CONV) y capas de submuestreo, conocidas como POOLING. Seguidas por una o más capas.")
    st.write(
        "La capa convolucional aprende patrones locales dentro de la imagen en pequeñas ventanas de 2 dimensiones.")
    st.write(
        "De forma general podemos decir que el propósito de la capa convolucional es detectar características o rasgos visuales en las imágenes que utiliza para el aprendizaje.")

    st.write("Estas características que aprende pueden ser: aristas, colores, formas, conjuntos de píxeles.")
    image1 = Image.open(
        "imagenes/foto1.jpg")
    st.subheader(" Convolucion")
    st.image(image=image1,
             caption="pandas2",
             use_column_width=True)
    st.write(
        "De manera intuitiva, se puede decir que una capa convolucional es detectar características o rasgos visuales en las imágenes, como aristas, líneas, gotas de color, partes de una cara.")

    st.write(
        "Esto ayuda a que una vez que la red aprendió esta característica, la puede reconocer en cualquier imagen.")

    st.write(
        "En general las capas convolucionales operan sobre tensores 3D, llamados mapas de características (feature maps) donde se tienen las dimensiones de largo y ancho y una tercera que es el canal de las capas RGB.")
    st.subheader("Pooling")
    image2 = Image.open(
        "imagenes/foto.jpg")
    st.image(image=image2,
             caption="pandas2",
             use_column_width=True)
    st.write("Esta capa se suele aplicar inmediatamente después de la capa de convolución.")

    st.write(
        "Lo que hace la capa de pooling de manera simplificada es: reducir la información recogida por la capa convolucional y crean una versión condensada de la información contenida en esta capa.")


if __name__ == "__main__":
    modelo_page()