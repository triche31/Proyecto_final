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
from modelo_page import modelo_page
from final import final
from intro import intro
from m_imagenes import m_imagenes



def main():
    menu = ["Introducción", "Modelo CNN", "Muestra aleatoria de imagenes", "Tipos de modelo", "Final"]

    st.set_page_config(page_title="CNN",
                       page_icon=":star2:",
                       layout="wide",
                       initial_sidebar_state="collapsed")


    menu = ["Introducción", "Modelo CNN", "Muestra aleatoria de imagenes", "Tipos de modelo", "Final"]

    page = st.sidebar.selectbox(label="Menu", options=menu)
    if page == "Introducción":

        st.title("Proyecto final: Convolutional Neural Networks (CNN) ")
        image2 = Image.open(
            "imagenes/hack.jpg")
        st.image(image=image2, width=400)

        st.subheader('Beatriz Mimosa')
        st.subheader('Aida Amoedo')
        st.subheader('Esther García')


        pass
    elif page == "Modelo CNN":
        modelo_page()
        pass
    elif page == "Muestra aleatoria de imagenes":
        m_imagenes()
        pass
    elif page == "Tipos de modelo":
        intro()

    elif page == "Final":
        final()
        pass

if __name__ == "__main__":
     main()


