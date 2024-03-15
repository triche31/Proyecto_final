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
import tensorflow as tf
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# from keras.preprocessing.image import ImageDataGenerator
from prueba import prueba
import seaborn as sns
import plotly.express as px
import ast
from modelo_edad import modelo_edad
from mod_edad_concreta import edad


def intro():
    st.sidebar.markdown('******')

    tipo_modelos = ["Female-Male", "Estimación de la edad en rangos", "Estimación de la edad"]
    modelos = st.sidebar.selectbox(label="Modelos",
                                   options=tipo_modelos)

    if modelos == "Female-Male":

        st.title("Modelo de estimación del sexo :clapper:")

        image = Image.open(
            "imagenes/intro_principal.jpg")
        st.image(image=image,
                 caption="Masculino vs Femenino",
                 width=400)

        informacion = ["Información de los datos", "Código del Modelo", "Modelo en producción", "Métricas", "Gráficas"]
        choice = st.selectbox("**Información**", options=informacion)
        st.header(f"{choice}")

        if choice == "Información de los datos":
            st.write("""Para nuestro proyecto, hemos utilizado los datos proporcionados por IMDB-WIKI. """
                     "Estos datos, en principio, se han utilizado en proyectos anteriores, para generar modelos de estimación de edad. No obstante, cuando descargamos el contenido, las imágenes y CSVs no estaban clasificados correctamente, "
                     "por lo que decidimos utilizarlos para generar un modelo Convolucional que clasificara solo entre mujer y hombre ")
            st.markdown("""Se puede acceder a la información y datos a través del siguiente enlace: 
                            [Open IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).""")
            st.write(
                "Aquí mostramos la cantidad de imágenes que se han utilizado para realizar el modelo en las distintas fases de Train, Test y Validación")
            df1 = pd.read_csv(
                "datos_f_m/datos_numero_fotos.csv")

            nuevos_nombres = {
                'cant_ima_entre_femenino': 'Entrenamiento imágenes mujer',
                'cant_ima_entre_masculino': 'Entrenamiento imágenes hombre',
                'cant_ima_val_femenino': 'Validación imágenes mujer',
                'cant_ima_val_masculino': 'Validación imágenes hombre',
                'cant_ima_test_femenino': 'Test imágenes mujer',
                'cant_ima_test_masculino': 'Test imágenes hombre'}

            df1 = df1.rename(columns=nuevos_nombres)

            fig_bar = px.bar(data_frame=df1,
                             text_auto=True)
            fig_bar.update_yaxes(categoryorder="total ascending")
            fig_bar.update_xaxes(title_text="Representación imágenes usadas")
            fig_bar.update_yaxes(title_text="")
            st.plotly_chart(fig_bar)

        if choice == "Código del Modelo":
            st.write("Gestionamos las imágenes que vamos a pasar al modelo y realizamos en ellas ciertos ajustes:")
            code = """
                nrows = 4
                ncols = 4
                pic_index = 0
                fig = plt.gcf()
                fig.set_size_inches(ncols * 4, nrows * 4)
                """
            st.code(body=code, language="python")

            st.write(
                "Procedemos a crear nuestro modelo, empezando con las convolucionales en 32 y terminando con un clasficador binario:")
            code = """modelo = Sequential()

modelo.add(Conv2D(32, (3,3), input_shape = (218, 178, 1), padding = 'same', activation='relu'))
modelo.add(MaxPooling2D(2,2))

modelo.add(Conv2D(64,(3,3), padding = 'same',activation='relu'))
modelo.add(MaxPooling2D(2,2))

modelo.add(Conv2D(128,(3,3), padding = 'same', activation='relu'))
modelo.add(MaxPooling2D(2,2))

modelo.add(Flatten())

modelo.add(Dense(256, activation='relu'))
modelo.add(Dropout(0.5))

modelo.add(Dense(512, activation='relu'))
modelo.add(Dropout(0.5))

modelo.add(Dense(1, activation='sigmoid'))

modelo.summary()"""

            st.code(body=code, language="python")
            st.write("Copilamos el modelo:")
            code = """modelo.compile(loss= 'binary_crossentropy', optimizer='adam', metrics= ['accuracy'])"""
            st.code(body=code, language="python")
            st.write(
                "Para construir nuestra matriz X y vector hacemos una normalización de los datos de entrada. Además, vamos a generar imágenes modificando las que tenemos.")
            code = """ train_datagen =  ImageDataGenerator(
    rescale = 1.0/255.0,
    rotation_range = 40, 
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest', 
    )

val_datagen =  ImageDataGenerator(rescale = 1.0/255.0)
test_datagen =  ImageDataGenerator(rescale = 1.0/255.0)


train_generator = train_datagen.flow_from_directory(
    train_dir, batch_size = 20, class_mode = 'binary', color_mode = "grayscale" ,
    target_size = (218, 178))

test_generator = test_datagen.flow_from_directory(
    test_dir, batch_size = 20, class_mode = 'binary', color_mode = "grayscale",
    target_size = (218, 178))

val_generator = val_datagen.flow_from_directory(
    validation_dir, batch_size = 20, class_mode = 'binary', color_mode = "grayscale",
    target_size = (218, 178))

print('Tenemos las clases', test_generator.class_indices)"""
            st.code(body=code, language="python")
            st.write(
                "Ahora, establecemos nuestros hiperparámetros antes de proceder a entrenar el modelo. Hemos elegido utilizar 15 épocas porque viendo nuestra curva de aprendizaje era lo mejor en este caso:")
            code = """batch_size = 20
steps_per_epoch = train_generator.n // batch_size
validation_steps = val_generator.n // batch_size
history = modelo.fit(train_generator, 
                               steps_per_epoch= steps_per_epoch, 
                               epochs=15,
                               validation_data = val_generator,
                               validation_steps = validation_steps)"""
            st.code(body=code, language="python")

        if choice == "Modelo en producción":
            prueba()

            pass
        if choice == "Métricas":
            df = pd.read_csv(
                "datos_f_m/Metricas.csv")

            accuracy = df["accuracy"][0][-18:-2]
            loss = df["loss"][0][-19:-2]

            st.write(f"Accuracy: {accuracy}")
            st.write(f"Perdida: {loss}")

        if choice == "Gráficas":
            tab1, tab2, tab3 = st.tabs(["Gráfica de los datos", "Accuracy", "Perdidas"])

            df1 = pd.read_csv(
                "datos_f_m/datos_numero_fotos.csv")

            nuevos_nombres = {
                'cant_ima_entre_femenino': 'Entrenamiento imágenes mujer',
                'cant_ima_entre_masculino': 'Entrenamiento imágenes hombre',
                'cant_ima_val_femenino': 'Validación imágenes mujer',
                'cant_ima_val_masculino': 'Validación imágenes hombre',
                'cant_ima_test_femenino': 'Test imágenes mujer',
                'cant_ima_test_masculino': 'Test imágenes hombre',
                # Añade más pares de nombres de columnas originales y nuevos según sea necesario
            }

            df1 = df1.rename(columns=nuevos_nombres)
            with tab1:
                fig_bar = px.bar(data_frame=df1,
                                 text_auto=True)
                fig_bar.update_yaxes(categoryorder="total ascending")
                fig_bar.update_xaxes(title_text="Representación imágenes usadas")
                fig_bar.update_yaxes(title_text="")
                st.plotly_chart(fig_bar)

            grafica1 = Image.open("datos_f_m/Performance de mi red neuronal2.png")
            with tab2:
                st.image(image=grafica1,
                         caption="Performance de la red neuronal",
                         use_column_width=True)
            grafica2 = Image.open("datos_f_m/Training and validation loss.png")
            with tab3:
                st.image(image=grafica2,
                         caption="Training and validation loss",
                         use_column_width=True)

    if modelos == "Estimación de la edad en rangos":
        st.title("Estimación de la edad en rangos :baby_bottle:	:glass_of_milk::beer::wine_glass:")
        image = Image.open(
            "imagenes/edad1.jpg")
        st.image(image=image,
                 use_column_width=True)

        informacion = ["Información de los datos", "Código del Modelo", "Modelo en producción", "Métricas", "Gráficas"]
        choice = st.selectbox("**Información**", options=informacion)
        st.header(f"{choice}")
        if choice == "Información de los datos":
            st.write(
                "Para realizar el estudio de los rangos de edad, en primer lugar se ha entrenado el modelo para diferenciar entre sexos. Posteriormente, hemos generado un modelo"
                " de clasificación multiclase, donde diferenciamos cuatro rangos de edad. ")

            mapeo_edad = {0: (1, 14), 1: (15, 40), 2: (41, 60), 3: (61, 116)}
            df_mapeo_edad = pd.DataFrame.from_dict(mapeo_edad, orient='index', columns=['Edad mínima', 'Edad máxima'])
            st.dataframe(df_mapeo_edad)

            st.write(
                "Para ello hemos trabajado con distintos tipos de datos. En un primer momento, clasificamos a mano datos de fotos con un csv que contenía la información de la edad.")
            st.write(" A continuación mostramos el csv con la información:")
            df7 = pd.read_csv(
                "datos_mod2/primer.csv", )

            st.dataframe(df7)

            st.write(
                "Los resultados obtenidos, tras la clasificación manual y el entrenamiento del modelo, no eran buenos, el modelo no entrenaba y por tanto las métricas no cambiaban. Determinamos la necesidad de incluir más datos para poder entrenar el modelo.")
            st.write("Mostramos el resultado de los primeros datos de entrenamiento con accuracy constante:")
            image = Image.open(
                "datos_mod2/acc.jpg")
            st.image(image=image,
                     caption="Estimación de la edad",
                     width=700)
            st.write(
                " Despúes de procesar los datos, que no estaban clasificados, pasamos de entrenar un modelo de 3000 imágenes aproximadamente a 34000 imágenes. El modelo finalmente entrenó y mejoró las métricas. No obtuvimos un modelo con un accuracy eleveado"
                " pero contábamos con ello.")
            st.write("Este es el DataFrame con el que finalmente fue entrenado el modelo:")
            df8 = pd.read_csv("datos_mod2/data.csv")

            st.dataframe(df8)
            st.write(
                "El contenido del DataFrame se modificó para acceder a la edad y el sexo, información que estaba contenida en los primeros dígitos del nombre.")

            st.markdown("""A continuación mostramos las páginas donde hemos extraído los datos: 
                                        [Open KAGGEL](https://www.kaggle.com/datasets/jangedoo/utkface-new) y [Open APPA-REAL](https://chalearnlap.cvc.uab.es/dataset/26/description/)""")
            st.write(
                "Aquí mostramos la cantidad de imágenes que se han utilizado para realizar el modelo en las distintas fases de Train, Test y Validación:")
            df10 = pd.read_csv(
                "datos_mod2/datos_num.csv")

            fig_bar = px.bar(data_frame=df10,
                             text_auto=True)
            fig_bar.update_yaxes(categoryorder=None)
            fig_bar.update_xaxes(title_text="Representación imágenes usadas")
            fig_bar.update_yaxes(title_text="")
            st.plotly_chart(fig_bar)

            pass
        if choice == "Código del Modelo":
            st.write("Sacamos las imágenes del data y los datos asociados a ello en un DataFrame:")
            code = """datos_imagenes=[]

for img in os.listdir(data):

    img_path = f"{data}/{img}" 

    datos_imagenes.append({'ruta':img_path,'img':img})

df_data = pd.DataFrame(datos_imagenes)


df_data.head(3)"""
            st.code(body=code, language="python")
            st.write("Creamos la columna sexo donde 0 es masculino, 1 es femenino:")
            code = """
df_data['sexo']= df_data['img'].map(lambda x: x.split("_")[1])"""
            st.code(body=code, language="python")
            st.write("Creamos la columna edad:")
            code = """ df_data['edad'] = [valor.split('_')[0] for valor in df_data['img']]"""
            st.code(body=code, language="python")
            st.write(" Mapeamos cada rango de edad a un número único:")
            code = """ mapeo_edad = {0: (1, 14), 1: (15, 40), 2: (41, 60), 3: (61, 116)}

def asignar_rango_edad(edad):
    for key, (inicio, fin) in mapeo_edad.items():
        if inicio <= edad <= fin:
            return key
    return None """
            st.code(body=code, language="python")
            st.write("Cargamos las imágenes del Train")
            code = """X = []
y = []

tamaño = (200, 200)

for img_path, edad in zip (df_data['ruta'], df_data['edad']):

    imagen_train = Image.open(img_path)

    imagen_resize = imagen_train.resize(tamaño)

    X.append(imagen_resize)

    y.append(asignar_rango_edad(int(edad)))
                 """
            st.code(body=code, language="python")
            st.write(
                "Para poder trabajar con estas imágenes tenemos que pasarlo a numpy y luego ya quedarnos con los datos de X e y . También separamos las variables:")
            code = """ X=np.array(X)
y=np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

print(f"Conjunto de Train: {X_train.shape, y_train.shape}")
print(f"Conjunto de Test: {X_test.shape, y_test.shape}")"""
            st.code(body=code, language="python")

            code = """X_train_cantidad=X_train.shape
y_train_cantidad=y_train.shape
X_test_cantidad=X_test.shape
y_test_cantidad=y_test.shape"""
            st.code(body=code, language="python")
            st.write("One Hot Encoding:")
            code = """ y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = len(y_train[0])
num_classes"""
            st.code(body=code, language="python")
            st.write("Procedemos a crear nuestro modelo para imágenes:")
            code = """ modelo = Sequential()

modelo.add(Conv2D(32, (3,3), input_shape = (200,200,3), activation='relu'))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(2,2))


modelo.add(Conv2D(64,(3,3), activation='relu'))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(2,2))

modelo.add(Conv2D(200,(3,3), activation='relu'))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(2,2))

modelo.add(Flatten())

modelo.add(Dense(200, activation='relu'))
modelo.add(Dropout(0.5))

modelo.add(Dense(200, activation='relu'))
modelo.add(Dropout(0.5))

modelo.add(Dense(64, activation = "relu"))
modelo.add(Dropout(0.5))

modelo.add(Dense(4, activation='softmax'))

modelo.summary()
modelo.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])
history = modelo.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 35, verbose = 1)"""
            st.code(body=code, language="python")

        if choice == "Modelo en producción":
            modelo_edad()
            pass
        if choice == "Métricas":
            df11 = pd.read_csv(
                "Scores (1).csv")

            accuracy = df11['0'].iloc[1]
            loss = df11['0'].iloc[0]

            st.write(f"Accuracy: {accuracy}")
            st.write(f"Perdida: {loss}")
            pass
        if choice == "Gráficas":
            tab1, tab2, tab3 = st.tabs(["Gráfica de los datos", "Accuracy", "Perdidas"])

            df2 = pd.read_csv("datos_mod2/conteo_sexo.csv")

            with tab1:
                fig_bar = px.bar(data_frame=df2,
                                 text_auto=True)
                fig_bar.update_yaxes(categoryorder="total ascending")
                fig_bar.update_xaxes(title_text="Representación imágenes usadas")
                fig_bar.update_yaxes(title_text="")
                st.plotly_chart(fig_bar)

            grafica1 = Image.open(
                "datos_mod2/WhatsApp Image 2024-02-15 at 10.46.50.jpeg")
            with tab2:
                st.image(image=grafica1,
                         caption="Performance de la red neuronal",
                         use_column_width=True)
            grafica2 = Image.open(
                "datos_mod2/WhatsApp Image 2024-02-15 at 10.46.50 (1).jpeg")
            with tab3:
                st.image(image=grafica2,
                         caption="Función de perdida",
                         use_column_width=True)

    if modelos == "Estimación de la edad":
        st.title("Estimación de la edad exacta	:bulb:")
        image = Image.open(
            "imagenes/edad_concreta.jpg")
        st.image(image=image,
                 caption="Estimación de la edad",
                 width=400)

        informacion = ["Información de los datos", "Código del Modelo", "Modelo en producción", "Métricas", "Gráficas"]
        choice = st.selectbox("**Información**", options=informacion)
        st.header(f"{choice}")
        if choice == "Información de los datos":
            st.markdown(
                "En este caso, hemos utilizado los datos del modelo de estimación de edad por rangos. Sin embargo, hemos incluído un modelo de regresión logística para poder predecir la edad exacta.")
            st.write(
                "No hemos obtenido métricas adecuadas para un buen uso del modelo, pero aún así hemos querido publicar el mismo para poder mostrar el trabajo y así,  ponerlo a prueba.")
            st.write(
                "La idea en un futuro, es poder modificar y añadir información para que el modelo sea más fiable y sensible.")
            st.write(
                "En los siguientes apartados del modelo, se podrá encontrar la información del mismo. En el caso de querer acceder a los datos usados de entrenamiento y test, se podrá acceder en el apartado del anterior modelo en Infomación de los Datos")

            pass
        if choice == "Código del Modelo":
            st.write("Sacamos las imágenes del data y los datos asociados a ello en un DataFrame:")
            code = """ datos_imagenes=[]

for img in os.listdir(data):

    img_path = f"{data}/{img}" 

    datos_imagenes.append({'ruta':img_path,'img':img})

df_data = pd.DataFrame(datos_imagenes)


df_data.head(3)"""
            st.code(body=code, language="python")
            st.write("Creamos la columna sexo donde 0 es masculino, 1 es femenino:")
            code = """ df_data['sexo']= df_data['img'].map(lambda x: x.split("_")[1])"""
            st.code(body=code, language="python")
            st.write("Creamos la columna edad:")
            code = """ df_data['edad'] = [valor.split('_')[0] for valor in df_data['img']]"""
            st.code(body=code, language="python")
            st.write("Cargamos las imagenes del train:")
            code = """ X = []
y = []

tamaño = (200, 200)

for img_path, edad in zip (df_data['ruta'], df_data['edad']):

    imagen_train = Image.open(img_path)

    imagen_resize = imagen_train.resize(tamaño)

    X.append(imagen_resize)

y = np.asarray(df_data['edad'].apply(lambda x : float(x)))
X=np.array(X)"""
            st.code(body=code, language="python")
            st.write("Separamos las variables:")
            code = """ X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

print(f"Conjunto de Train: {X_train.shape, y_train.shape}")
print(f"Conjunto de Test: {X_test.shape, y_test.shape}")"""
            st.write("Generamos el modelo")
            code = """modelo = Sequential()

modelo.add(Conv2D(32, (3,3), input_shape = (200,200,3), activation='relu'))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(2,2))


modelo.add(Conv2D(64,(3,3), activation='relu'))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(2,2))

modelo.add(Conv2D(200,(3,3), activation='relu'))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(2,2))

modelo.add(Flatten())

modelo.add(Dense(200, activation='relu'))
modelo.add(Dropout(0.5))

modelo.add(Dense(200, activation='relu'))
modelo.add(Dropout(0.5))

modelo.add(Dense(64, activation = "relu"))
modelo.add(Dropout(0.5))

modelo.add(Dense(1, activation='linear'))

modelo.summary()"""
            st.code(body=code, language="python")
            code = """ modelo.compile(loss= 'mean_squared_error', optimizer='adam', metrics= ['mae'])
            history = modelo.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 10, verbose = 1)
            """
            st.code(body=code, language="python")

            pass
        if choice == "Modelo en producción":
            edad()
            pass
        if choice == "Métricas":
            df11 = pd.read_csv(
                "m_regresion/Scores_regresion.csv")

            mse = df11['0'].iloc[0]
            mae = df11['0'].iloc[1]

            st.write(f"MSE: {mse}")
            st.write(f"MAE: {mae}")
            st.write("Error: -17.687942504882812")

            pass
        if choice == "Gráficas":
            tab1, tab2, tab3, tab4 = st.tabs(
                ["Mae", "Diagrama de dispersión", "Función de pérdida", "Performance del modelo de regresión"])

            with tab1:
                gra1 = Image.open(
                    "m_regresion/mae.jpeg")
                st.image(image=gra1,
                         caption="Mae",
                         use_column_width=True)
            with tab2:
                gra2 = Image.open(
                    "m_regresion/dispersion.jpeg")
                st.image(image=gra2,
                         caption="Diagrama de dispersión",
                         use_column_width=True)
            with tab3:
                gra3 = Image.open(
                    "m_regresion/perdida.jpeg")
                st.image(image=gra3,
                         caption="Función de pérdida",
                         use_column_width=True)
            with tab4:
                gra4 = Image.open(
                    "m_regresion/performance.jpeg")
                st.image(image=gra4,
                         caption="Performance del modelo de regresión",
                         use_column_width=True)


if __name__ == "__main__":
    intro()
