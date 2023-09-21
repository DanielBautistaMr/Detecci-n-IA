import streamlit as st
# Título
st.title("Creacion Proyecto Deteccion de Delitos Bucaramanga")
# Introducción
st.write(f"Te presentaremos el caso de estudio desde el analisis hasta la creacion de un modelo de IA para la deteccion de delitos del grupo de Daniel Bautista, Kevin Llanos, Cristian Muñoz ")

# Imagen
st.image("./images/intro.jpeg")
# Código de ejemplo
st.write("# ANALISIS")


st.write("Todo los datos salieron del siguiente DataSet: https://www.datos.gov.co/Seguridad-y-Defensa/92-Delitos-en-Bucaramanga-enero-2016-a-julio-de-20/x46e-abhz")


codigo_python = """
#TRATAMIENTO DE DATOS
import pandas as pd
import numpy as np

#SISTEMA OPERATIVO
import os

#GRAFICO
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

#LEER ARCHIVOS DE WEB
import urllib

#MAPA DE CALOR

import plotly.express as px
"""

st.code(codigo_python, language="python")


st.write("""Se conecta Drive con el google colab para poder aceder a los datos subidos en 

1.   Elemento de la lista
2.   Elemento de la lista

este""")
codigo_python = """
from google.colab import drive
drive.mount('/content/drive')
df=pd.read_csv('/content/drive/MyDrive/Delitos proyecto/delitos_bucaramanga.csv')
df
"""
st.code(codigo_python, language="python")

st.write("""Se conecta Drive con el google colab para poder aceder a los datos subidos en 

1.   Elemento de la lista
2.   Elemento de la lista

este""")
codigo_python = """
from google.colab import drive
drive.mount('/content/drive')
df=pd.read_csv('/content/drive/MyDrive/Delitos proyecto/delitos_bucaramanga.csv')
df

dfbarrios=pd.read_csv('https://raw.githubusercontent.com/adiacla/bigdata/master/ubicacion_comuna.csv',encoding='utf-8')
dfbarrios

#Archivo compartido para poder sectorizar por latitud y longitud para poder graficar en el mapa
"""
st.code(codigo_python, language="python")
#G R A F I C A C I O N
st.write(" Las graficas nos muestran los datos de una manera mas entendible y este es el momento de usarlos:")
codigo_python = """
#Graficamos la informacion de arriba en barras
fig,ax,=plt.subplots()
ax.bar(cantidadaño.index,cantidadaño["DESCRIPCION_CONDUCTA"])
"""
st.code(codigo_python, language="python")


st.write("Se arreglan los datasets y se limpian: ")
codigo_python = """
# Se muestra cuantas veces se repite un barrio en el data frame para comprender la distribución de datos en la columna de BARRIOS_HECHOS
df.BARRIOS_HECHO.value_counts()

# Se busca las concidencias en los data frames df y dfbarrios (data frame donde esta la latitud y la longitud)
# para asi fusionarlas en una sola la cual es "NOM_COM"y asi que los dos data frames queden en solo uno
df = pd.merge(df, dfbarrios, on="NOM_COM")

# Se elimina 'Unnamed: 0' y 'loc' del DataFrame 'df'.
# El argumento 'axis=1' especifica que las columnas deben eliminarse en lugar de las filas.
# El argumento 'inplace=True' indica que la operación debe realizarse directamente en el DataFrame 'df'.
df.drop(['Unnamed: 0', 'loc'], axis=1, inplace=True)

# Se crea una nueva columna llamada "FECHA COMPLETA" la cual es la combinacion entre "FECHA_HECHO" y "HORA_HECHO",
# colocando un espacion en blanco entre las dos para asi porder ternerlos es un solo formato
df['FECHA_COMPLETA'] = df["FECHA_HECHO"] + ' ' + df["HORA_HECHO"]

#Se el tipo de dato de la columna "FECHA_HECHO" en datetime

df=df.astype({"FECHA_HECHO":"datetime64[ns]"})

#Se convierte la columna "FECHA_HECHO" en modo datatime, utilizando el metodo 'pd.to_datetime()'
#ademas se utiliza el argumento 'format="DD/MM/YYYY"' para especifircar el formato de la fecha original
#siendo este un dia con dos digitos , mes con dos digitos y el año con 4.

df["FECHA_HECHO"] = pd.to_datetime(df["FECHA_HECHO"], format="DD/MM/YYYY")

#Se calcula la cantidad de delitos por año en la columna "FECHA_HECHO"
# y almacenando los resultados en un DataFrame llamado "cantidadaño"
#.groupby(df["FECHA_HECHO"].dt.year)["DESCRIPCION_CONDUCTA"].count() agrupan los datos en función del año
#y los cuenta, .to_frame() convierte la serie resultante en un DataFrame

cantidadaño=df.groupby(df["FECHA_HECHO"].dt.year)["DESCRIPCION_CONDUCTA"].count().to_frame()
cantidadaño
"""
st.code(codigo_python, language="python")

st.write("Comenzamos graficando los datos para ver sus relaciones y asi analizarlo")
codigo_python = """
#Se hace una grafica de barras a partir de los datos contenidos en "cantidadaño"
#para asi poder analisar mejor los datos y sacar conclusiones
fig,ax,=plt.subplots()
ax.bar(cantidadaño.index,cantidadaño["DESCRIPCION_CONDUCTA"])
ax.set_xlabel("Años")
ax.set_ylabel("Cantidad de Delitos")
plt.show()

#De la grafica sacamos las siguientes conclusiones:
#1. Los delitos tienden a aumentar a medida que pasan los años
#2. En el 2020 ocurrio una baja esto se concluse que ocurrio por la pandemia
#3. El año 2023 se ve tan bajo ya que los datos utilizados solo se toman hasta julio del 2023
"""

st.code(codigo_python, language="python")

st.write('')

st.image("./images/grafico1.png")

codigo_python = """
#Se hace una grafica de lineas para asi poder visualizar mejor la cantidad de delitos atraves de los años
#teniendo en cuenta los meses que en los que se realizaron


ax = cantidadmesxaño.plot(kind="line")
ax.set_xlabel("Año/Mes")
ax.set_ylabel("Cantidad de Delitos")
plt.show()
"""

st.code(codigo_python, language="python")
st.write('')

st.image("./images/grafico2.png")

codigo_python = """
#Se utiliza Seaborn para crear un gráfico de regresión para asi poder ver la tendencia que representa
#la linea verda y la cantidad de delitos por año se representa como puntos morados

sns.regplot(x=cantidadxañosin2023.index,y=cantidadxañosin2023["DESCRIPCION_CONDUCTA"],scatter_kws={"color":"purple", "alpha":0.8},line_kws={"color":"green","alpha":0.8})
"""

st.code(codigo_python, language="python")
st.write('')

st.image("./images/grafico3.png")

st.write("Intentamos encontrar con regresión encontrar relacion entre dos variables:")
codigo_python = """
#Usamos la libreria Seaborn, que pueda mostrar la relacion entre dos variables
# X, representa los años
# Y, representa la cantidad de casos
#line_kws: La apariencia de la regresion lineal
sns.regplot(x=cantidadxañosin2023.index,y=cantidadxañosin2023["DESCRIPCION_CONDUCTA"],scatter_kws={"color":"purple", "alpha":0.8},line_kws={"color":"green","alpha":0.8})
"""
st.code(codigo_python, language="python")
st.image("3grafico.jpg")
#
st.write("Mostramos un mapa de calor, despues de haber modificado nuestro DataFrame para que nos diera las coordenadas:")
codigo_python = """
fig = px.density_mapbox(cantidadComuna, lat = 'lat', lon = 'lon',z='DESCRIPCION_CONDUCTA',
                        radius = 50,
                        hover_name='NOM_COM',
                        color_continuous_scale='rainbow',
                        center = dict(lat = 7.12539, lon = -73.1198),
                        zoom = 12,
                        mapbox_style = 'open-street-map')
fig.show()
"""
st.code(codigo_python, language="python")
st.image("4grafico.jpg")
#
st.write("Mostramos un mapa de calor, despues de haber modificado nuestro DataFrame para que nos diera las coordenadas:")
codigo_python = """
frecuencias_barrios_filtradas = frecuencias_barrios[frecuencias_barrios >= 1000] #filtramos
plt.figure(figsize=(8, 4))  # Tamaño de la figura
frecuencias_barrios_filtradas.plot(kind='barh', color='skyblue')  # Tipo de gráfico y sus colores
plt.title('Registro de frecuencia en Barrios (Filtrados & Organizados)')
plt.xlabel('FRECUENCIA')  # Etiqueta del eje x
plt.ylabel('BARRIOS')  # Etiqueta del eje y
plt.tight_layout()
plt.show()  # Muestra el gráfico
"""
st.code(codigo_python, language="python")
st.image("5grafico.jpg")

#
st.write("Graficos de barras Frecuencias Delito:")
codigo_python = """
# Filtrar los tipos de delito con una cantidad mayor a 400
frecuencias_filtradas = frecuencias_delito[frecuencias_delito > 400]

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))  # Tamaño del gráfico
frecuencias_filtradas.plot(kind='bar', color='red')
plt.title('Cantidad de Delitos por Tipo (Filtrado)')  # Título del gráfico
plt.xlabel('Tipo de Delito')  # Etiqueta del eje x
plt.ylabel('Cantidad')  # Etiqueta del eje y
plt.xticks(rotation=90)  # Rotar las etiquetas del eje x para mayor legibilidad
plt.tight_layout()

# Mostrar el gráfico
plt.show()
"""
st.code(codigo_python, language="python")
st.image("6grafico.png")
#

st.write("Graficos de Pastel: Frecuencia del Genero")
codigo_python = """
plt.figure(figsize=(5, 5))
plt.pie(frecuencias_genero, labels=frecuencias_genero.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Frecuencia GENERO')
plt.show()
"""
st.code(codigo_python, language="python")
st.image("7grafico.png")

st.write("Grafico pastel: Frecuencia de Edades")
codigo_python = """
# Crea el gráfico de pastel
plt.figure(figsize=(13, 12))
plt.pie(cantidadporrango, labels=cantidadporrango.index, autopct='%1.1f%%', pctdistance=0.8, startangle=140, labeldistance=1.05)
plt.axis('equal')
plt.title('Frecuencia de Edades')
plt.show()
"""
st.code(codigo_python, language="python")
st.image("8grafico.png")

st.write("Grafico pastel: Frecuencia de Horario")
codigo_python = """
# Crea el gráfico de pastel
plt.figure(figsize=(13, 12))
plt.pie(cantidadporrango, labels=cantidadporrango.index, autopct='%1.1f%%', pctdistance=0.8, startangle=140, labeldistance=1.05)
plt.axis('equal')
plt.title('Frecuencia de Edades')
plt.show()
"""
st.code(codigo_python, language="python")
st.image("8grafico.png")







# Conclusiones

# Agrega emojis y estilos de fuente personalizados
st.markdown("# 🚀 *Conclusión * 🎨")

st.markdown("1. 🏙️ *El Centro es el Hotspot:* La mayor cantidad de robos ocurre en el corazón de la ciudad, posiblemente debido a su vibrante actividad comercial y la falta de presencia policial en áreas cercanas a los barrios residenciales.")

st.markdown("2. 🌟 *Estratégico y Vulnerable:* Se podría inferir que la concentración de robos en el centro se debe a su ubicación estratégica y a la proximidad de barrios con menor presencia policial. ¡Un desafío para la seguridad!")

st.markdown("3. 💼 *Delitos No Sexuales Dominan:* En el lado oscuro de la estadística, los delitos no sexuales superan en número a los delitos sexuales en la ciudad. ¿Cómo podemos abordar esta variabilidad en la seguridad?")

st.markdown("4. 🌞🌙 *Hora de la Delincuencia:* Los delitos matutinos y madrugadores tienden a tener horarios fijos, mientras que los delitos en la tarde y noche son más impredecibles. ¡La ciudad nunca duerme!")

st.markdown("5. 👶👴 *Edades y Delincuencia:* Los adultos son los más afectados por la delincuencia, mientras que los más pequeños (la primera infancia) experimentan menos problemas. ¡Protejamos a nuestros ciudadanos más jóvenes!")

st.markdown("6. 🏰 *Estrato vs. Delincuencia:* Sorprendentemente, incluso un barrio de alto estrato como Cabecera del Llano comparte índices de delincuencia similares a los de un barrio de estrato más bajo, como El Centro. ¿Dónde radica la igualdad?")

st.markdown("7. 🚶‍♀️ *Caminar con Cuidado:* Caminar por algunas partes de la ciudad puede ser arriesgado. ¡Mantén tus sentidos alerta y tu seguridad en mente!")

st.markdown("En resumen, estos hallazgos sugieren la necesidad de implementar *estrategias creativas y efectivas* para reducir la incidencia de robos, proteger a nuestros ciudadanos y mantener nuestra ciudad hermosa y segura. ¡Sigamos trabajando juntos para un futuro más seguro!")
# Barra de navegación
st.sidebar.title("Navegación")
pagina_actual = st.sidebar.radio("Selecciona una página:", ["Inicio", "Acerca de", "Contacto"])

if pagina_actual == "Inicio":
    st.sidebar.write("Bienvenido a la página de inicio.")
elif pagina_actual == "Acerca de":
    st.sidebar.write("Esta es la página de información acerca de la aplicación.")
elif pagina_actual == "Contacto":
    st.sidebar.write("Puedes ponerte en contacto con nosotros aquí.")
import pandas as pd
# Cargar el DataFrame
df = pd.read_csv('92._Delitos_en_Bucaramanga_enero_2016_a_julio_de_2023.csv')  # Reemplaza 'tu_archivo.csv' con la ruta a tu archivo CSV
info_df = pd.DataFrame({
    'Nombre de la columna': df.columns,
    'No. de valores no nulos': df.count().values,
    'Tipo de datos': df.dtypes.values
})

# Mostrar el resumen en Streamlit
st.title('Información del DataFrame')
st.write('A continuación se muestra la información del DataFrame que utilizamos:')
st.write(info_df)

# p r e p r o c e s a m i e n t o
