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
st.write('')


codigo_python = """
#Se utiliza Seaborn para crear un gráfico de regresión para asi poder ver la tendencia que representa
#la linea verda y la cantidad de delitos por año se representa como puntos morados

sns.regplot(x=cantidadxañosin2023.index,y=cantidadxañosin2023["DESCRIPCION_CONDUCTA"],scatter_kws={"color":"purple", "alpha":0.8},line_kws={"color":"green","alpha":0.8})
"""

st.code(codigo_python, language="python")
st.write('')

st.image("./images/grafico3.png")
st.write('')

st.write("Se hace el calculo con el fin de sacar un mapa de calor")
codigo_python = """
# reasignar el nombre de algunas comunas al valor "CENTRO".

# Diccionario que define los mapeos de nombres originales a los nuevos nombres.
# Las claves representan los nombres originales de las comunas que queremos reemplazar.
# Los valores son el nombre "CENTRO" al que queremos reasignar.

comuna={'SIN REGISTRO':'CENTRO','CORREGIMIENTO 3':'CENTRO','CORREGIMIENTO 2':'CENTRO','CORREGIMIENTO 1':'CENTRO'}

# Usamos el método replace de pandas para realizar el reemplazo de nombres.
# La opción regex=True nos permite hacer un reemplazo basado en expresiones regulares.
# El argumento inplace=True indica que queremos hacer el cambio directamente en el DataFrame original

df.NOM_COM.replace(comuna,regex=True,inplace=True)


# Divide la columna 'localizacion' del DataFrame df en dos nuevas columnas.
# La división se basa en el delimitador ','.
# El argumento expand=True indica que queremos dividir la cadena en varias columnas.
# Renombra las dos columnas resultantes a 'lat' y 'lon'.

df_localizacion=df.localizacion.str.split(',',expand=True)
df_localizacion=df_localizacion.rename(columns={0:'lat',1:'lon'})
df_localizacion=df_localizacion.replace('\[','',regex=True).replace(']','',regex=True).astype(float)


# Muestra los tipos de datos de las columnas del DataFrame df_localizacion.
df_localizacion.dtypes


#concatear el df con las coordenadas
df=pd.concat([df,df_localizacion],axis=1)
df

#Se eliminan las columnas de localizacion y localidad puesto que se vuelven irrelevantes debido a que anteriormente se crearon las columas de latitud y longitud

df.drop(['localizacion','LOCALIDAD'],axis=1,inplace=True)



# Agrupa el DataFrame original, df, por 'NOM_COM', 'lat' y 'lon'.
# Luego, cuenta la cantidad de 'DESCRIPCION_CONDUCTA' para cada grupo,
# lo que proporciona la cantidad de delitos reportados para cada comuna y ubicación.
cantidadComuna = df.groupby(['NOM_COM', 'lat', 'lon'])['DESCRIPCION_CONDUCTA'].count().to_frame()

# Extrae el nombre de la comuna (NOM_COM), latitud (lat) y longitud (lon) desde el índice
# multindex y los asigna como columnas separadas.
cantidadComuna['NOM_COM'] = cantidadComuna.index.get_level_values(0)
cantidadComuna['lat'] = cantidadComuna.index.get_level_values(1)
cantidadComuna['lon'] = cantidadComuna.index.get_level_values(2)

# Restablece el índice del DataFrame para que el índice sea numérico
cantidadComuna = cantidadComuna.reset_index(drop=True)
cantidadComuna
"""

st.code(codigo_python, language="python")

st.write('')


st.write("Mapa de calor: ")
codigo_python = """
#Se muestra el mapa de densidad, respecto a las coordenadas y la cantidad de delitos por cada una de estas, esto ayuda a ver graficamente donde se han cometidos los delitos registrados en el archivo csv

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

st.write('')
st.image("./images/aa.png")
st.write('')


st.write("Comienza el analisis y la creacion de nuevas preguntas y graficas")
codigo_python = """
#Se filta los barrios por la cantidad de delitos para quedarse solamente los que tienen mas de 1000 delitos
#esto se hace para poder centrarte en los barrios más comunes o significativos.

frecuencias_barrios_filtradas = frecuencias_barrios[frecuencias_barrios >= 1000]

plt.figure(figsize=(10, 6))  # Tamaño de la figura
frecuencias_barrios_filtradas.plot(kind='barh', color='skyblue')  # Tipo de gráfico y color de las barras
plt.title('Frecuencia de Barrios en los Registros (Filtrados)')  # Título del gráfico
plt.xlabel('Frecuencia')  # Etiqueta del eje x
plt.ylabel('Barrios')  # Etiqueta del eje y
plt.tight_layout()  # Ajustar el diseño del gráfico
plt.show()  # Muestra el gráfico

"""
st.code(codigo_python, language="python")

st.write('')
st.image("./images/grafico4.png")
st.write('')

codigo_python = """
# Agrupa el DataFrame original, df, por la columna 'BARRIOS_HECHO'.
# Luego, cuenta la cantidad de 'DESCRIPCION_CONDUCTA' para cada barrio,
# proporcionando el número de delitos reportados para cada uno.


cantidadbarrio=df.groupby(df["BARRIOS_HECHO"])["DESCRIPCION_CONDUCTA"].count().to_frame()
cantidadbarrio

df.isnull().sum()


#Muestra la cantidad de datos faltantes por categoria

#Se cuenta la cantidad sucesos por delito

frecuencias_delito = df['DELITO_SOLO'].value_counts()
frecuencias_delito

frecuencias_delito.index

#Se grafica la frecuencia con la que se repite un delito para poder vizualizarlos mejor y asi poder
#sacar mejores conclusiones

plt.figure(figsize=(19, 19))
frecuencias_delito.plot(kind='barh', color='skyblue')
plt.title('Frecuencia de delito')
plt.xlabel('Frecuencia')
plt.ylabel('Delito')
plt.tight_layout()
plt.show()

#Las conclusiones que se sacas son las siguientes:
#La mayoria de delitos son de hurto personal
#Hay muchos delitos los cuales ocurren tan poco que al momento
#de predecir su probabilidad no van a variar tanto haciendo que no tenga peso



"""
st.code(codigo_python, language="python")

st.write('')
st.image("./images/grafico5.png")
st.write('')


codigo_python = """
df.NOM_COM.unique()

frecuencias_local = df['NOM_COM'].value_counts()
frecuencias_local
plt.figure(figsize=(20, 10))
plt.pie(frecuencias_local, labels=frecuencias_local.index, autopct='%1.0f%%', startangle=180)
plt.axis('equal')
plt.title('Frecuencia de Localidad')
plt.show()

"""
st.code(codigo_python, language="python")

st.write('')
st.image("./images/grafico6.png")
st.write('')


codigo_python = """
df.RANGO_HORARIO_ORDEN =df.RANGO_HORARIO_ORDEN.astype(int)
df.dtypes
df.RANGO_HORARIO_ORDEN.value_counts()

#Se coloca un rango de horario siendo este
#de 0 a 7 madrugada , de 7 a 13 mañana, de 13 a 19 tarde
#y desde las 19 noche

bins = [-np.inf, 7, 13, 19, np.inf]
names = ['MADRUGADA','MAÑANA','TARDE','NOCHE']

df['rangoHORARIO']=pd.cut(df['RANGO_HORARIO_ORDEN'],bins,labels=names)

cantidadporrangohora=df['rangoHORARIO'].value_counts()


# Crea el gráfico de pastel
plt.figure(figsize=(10, 8))
plt.pie(cantidadporrangohora,labels=cantidadporrangohora.index, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.tight_layout()
plt.title('Frecuencia de hora')
plt.show()

#De este grafico se puede concluir que entre la mañana y la tarde ocurren la mayoria de delitos y
#en la noche es cuando ocurren menos delitos
"""
st.code(codigo_python, language="python")

st.write('')
st.image("./images/grafico7.png")
st.write('')





# ACA VA EL PREPROCESAMIENTO ------------------------------------------ CRISTIAN





#ACA VA EL MODELO ------------------------------------------------------- KEVIN

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
