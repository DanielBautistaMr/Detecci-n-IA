import streamlit as st

# Título centrado
st.markdown("<h1 style='text-align: center;'>Creacion Proyecto Deteccion de Delitos en Bucaramanga</h1>", unsafe_allow_html=True)

# Introducción
st.write(f"Te presentaremos el caso de estudio desde el analisis hasta la creacion de un modelo de IA para la deteccion de delitos del grupo de Daniel Bautista, Kevin Llanos, Cristian Muñoz ")

# Imagen
st.image("./images/intro.jpeg")
# Código de ejemplo
st.write("# ANALISIS 🧐")


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



st.write("Se conecta Drive con el google colab para poder aceder a los datos subidos del dataset de delitos")


codigo_python = """
from google.colab import drive
drive.mount('/content/drive')
df=pd.read_csv('/content/drive/MyDrive/Delitos proyecto/delitos_bucaramanga.csv')
df
"""
st.code(codigo_python, language="python")

st.write("Se conecta Drive con el google colab para poder aceder a los datos subidos del dataset de barrios")
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

st.write("# PROCESO DE LIMPIEZA")

st.write(""" 


El dataset importado no a sido claramente filtrado y limpiado aqui hay algunas razones para hacerlo: 


1.Minimización de sesgos: La limpieza de datos puede ayudar a identificar y mitigar sesgos en el dataset, lo que es fundamental para obtener resultados justos y equitativos en análisis y modelos. 📉

2.Precisión de los resultados: Eliminar datos incorrectos o inconsistentes mejora la precisión de los análisis y modelos.😑

3.Reducción de ruido: La eliminación de valores atípicos y datos irrelevantes reduce el ruido en los datos. 💥

4.Conformidad con requisitos: Preparar los datos adecuadamente asegura que cumplan con los requisitos técnicos y legales.👨🏿‍⚖️

5.Mejora de la interpretación: Datos limpios facilitan la interpretación de los resultados y la toma de decisiones informadas. 📢

6.Eficiencia computacional: Reduce la carga de procesamiento y acelera la velocidad de análisis y modelado. 🛜

7.Mejora la generalización: Evita el sobreajuste al eliminar datos que pueden confundir a los modelos. ❌

8.Consistencia: Garantiza que las variables tengan el mismo formato y unidad, lo que facilita la comparación. 🙏🏽

9.Confianza en los datos: Incrementa la confianza en los resultados y la credibilidad de los informes. 😸

10.Protección de la privacidad: Elimina información sensible o identificable para proteger la privacidad de los individuos. 🕵🏿‍♀️

11.Facilita la colaboración: Datos limpios son más fáciles de compartir y colaborar en análisis interdisciplinarios. 👨‍👩‍👧

""")




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

st.subheader("Cantidad de delitos 😵")

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


st.write('')


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

st.subheader("Mapa de calor 🔥")

st.write("Se la limpieza de los datos separandolos")
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

st.subheader("Frecuencias Barrios 🏡")

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

st.subheader("Frecuencias Localidad 📟")


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

st.subheader("Frecuencias de Hora 🕐")


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

st.subheader("Frecuencias de Genero 👫")


codigo_python = """
frecuencias_genero = df['GENERO'].value_counts()
frecuencias_genero

plt.figure(figsize=(5, 5))
plt.pie(frecuencias_genero, labels=frecuencias_genero.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Frecuencia GENERO')
plt.show()
"""
st.code(codigo_python, language="python")

st.write('')
st.image("./images/grafica9.png")
st.write('')

st.subheader("Frecuencias de Edad 👴👵")


codigo_python = """
#¿ cuál es la edad más afectada?
frecuencias_edad = df['EDAD'].value_counts()
frecuencias_edad

df.EDAD.replace('SIN REGISTRO', np.nan ,inplace=True,regex=True)
#Reemplaza los valores sin registro por valor NAN

df['EDAD'].fillna(30, inplace=True)
df['EDAD']=df['EDAD'].astype(int)
df.isna().sum()

# Define los rangos de edades y la lista de etiquetas para el gráfico

bins = [-np.inf, 6, 12, 19, 26,60,np.inf]
names = ['PRIMERA INFANCIA','INFANCIA','ADOLECENCIA','JUVENTUD','ADULTEZ','PERSONA MAYOR']

df['RangoEdad'] = pd.cut(df['EDAD'], bins, labels=names)

cantidadporrango=df['RangoEdad'].value_counts()

# Crea el gráfico de pastel utilizando los rango de edades
plt.figure(figsize=(13, 12))
plt.pie(cantidadporrango, labels=cantidadporrango.index, autopct='%1.1f%%', pctdistance=0.8, startangle=140, labeldistance=1.05)
plt.axis('equal')
plt.title('Frecuencia de Edades')
plt.show()

#De este grafico se concluye que entre mas edad se tiene mas problabilidad hay de que sea victima
#de un delito
"""
st.code(codigo_python, language="python")

st.write('')
st.image("./images/graficar10.png")
st.write('')

st.subheader("Guardar el CSV 💾")


codigo_python = """
# Esta línea de código guarda el DataFrame 'df' en un archivo CSV en la ubicación especificada.
# Utiliza el método 'to_csv()' para exportar los datos del DataFrame a un archivo CSV.
# La ruta '/content/drive/MyDrive/Delitos proyecto/Delito Bucaramanga_preprocesar.csv' especifica la ubicación
# y el nombre del archivo CSV en el que se guardarán los datos.
df.to_csv('/content/drive/MyDrive/Delitos proyecto/Delito Bucaramanga_preprocesar.csv')
"""
st.write('')

st.code(codigo_python, language="python")

st.write('')




st.write("#  CONCLUSIONES 🦾 ")


st.markdown("1. Los delitos en Bucaramanga los ultimos años han aumentado, en 2020 hubo una disminución pero se asume a que fue debido a la pandemia y ademas en 2023 se ve que pocos delitos ya que la base de datos de donde sacamos la informacion solo la toma hasta julio. ")

st.markdown("2. El barrio donde mas hay delitos es en el centro con una gran diferencia de las demas seguido de cabecera del llano. ")

st.markdown("3. El delito mas usual es el hurto a personas o delitos contra el patrimonio de economico. ")

st.markdown("4. El genero que mas se ve afectado por los delitos en el femenino, pero aun asi no se genera mucha diferencia entre los masculinos. ")

st.markdown("5. Con el 58,6% el rango de edades más afectadas es la adultez, entre 27-59 años. ")

st.markdown("6. Entre mas edad se tiene mas problabilidad hay de que sea victima de un delito.")

st.markdown("7. El arma más utilizada es la blanca cortopunzante, la mayoria de delitos se cometen a las 12pm de la noche y las victimas normalmente van a pie. ")

st.markdown("8. No existe una gran diferencia entre los que van en motocicleta a los que van en un vehiculo. ")


st.write("#  PREPROCESSAMIENTO 🤔")

st.write('Se muestra el modelo escogido y su funcionamiento como el codigo ')
st.write('Librerias usadas: ')

codigo_python = """

from sklearn.metrics import confusion_matrix

from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn import tree
"""
st.code(codigo_python, language="python")

st.subheader('Creacion del modeloBA')

st.write("Este modelo de bosque, que opera como un conjunto de árboles de decisión, es una poderosa herramienta de aprendizaje automático que utiliza múltiples estimadores para tomar decisiones más precisas y robustas. Cada árbol en el bosque emite su propia predicción y, finalmente, se combina para obtener un resultado final. Esto hace que el modelo sea resistente al sobreajuste y muy adecuado para tareas de clasificación y regresión. Además, el modelo puede proporcionar información sobre la importancia relativa de las características utilizadas en las predicciones, lo que nos permite entender mejor cómo se toman las decisiones. En esta aplicación, exploraremos cómo este modelo de bosque se aplica a su conjunto de datos y cómo sus características influyen en las predicciones resultantes.")

codigo_python = """

#Defino el algoritmo a utilizar
modeloBA= RandomForestClassifier(random_state=0)
#Entreno el modelo
modeloBA.fit(X_train, y_train)



#accuracy del set de entrenamiento

modeloBA.score(X_train,y_train)*100

98.79846418215014


modeloBA.score(X_test,y_test)*100

65.56760820534852

"""
st.code(codigo_python, language="python")


st.subheader('Matriz')

codigo_python = """
#confusion_matrix con los datos de prueba
y_predict=modeloBA.predict(X_test)
print(y_test.head(20))
print(pd.DataFrame(y_predict).head(20))

#matrix de confusión para analizar los errores de predicción
matrix=confusion_matrix(y_test,y_predict,labels=modeloBA.classes_)
displaymatrix=ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=modeloBA.classes_)
displaymatrix.plot(xticks_rotation='vertical')

#confusion_matrix con los datos de prueba
y_predict=modeloBA.predict(X_test)
print(y_test.head(20))
print(pd.DataFrame(y_predict).head(20))

"""
st.code(codigo_python, language="python")

st.write('')
st.image("./images/graficar11.png")
st.write('Se guardo el modelo .bin')
codigo_python = """
jb.dump(modeloBA,"/content/drive/MyDrive/Delitos proyecto/modeloBA.bin",compress=True)
"""
st.code(codigo_python, language="python")




st.subheader("¿Por que usar el modeloBA? ❤️")

st.write("""Se escogio debido a que su accuracy nos dio mayor exactitud a comparacion de otros modelo sus resultados fueron mayores.

Otros motivos son porque en general es bueno para predecir cosas con precisión, incluso cuando tenemos muchos datos para mirar. Además, es bueno para tratar con datos desequilibrados y no exagerar las predicciones.""")


st.write("#  MODELO EN FUNCIONAMIENTO 🤖 ")

st.write('El siguiente video muestra el video de la IA usando el modeloBA escogido despues del preprocessamiento: ')

# Cargar el video desde el sistema local
video_file = open("./images/vide.mov", "rb")
video_bytes = video_file.read()

# Mostrar el video
st.video(video_bytes)


st.write("Como podemos ver el modelo esta funcionando de forma perfecta nos imprime una grafica en donde podemos ver de manera visual la probabilidad de sufrir un delito")
