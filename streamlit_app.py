import streamlit as st

# Título
st.title("Análisis del Crimen en Bucaramanga")

# Introducción
st.write("""
Visualización y deducciones tras examinar los delitos registrados en Bucaramanga. Equipo:
Andres Felipe Jaimes Rico y Manuel Delgado Mantilla.
""")
lector = st.text_input("Escribe tu nombre aquí", "Nombre del lector")

if lector:
    st.write(f"¡Hola, {lector}! Te agradecemos por interesarte en este estudio sobre criminalidad en Bucaramanga. Tu atención nos ayuda a entender mejor esta situación.")
else:
    st.write("Por favor, ingresa tu nombre en el campo superior para una saludo personalizado.")

# Código de ejemplo
st.write("Comencemos a detallar este estudio:")
st.write("# DESGLOSE")
st.write("Primero, cargamos las bibliotecas necesarias:")
codigo_python1 = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
"""
st.code(codigo_python1, language="python")
st.write("##A continuación, traemos nuestros datos. Estos son proporcionados por una autoridad oficial desde el siguiente enlace: [Enlace a datos](https://www.datos.gov.co/Seguridad-y-Defensa/92-Delitos-en-Bucaramanga-enero-2016-a-julio-de-20/x46e-abhz)")
st.code(codigo_python1, language="python")
st.write("Para el procesamiento de los datos, realizamos algunos ajustes como combinar columnas, eliminar columnas innecesarias, entre otros.")
codigo_python2 = """
df = pd.merge(df, dfbarrios, on="NOM_COM")
df.drop(['Unnamed: 0', 'loc'], axis=1, inplace=True)
df['FECHA_COMPLETA'] = df["FECHA_HECHO"] + ' ' + df["HORA_HECHO"]
cantidadaño = df.groupby(df["FECHA_HECHO"].dt.year)["DESCRIPCION_CONDUCTA"].count().to_frame()
"""
st.code(codigo_python2, language="python")

# Gráficas
st.write("Las representaciones visuales nos permiten comprender mejor los datos. Veamos algunas gráficas:")

codigo_python3 = """
fig, ax = plt.subplots()
ax.bar(cantidadaño.index, cantidadaño["DESCRIPCION_CONDUCTA"])
"""
st.code(codigo_python3, language="python")

st.write("Aquí se visualiza la relación entre la cantidad de incidentes y el año.")

st.code(codigo_python3, language="python")

st.write("Intentamos identificar la relación entre dos variables utilizando regresión:")

codigo_python4 = """
sns.regplot(x=cantidadxañosin2023.index, y=cantidadxañosin2023["DESCRIPCION_CONDUCTA"], scatter_kws={"color": "purple", "alpha": 0.8}, line_kws={"color": "green", "alpha": 0.8})
"""
st.code(codigo_python4, language="python")

st.write("Ahora visualizamos un mapa de calor basado en coordenadas que obtuvimos tras procesar nuestros datos:")

codigo_python5 = """
fig = px.density_mapbox(cantidadComuna, lat='lat', lon='lon', z='DESCRIPCION_CONDUCTA', radius=50, hover_name='NOM_COM', color_continuous_scale='rainbow', center=dict(lat=7.12539, lon=-73.1198), zoom=12, mapbox_style='open-street-map')
fig.show()
"""
st.code(codigo_python5, language="python")

st.write("Otros gráficos y análisis ... [continúa con otros gráficos y códigos según sea necesario]")

# Conclusiones

st.markdown("# 🚀 Reflexiones Finales 🎨")
st.markdown("1. 🏙️ El epicentro: El centro de la ciudad es el principal foco de delitos, probablemente debido a su dinámica comercial y la insuficiente presencia policial en zonas residenciales cercanas.")
st.markdown("2. 🌟 Ubicación clave y vulnerabilidad: La elevada incidencia de delitos en el centro podría estar relacionada con su situación geográfica y la cercanía a zonas con menor vigilancia. Un reto en materia de seguridad.")
st.markdown("3. 💼 Predominancia de delitos no sexuales: Lamentablemente, los delitos no sexuales son más comunes que los sexuales. ¿Qué medidas pueden adoptarse?")
st.markdown("4. 🌞🌙 Momentos del delito: La criminalidad tiene sus picos en horas de la mañana y al amanecer, siendo más variable por la tarde y noche.")
st.markdown("5. 👶👴 Edad y criminalidad: Los delitos afectan más a los adultos, siendo los más jóvenes quienes menos riesgos enfrentan.")
st.markdown("6. 🏰 Estratos y delitos: Resulta sorprendente que zonas de alto estrato, como Cabecera del Llano, presenten índices similares a zonas de menor estrato.")
st.markdown("7. 🚶‍♀️ Cautela al caminar: Desplazarse a pie por ciertas zonas puede tener sus riesgos. ¡Cuidado!")
st.markdown("En conclusión, los datos nos revelan la urgencia de estrategias innovadoras para disminuir la tasa de delitos, proteger a los habitantes y conservar la belleza y seguridad de nuestra ciudad. ¡Juntos por una Bucaramanga más segura!")

# Comentarios adicionales
st.markdown("Si tienes comentarios o sugerencias sobre nuestro estudio, ¡nos encantaría escucharte!")
comentario = st.text_area("Déjanos tu comentario o sugerencia:", "Escribe aquí...")
if comentario != "Escribe aquí...":
    st.success("¡Gracias por tu comentario!")

# Pie de página
st.write("---")
st.write("🔬 *Análisis de Datos por:* Andres Felipe Jaimes Rico & Manuel Delgado Mantilla")
st.write("📚 *Datos proporcionados por:* [www.datos.gov.co](https://www.datos.gov.co/Seguridad-y-Defensa/92-Delitos-en-Bucaramanga-enero-2016-a-julio-de-20/x46e-abhz)")
