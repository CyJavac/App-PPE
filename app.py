import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Cargar modelo YOLOv8
@st.cache_resource
def load_model():
    return YOLO("model/modelo.pt")

model = load_model()

st.title("Detector de EPP con YOLOv8")
st.write("Sube una imagen o usa la cámara para verificar el equipo de protección.")

# Opciones: subir o tomar imagen
option = st.radio("Selecciona fuente de imagen:", ('Subir imagen', 'Usar cámara'))

image = None
if option == 'Subir imagen':
    uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif option == 'Usar cámara':
    camera_image = st.camera_input("Toma una foto")
    if camera_image:
        image = Image.open(camera_image)

# Procesar imagen
if image:
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Ejecutar detección
    results = model.predict(image)

    # Mostrar resultados anotados
    annotated = results[0].plot()
    st.image(annotated, caption="Detecciones", use_container_width=True)

    # Analizar clases detectadas
    names = model.names
    detections = [names[int(cls)] for cls in results[0].boxes.cls]
    st.write("Objetos detectados:", ", ".join(set(detections)))

    # Verificación de EPP
    requeridos = {"casco", "guantes", "chaleco"}
    detectados = set(detections)
    faltantes = requeridos - detectados

    if faltantes:
        st.error(f"⚠️ Alerta SISO: Faltan los siguientes elementos: {', '.join(faltantes)}")
    else:
        st.success("✅ Todos los elementos de protección están presentes.")
