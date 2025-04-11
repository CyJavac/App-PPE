import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

# Cargar el modelo
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='model/modelo_entrenado.pt', force_reload=False)

model = load_model()
model.conf = 0.5  # confianza mínima

# Clases esperadas
expected_classes = {'casco', 'guantes', 'chaleco'}

st.title("Detector de Equipos de Protección Personal")
st.write("Sube una imagen o usa tu cámara para verificar si una persona tiene los EPP adecuados.")

# Elegir fuente de imagen
option = st.radio("Selecciona una opción:", ('Subir imagen', 'Usar cámara'))

# Obtener imagen
if option == 'Subir imagen':
    uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif option == 'Usar cámara':
    camera_image = st.camera_input("Toma una foto")
    if camera_image:
        image = Image.open(camera_image)

# Procesar imagen si hay una
if 'image' in locals():
    st.image(image, caption='Imagen cargada', use_container_width=True)

    # Convertir imagen a formato compatible
    img_array = np.array(image)
    results = model(img_array)

    # Mostrar resultados
    results.render()
    st.image(results.ims[0], caption="Resultado con detecciones", use_column_width=True)

    # Obtener clases detectadas
    detected = set([model.names[int(x)] for x in results.xyxy[0][:, -1]])
    st.write("Objetos detectados:", ", ".join(detected))

    # Verificar si falta alguna protección
    missing = expected_classes - detected
    if missing:
        st.error(f"⚠️ Alerta SISO: Faltan los siguientes elementos de protección: {', '.join(missing)}")
    else:
        st.success("✅ Todos los elementos de protección están presentes.")
