import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Cargar el modelo
model = YOLO('model/modelo_entrenado.pt')

# Función para detectar PPE
def detect_ppe(image):
    # Ejecutar predicción directamente con la imagen PIL
    results = model(image)  # Aquí está el cambio clave

    # Obtener clases detectadas
    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]

    # PPE que esperamos detectar
    etiquetas = ['persona', 'casco', 'guantes', 'chaleco']
    detecciones = {etiqueta: (etiqueta in detected_classes) for etiqueta in etiquetas}

    # Crear alerta
    faltantes = [k for k, v in detecciones.items() if k != 'persona' and not v]
    if faltantes:
        alerta = f"⚠️ Alerta SISO: falta {' - '.join(faltantes)}"
    else:
        alerta = "✅ Todo el equipo de protección está presente"

    return {"detected": detecciones, "alert": alerta, "results": results}

# Streamlit UI
st.title('Detección de PPE en Imágenes')

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)

    resultado = detect_ppe(image)

    st.subheader("Resultado de la Detección")
    st.write(resultado['alert'])
    for item, presente in resultado['detected'].items():
        st.write(f"{item}: {'✅' if presente else '❌'}")
