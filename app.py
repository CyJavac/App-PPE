import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO  # Importar YOLOv8 desde ultralytics

# Cargar el modelo de IA YOLOv8
model = YOLO('model/yolov8n.pt')  # Cargar el modelo entrenado YOLOv8

# Función para detectar PPE (cascos, guantes, chalecos)
def detect_ppe(image):
    # Redimensionar la imagen a 640x640 píxeles (tamaño requerido por YOLOv8)
    img_resized = cv2.resize(image, (640, 640))  # Ajuste según el modelo YOLOv8
    img_array = np.expand_dims(img_resized / 255.0, axis=0)  # Normalizar la imagen y añadir un batch dimension

    # Inferencia con el modelo
    results = model(img_array)  # Usar el modelo YOLOv8 para hacer la inferencia

    # Extraer resultados de las detecciones
    detected = {}
    labels = ['persona', 'casco', 'guantes', 'chaleco']

    for i, label in enumerate(labels):
        detected[label] = any([result['class'] == i for result in results.xywh[0]])  # Check if class i is detected

    # Generar alerta SISO
    missing = [k for k, v in detected.items() if k != 'persona' and not v]
    if missing:
        alert = f"⚠️ Alerta SISO: falta {' - '.join(missing)}"
    else:
        alert = "✅ Todo el equipo de protección está presente"

    return {"detected": detected, "alert": alert}

# Configuración de la interfaz Streamlit
st.title('Detección de PPE en Imágenes')

# Subir imagen
uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_column_width=True)

    # Convertir la imagen a formato que OpenCV pueda procesar
    img_array = np.array(image)
    if img_array.shape[2] == 4:  # Si tiene canal alfa, eliminarlo
        img_array = img_array[:, :, :3]
    
    # Llamar a la función para detectar PPE
    result = detect_ppe(img_array)

    # Mostrar los resultados de la detección
    st.subheader("Resultado de la Detección")
    st.write(result['alert'])
    for item, status in result['detected'].items():
        st.write(f"{item}: {'✅' if status else '❌'}")
