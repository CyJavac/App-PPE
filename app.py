import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

try:
    import cv2
    print("OpenCV se ha importado correctamente")
except ImportError as e:
    print(f"Error al importar OpenCV: {e}")
    raise


# Cargar el modelo de IA
model = load_model('model/yolov8n.pt')

# Función para detectar PPE (cascos, guantes, chalecos)
def detect_ppe(image):
    # Preprocesar la imagen
    img_resized = cv2.resize(image, (224, 224))  # Ajusta según tu modelo
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    # Inferencia con el modelo
    predictions = model.predict(img_array)[0]
    labels = ['persona', 'casco', 'guantes', 'chaleco']
    detected = {label: bool(round(pred)) for label, pred in zip(labels, predictions)}

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
