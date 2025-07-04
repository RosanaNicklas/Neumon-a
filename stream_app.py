import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import datetime

# Configuración de la página
st.set_page_config(
    page_title="Detector de Neumonía",
    page_icon="🩺",
    layout="wide"
)

# Título y descripción
st.title("🩺 Detector de Neumonía en Radiografías de Tórax")
st.markdown("""
Esta aplicación utiliza un modelo de deep learning basado en EfficientNetV2 para detectar neumonía en radiografías de tórax.
Sube una imagen y el modelo predecirá si es normal o presenta signos de neumonía.
""")

# Cargar el modelo (con caché para no cargarlo repetidamente)
@st.cache_resource
def load_model():
    # Aquí debes reemplazar con la ruta a tu modelo guardado
    model_path = 'pneumonia_effnet.h5'  # Cambia esto por tu modelo
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error("Modelo no encontrado. Por favor asegúrate de tener el archivo 'pneumonia_effnet.h5' en el directorio.")
        return None

model = load_model()

# Función para preprocesar la imagen
def preprocess_image(img):
    img = img.resize((224, 224))  # Tamaño esperado por el modelo
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalización como en el entrenamiento
    return img_array

# Función para hacer la predicción
def predict(image_array):
    if model is not None:
        prediction = model.predict(image_array)
        prob = float(prediction[0][0])
        if prob > 0.5:
            return "PNEUMONIA", prob
        else:
            return "NORMAL", 1 - prob
    return None, None

# Sección para subir imagen
uploaded_file = st.file_uploader(
    "Sube una radiografía de tórax (formato JPG o PNG)",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # Mostrar la imagen subida
    img = Image.open(uploaded_file)
    st.image(img, caption="Imagen subida", width=300)
    
    # Preprocesar y predecir
    img_array = preprocess_image(img)
    
    with st.spinner('Analizando imagen...'):
        label, confidence = predict(img_array)
    
    if label is not None:
        st.subheader("Resultado:")
        
        if label == "PNEUMONIA":
            st.error(f"🔴 **Predicción:** {label} (confianza: {confidence:.2%})")
            st.warning("""
            **Nota importante:**  
            Esta predicción no sustituye el diagnóstico médico. 
            Si el resultado indica neumonía, por favor consulta con un profesional de la salud.
            """)
        else:
            st.success(f"🟢 **Predicción:** {label} (confianza: {confidence:.2%})")
        
        # Mostrar barra de confianza
        st.progress(float(confidence))
        st.caption(f"Confianza de la predicción: {confidence:.2%}")

# Información adicional en la barra lateral
st.sidebar.title("Acerca de")
st.sidebar.info("""
**Modelo utilizado:** EfficientNetV2B0  
**Precisión en pruebas:** ~90-94%  
**Dataset:** Chest X-Ray Images (Pneumonia)  
**Limitaciones:**  
- Solo funciona con imágenes de rayos X de tórax  
- Puede tener falsos positivos/negativos  
- No distingue entre tipos de neumonía  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Instrucciones:**  
1. Sube una imagen de radiografía de tórax  
2. Espera a que el modelo haga la predicción  
3. Revisa los resultados  
""")

# Nota importante al pie de página
st.markdown("---")
st.caption("""
⚠️ **Aviso importante:** Esta aplicación está destinada únicamente para fines de investigación y demostración. 
No debe utilizarse como herramienta de diagnóstico médico sin la supervisión de profesionales calificados.
""")