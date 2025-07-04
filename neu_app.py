import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps
import os
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuración mejorada de la página
st.set_page_config(
    page_title="NeumoniaScan Pro - Detector Avanzado",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Estilos CSS mejorados
st.markdown("""
<style>
    .header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    .result-card {
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .normal {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .pneumonia {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .confidence-meter {
        height: 25px;
        border-radius: 5px;
        background: linear-gradient(90deg, #f44336 0%, #ff9800 50%, #4caf50 100%);
        margin: 10px 0;
    }
    .upload-box {
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .footer {
        font-size: 0.8rem;
        text-align: center;
        margin-top: 30px;
        color: #7f8c8d;
    }
    @media screen and (max-width: 600px) {
    .result-card { padding: 15px; }
}        
</style>
""", unsafe_allow_html=True)

# Título mejorado
st.markdown('<h1 class="header">🏥 NeumoniaScan Pro</h1>', unsafe_allow_html=True)
st.markdown("""
**Sistema avanzado de detección de neumonía en radiografías de tórax mediante inteligencia artificial**
""")

# Cargar el modelo con verificación mejorada
@st.cache_resource
def load_model():
    try:
        model_path = 'pneumonia_effnet.h5'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            # Verificación básica del modelo
            if len(model.layers) > 0:
                return model
            else:
                st.error("El modelo cargado no es válido.")
                return None
        else:
            st.error("Archivo del modelo no encontrado.")
            return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

model = load_model()
if model:
    model.make_predict_function()

# Función mejorada de preprocesamiento
# Antes de procesar la imagen en tu función

def preprocess_image(img):
    try:
        # Convertir a RGB si es escala de grises
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Mejorar el contraste y normalizar
        img = ImageOps.autocontrast(img)
        img = img.resize((224, 224))
        
        # Convertir a array y preprocesar para EfficientNet
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        st.error(f"Error al procesar la imagen: {str(e)}")
        return None

# Función de predicción mejorada
def predict(image_array):
    try:
        if model is None:
            return None, None, "Modelo no cargado"
            
        predictions = model.predict(image_array)
        pneumonia_prob = float(predictions[0][0])
        normal_prob = 1 - pneumonia_prob
        
        # Determinar resultado y nivel de confianza
        if pneumonia_prob > 0.7:
            return "Neumonía detectada", pneumonia_prob, "Alta confianza"
        elif pneumonia_prob > 0.55:
            return "Posible neumonía", pneumonia_prob, "Confianza moderada"
        elif pneumonia_prob > 0.45:
            return "Incierto - Consulte especialista", pneumonia_prob, "Baja confianza"
        else:
            return "Normal", normal_prob, "Alta confianza"
            
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        return None, None, str(e)

# Visualización de la imagen con análisis
def display_image_analysis(img):
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Imagen original", use_container_width=True)
    
    with col2:
        # Mostrar histograma de la imagen
        plt.figure(figsize=(6, 4))
        hist = np.array(img.convert('L')).flatten()
        plt.hist(hist, bins=50, color='blue', alpha=0.7)
        plt.title("Distribución de pixeles")
        plt.xlabel("Intensidad")
        plt.ylabel("Frecuencia")
        st.pyplot(plt)

# Sección de carga de archivos mejorada
st.markdown("### 📤 Subir radiografía de tórax")
with st.container():
    uploaded_file = st.file_uploader(
        "Seleccione una imagen (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        
        # Mostrar análisis de imagen
        display_image_analysis(img)
        
        # Procesar y predecir
        with st.spinner('Analizando imagen con modelos avanzados...'):
            img_array = preprocess_image(img)
            
            if img_array is not None:
                result, prob, confidence = predict(img_array)
                
                # Mostrar resultados
                st.markdown("### 🔍 Resultados del análisis")
                
                if result is not None:
                    # Determinar clase de resultado
                    result_class = "pneumonia" if "neumonía" in result.lower() else "normal"
                    
                    # Mostrar tarjeta de resultados
                    st.markdown(f"""
                    <div class="result-card {result_class}">
                        <h3>{result}</h3>
                        <p>Nivel de confianza: <strong>{confidence}</strong></p>
                        <div class="confidence-meter" style="width: {prob*100}%; 
                            background-color: {'#4caf50' if result_class == 'normal' else '#f44336'};"></div>
                        <p>Probabilidad: <strong>{prob:.1%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Recomendaciones basadas en resultados
                    if result_class == "pneumonia":
                        st.warning("""
                        **Recomendaciones:**
                        - Consultar urgentemente con un neumólogo
                        - Realizar pruebas complementarias
                        - No automedicarse
                        """)
                    else:
                        st.info("""
                        **Recomendaciones:**
                        - Continuar con revisiones rutinarias
                        - Consultar si aparecen síntomas respiratorios
                        - Mantener seguimiento médico regular
                        """)
                    
                    # Generar reporte simple
                    report = f"""
                    **Reporte de Análisis - NeumoniaScan Pro**
                    Fecha: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
                    
                    **Resultado:** {result}
                    **Probabilidad:** {prob:.1%}
                    **Nivel de confianza:** {confidence}
                    
                    **Observaciones:**
                    Este resultado ha sido generado automáticamente por un sistema de IA
                    y debe ser validado por un profesional médico cualificado.
                    """
                    
                    # Botón para descargar reporte
                    st.download_button(
                        label="📄 Descargar reporte",
                        data=report,
                        file_name=f"reporte_neumonia_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
                
    except Exception as e:
        st.error(f"Error procesando la imagen: {str(e)}")

# Barra lateral con información técnica
with st.sidebar:
    st.markdown("## ℹ️ Información Técnica")
    st.markdown("""
    **Modelo:** EfficientNetV2B0  
    **Precisión:** 92-94%  
    **Dataset:** 5,856 imágenes etiquetadas  
    **Especialidad:** Neumonía bacteriana  
    
    **Limitaciones:**
    - No detecta neumonía viral
    - Puede fallar con imágenes de baja calidad
    - Requiere validación médica
    """)
    
    
    st.markdown("## 📝 Instrucciones")
    st.markdown("""
    1. Suba una radiografía de tórax frontal
    2. Espere el análisis automático
    3. Revise los resultados
    4. Consulte con su médico
    """)

# Pie de página profesional
st.markdown("""
<div class="footer">
    NeumoniaScan Pro v3.2 • Sistema de apoyo diagnóstico • 
    No sustituye la evaluación médica profesional • © 2023
</div>
""", unsafe_allow_html=True)