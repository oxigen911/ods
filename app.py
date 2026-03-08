import streamlit as st
import joblib
import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# Descargar recursos de NLTK
@st.cache_resource
def descargar_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt', quiet=True)

descargar_nltk()

# Definir la función de preprocesamiento
nltk_stopwords = stopwords.words("spanish")

def text_preprocess(text):
    """
    Función de preprocesamiento de texto.
    Debe estar definida ANTES de cargar el modelo.
    """
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = SnowballStemmer('spanish')
    
    tokens = tokenizer.tokenize(text.lower())
    tokens = [word for word in tokens if word not in nltk_stopwords]
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_ods.joblib")

modelo = cargar_modelo()

# Diccionario de ODS
ODS_NOMBRES = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas"
}

# Interfaz de usuario
st.title("Clasificador de Objetivos de Desarrollo Sostenible")
st.markdown("---")

# Ejemplos
ejemplos = [
    "Programas de alfabetización para adultos en comunidades rurales",
    "Acceso universal a servicios de salud y vacunación",
    "Instalación de paneles solares para energía limpia",
    "Políticas de igualdad salarial entre hombres y mujeres",
    "Reducción de emisiones de gases de efecto invernadero"
]

col1, col2 = st.columns([3, 1])
with col1:
    ejemplo_idx = st.selectbox("Usar ejemplo:", ["Escribe tu propio texto..."] + ejemplos)
with col2:
    st.write("")
    st.write("")

if ejemplo_idx != "Escribe tu propio texto...":
    texto_inicial = ejemplo_idx
else:
    texto_inicial = ""

texto = st.text_area(
    "Ingrese un texto relacionado con desarrollo sostenible:",
    value=texto_inicial,
    height=120,
    placeholder="Ejemplo: 'Reducción de la pobreza extrema mediante transferencias monetarias'"
)

if st.button("Clasificar", type="primary"):
    if texto.strip():
        # Realizar predicción
        pred = modelo.predict([texto])
        prob = modelo.predict_proba([texto])
        
        ods_predicho = int(pred[0])
        confianza = max(prob[0]) * 100
        
        # Mostrar resultado
        st.markdown("---")
        st.success(f"### ODS Predicho: **{ods_predicho}**")
        st.info(f"**{ODS_NOMBRES[ods_predicho]}**")
        st.metric("Confianza", f"{confianza:.2f}%")
        
        # Top 3
        st.markdown("#### Top 3 predicciones:")
        clases = modelo.classes_
        probabilidades = prob[0]
        
        # Ordenar por probabilidad
        indices_ordenados = probabilidades.argsort()[::-1][:3]
        
        cols = st.columns(3)
        for i, idx in enumerate(indices_ordenados):
            ods = int(clases[idx])
            prob_ods = probabilidades[idx] * 100
            
            with cols[i]:
                st.metric(
                    label=f"#{i+1} - ODS {ods}",
                    value=f"{prob_ods:.1f}%",
                    delta=ODS_NOMBRES[ods]
                )
    else:
        st.warning("Por favor ingrese un texto")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p><strong>Microproyecto 2 - Machine Learning No Supervisado - Alberto Zapata/Sebastian Tapias</strong></p>
</div>
""", unsafe_allow_html=True)
