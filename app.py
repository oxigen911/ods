import streamlit as st
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import joblib

# Configuración de la pagina
st.set_page_config(page_title="Clasificador de ODS")

# --- CONFIGURACIÓN DE NLTK ---
nltk_stopwords = set(stopwords.words('spanish'))

def text_preprocess(text):
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = SnowballStemmer('spanish')
    tokens = tokenizer.tokenize(text.lower())
    tokens = [word for word in tokens if word not in nltk_stopwords]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Cargar el pipeline
@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load('modelo_ods.joblib')
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

modelo = cargar_modelo()

ODS_NOMBRES = {
    1: "Fin de la Pobreza",
    2: "Hambre Cero",
    3: "Salud y Bienestar",
    4: "Educación de Calidad",
    5: "Igualdad de Género",
    6: "Agua Limpia y Saneamiento",
    7: "Energía Asequible y No Contaminante",
    8: "Trabajo Decente y Crecimiento Económico",
    9: "Industria, Innovación e Infraestructura",
    10: "Reducción de las Desigualdades",
    11: "Ciudades y Comunidades Sostenibles",
    12: "Producción y Consumo Responsables",
    13: "Acción por el Clima",
    14: "Vida Submarina",
    15: "Vida de Ecosistemas Terrestres",
    16: "Paz, Justicia e Instituciones Sólidas",
    17: "Alianzas para lograr los Objetivos"
}

# Interfaz de usuario
st.title("Clasificador de Objetivos de Desarrollo Sostenible (ODS)")
st.markdown("""
Esta aplicación utiliza un modelo de Machine Learning para identificar a qué ODS pertenece un texto.
""")

# Permitir al usuario ingresar un texto libre
texto_usuario = st.text_area(
    "Ingresa el texto a analizar:",
    placeholder="Ejemplo: Promover el crecimiento económico inclusivo y sostenible, el empleo y el trabajo decente para todos.",
    height=150
)

# Boton para procesar
if st.button("Clasificar Texto", type="primary"):
    if not texto_usuario.strip():
        st.warning("Por favor, ingresa algún texto para poder realizar la predicción.")
    elif modelo is not None:
        # Procesar el texto utilizando el mismo pipeline construido
        # El pipeline se encarga de la vectorización y la predicción automáticamente
        with st.spinner('Procesando...'):
            try:
                prediccion = modelo.predict([texto_usuario])
                ods_detectado = prediccion[0]
                nombre_ods = ODS_NOMBRES.get(int(ods_detectado), "Nombre no encontrado")
                st.success(f"### Resultado: ODS {ods_detectado} - {nombre_ods}")
                st.info(f"El texto ha sido clasificado dentro del Objetivo {ods_detectado}.")
                
            except Exception as e:
                st.error(f"Hubo un error al procesar el texto: {e}")
    else:
        st.error("El modelo no está disponible.")


st.divider()
st.caption("Proyecto de Clasificación de ODS - Desarrollado por Alberto Zapata y Sebastian Tapias")








