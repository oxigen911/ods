import streamlit as st
import joblib

# Configuración de la pagina
st.set_page_config(page_title="Clasificador de ODS")

def text_preprocess(text):
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = SnowballStemmer('spanish')
    tokens = tokenizer.tokenize(text.lower())
    tokens = [word for word in tokens if word not in nltk_stopwords]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Cargar el pipeline
# Usamos cache_resource para que el modelo se cargue una sola vez y la app sea rápida
@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load('modelo_ods.joblib')
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

modelo = cargar_modelo()

# Interfaz de usuario
st.title("Clasificador de Objetivos de Desarrollo Sostenible (ODS)")
st.markdown("""
Esta aplicación utiliza un modelo de Machine Learning para identificar a qué ODS pertenece un texto. 
Escribe una frase, meta o descripción de un proyecto a continuación.
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
                
                # Generar como salida la predicción del ODS correspondiente
                st.success(f"### Resultado: ODS {ods_detectado}")
                
                # Opcional: Mostrar una descripción visual o iconos según el ODS
                st.info(f"El texto ha sido clasificado dentro del Objetivo {ods_detectado}.")
                
            except Exception as e:
                st.error(f"Hubo un error al procesar el texto: {e}")
    else:
        st.error("El modelo no está disponible. Verifica que 'modelo_ods.joblib' esté en el repositorio.")


st.divider()
st.caption("Proyecto de Clasificación de ODS - Desarrollado por Alberto Zapata y Sebastian Tapias")

