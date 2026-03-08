import streamlit as st
import joblib

modelo = joblib.load("modelo_ods.joblib")

st.title("Clasificador de Objetivos de Desarrollo Sostenible")

texto = st.text_area("Ingrese un texto")

if st.button("Clasificar"):
    
    pred = modelo.predict([texto])
    
    st.write("ODS predicho:", pred[0])