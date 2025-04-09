import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from langchain import LLMChain, PromptTemplate
from langchain_groq import ChatGroq
import os
import re
import json
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

# Configurar el modelo LLM
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

noticias = [
    "Repsol, entre las 50 empresas que más responsabilidad histórica tienen en el calentamiento global",
    "Amancio Ortega crea un fondo de 100 millones de euros para los afectados de la dana",
    "Freshly Cosmetics despide a 52 empleados en Reus, el 18% de la plantilla",
    "Wall Street y los mercados globales caen ante la incertidumbre por la guerra comercial y el temor a una recesión",
    "El mercado de criptomonedas se desploma: Bitcoin cae a 80.000 dólares, las altcoins se hunden en medio de una frenética liquidación",
    "Granada retrasa seis meses el inicio de la Zona de Bajas Emisiones, previsto hasta ahora para abril",
    "McDonald's donará a la Fundación Ronald McDonald todas las ganancias por ventas del Big Mac del 6 de diciembre",
    "El Gobierno autoriza a altos cargos públicos a irse a Indra, Escribano, CEOE, Barceló, Iberdrola o Airbus",
    "Las aportaciones a los planes de pensiones caen 10.000 millones en los últimos cuatro años",
]

if "historial" not in st.session_state:
    st.session_state.historial = []
    st.session_state.contador = 0
    st.session_state.reacciones = []
    st.session_state.mostrada_noticia = False
    st.session_state.preocupacion = 50  # Valor por defecto para la barra deslizante

st.title("Chatbot de Análisis de Sentimiento")

for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["tipo"]):
        st.write(mensaje["contenido"])

if st.session_state.contador < len(noticias):
    if not st.session_state.mostrada_noticia:
        noticia = noticias[st.session_state.contador]
        with st.chat_message("bot", avatar="🤖"):
            st.write(f"¿Qué nivel de preocupación tienes sobre esta noticia? {noticia}")
        st.session_state.historial.append({"tipo": "bot", "contenido": noticia})
        st.session_state.mostrada_noticia = True
    
    preocupacion = st.slider(
        "Nivel de preocupación (0: Nada preocupado - 100: Muy preocupado)",
        min_value=0,
        max_value=100,
        step=1,
        value=st.session_state.preocupacion  # Restablecer al valor predeterminado
    )

    if st.button("Enviar respuesta"):
        st.session_state.historial.append({"tipo": "user", "contenido": f"Preocupación: {preocupacion}"})
        st.session_state.reacciones.append(preocupacion)
        st.session_state.contador += 1
        st.session_state.mostrada_noticia = False
        st.session_state.preocupacion = 50  # Reiniciar la barra
        st.rerun()
else:
    perfil = {
        "Ambiental": st.session_state.reacciones[0] if len(st.session_state.reacciones) > 0 else 0,
        "Social": st.session_state.reacciones[1] if len(st.session_state.reacciones) > 1 else 0,
        "Gobernanza": st.session_state.reacciones[2] if len(st.session_state.reacciones) > 2 else 0,
        "Riesgo": st.session_state.reacciones[3] if len(st.session_state.reacciones) > 3 else 0,
    }
    
    with st.chat_message("bot", avatar="🤖"):
        st.write(f"**Perfil del inversor:** {perfil}")
    st.session_state.historial.append({"tipo": "bot", "contenido": f"**Perfil del inversor:** {perfil}"})
    
    # Crear gráfico de barras
    fig, ax = plt.subplots()
    ax.bar(perfil.keys(), perfil.values())
    ax.set_ylabel("Puntuación (0-100)")
    ax.set_title("Perfil del Inversor")
    st.pyplot(fig)
    
    try:
        creds_json_str = st.secrets["gcp_service_account"]
        creds_json = json.loads(creds_json_str)
    except Exception as e:
        st.error(f"Error al cargar las credenciales: {e}")
        st.stop()
    
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)
    client = gspread.authorize(creds)
    sheet = client.open('BBDD_RESPUESTAS').sheet1
    fila = st.session_state.reacciones[:]
    sheet.append_row(fila)
    st.success("Respuestas y perfil guardados en Google Sheets en una misma fila.")
