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

plantilla_reaccion = """
Reacción del usuario: {reaccion}

Analiza el sentimiento y la preocupación expresada en relación con la noticia.
Clasifica la preocupación principal en una de estas categorías:
- Ambiental
- Social
- Gobernanza
- Riesgo

Si la respuesta es vaga o insuficiente, genera una pregunta de seguimiento para profundizar en su opinión. Devuelve SOLO LA PREGUNTA.
"""

prompt_reaccion = PromptTemplate(template=plantilla_reaccion, input_variables=["reaccion"])
cadena_reaccion = LLMChain(llm=llm, prompt=prompt_reaccion)

plantilla_perfil = """
Basado en las siguientes respuestas del usuario: {analisis}

Genera un perfil de su interés en temas ESG (Ambiental, Social y Gobernanza) y su aversión al riesgo.

Devuelve una puntuación de 0 a 100 para cada categoría:
Ambiental: [puntuación]
Social: [puntuación]
Gobernanza: [puntuación]
Riesgo: [puntuación]
"""

prompt_perfil = PromptTemplate(template=plantilla_perfil, input_variables=["analisis"])
cadena_perfil = LLMChain(llm=llm, prompt=prompt_perfil)

if "historial" not in st.session_state:
    st.session_state.historial = []
    st.session_state.contador = 0
    st.session_state.reacciones = []

st.title("Chatbot de Noticias e Inversión")

for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["tipo"]):
        st.write(mensaje["contenido"])

if st.session_state.contador < len(noticias):
    noticia = noticias[st.session_state.contador]
    with st.chat_message("bot", avatar="🤖"):
        st.write(f"¿Qué opinas sobre esta noticia? {noticia}")
    st.session_state.historial.append({"tipo": "bot", "contenido": noticia})
    
    user_input = st.chat_input("Escribe tu opinión...")
    if user_input:
        st.session_state.historial.append({"tipo": "user", "contenido": user_input})
        st.session_state.reacciones.append(user_input)
        
        analisis_reaccion = cadena_reaccion.run(reaccion=user_input)
        if "INSUFICIENTE" in analisis_reaccion:
            pregunta_seguimiento = analisis_reaccion.replace("INSUFICIENTE", "").strip()
            with st.chat_message("bot", avatar="🤖"):
                st.write(pregunta_seguimiento)
            st.session_state.historial.append({"tipo": "bot", "contenido": pregunta_seguimiento})
        else:
            st.session_state.contador += 1
            st.rerun()
else:
    perfil = cadena_perfil.run(analisis="\n".join(st.session_state.reacciones))
    with st.chat_message("bot", avatar="🤖"):
        st.write(f"**Perfil del usuario:** {perfil}")
    st.session_state.historial.append({"tipo": "bot", "contenido": f"**Perfil del usuario:** {perfil}"})
    
    puntuaciones = {
        "Ambiental": int(re.search(r"Ambiental: (\d+)", perfil).group(1)),
        "Social": int(re.search(r"Social: (\d+)", perfil).group(1)),
        "Gobernanza": int(re.search(r"Gobernanza: (\d+)", perfil).group(1)),
        "Riesgo": int(re.search(r"Riesgo: (\d+)", perfil).group(1)),
    }

    fig, ax = plt.subplots()
    ax.bar(puntuaciones.keys(), puntuaciones.values())
    ax.set_ylabel("Puntuación (0-100)")
    ax.set_title("Perfil del Usuario")
    st.pyplot(fig)
    
    try:
        creds_json = json.loads(st.secrets["gcp_service_account"])
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)
        client = gspread.authorize(creds)
        sheet = client.open('BBDD_RESPUESTAS').sheet1
        fila = st.session_state.reacciones[:] + list(puntuaciones.values())
        sheet.append_row(fila)
        st.success("Respuestas y perfil guardados en Google Sheets.")
    except Exception as e:
        st.error(f"Error al guardar en Google Sheets: {e}")
