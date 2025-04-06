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
    temperature=0.5,
    max_tokens=200,
    timeout=30,
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
Reacción del inversor: {reaccion}

Analiza el sentimiento y la preocupación expresada.  
Clasifica la preocupación principal en una de estas categorías:  
- Ambiental  
- Social  
- Gobernanza  
- Riesgo  

Evalúa si la respuesta es clara y detallada. Debe contener al menos una justificación o explicación. Si solo expresa una opinión sin justificación, devuelve "INSUFICIENTE".

Si la respuesta es insuficiente, genera una pregunta de seguimiento enfocada en la categoría detectada para profundizar en la opinión del inversor. Devuelve SOLO LA PREGUNTA, sin ninguna explicación adicional.
"""

prompt_reaccion = PromptTemplate(template=plantilla_reaccion, input_variables=["reaccion"])
cadena_reaccion = LLMChain(llm=llm, prompt=prompt_reaccion)

plantilla_perfil = """
Análisis de reacciones: {analisis}

Genera un perfil detallado del inversor basado en sus reacciones, enfocándote en los pilares ESG (Ambiental, Social y Gobernanza) y su aversión al riesgo.

Asigna una puntuación de 0 a 100 para cada pilar ESG y para el riesgo, donde 0 indica ninguna preocupación y 100 máxima preocupación o aversión.

Devuelve las 4 puntuaciones en formato: Ambiental: [puntuación], Social: [puntuación], Gobernanza: [puntuación], Riesgo: [puntuación]
"""

prompt_perfil = PromptTemplate(template=plantilla_perfil, input_variables=["analisis"])
cadena_perfil = LLMChain(llm=llm, prompt=prompt_perfil)

if "historial" not in st.session_state:
    st.session_state.historial = []
    st.session_state.contador = 0
    st.session_state.mostrada_noticia = False
    st.session_state.esperando_respuesta = False
    st.session_state.reacciones = []

st.title("Chatbot de Análisis de Sentimiento y Preferencias del Inversor")

# Mostrar historial de la conversación
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["tipo"]):
        st.write(mensaje["contenido"])

# Lógica para mostrar noticias e interactuar con el usuario
if st.session_state.contador < len(noticias):
    if not st.session_state.mostrada_noticia and not st.session_state.esperando_respuesta:
        noticia = noticias[st.session_state.contador]
        with st.chat_message("bot", avatar="🤖"):
            st.write(f"¿Qué opinas sobre esta noticia? {noticia}")
        st.session_state.historial.append({"tipo": "bot", "contenido": noticia})
        st.session_state.mostrada_noticia = True

    user_input = st.chat_input("Escribe tu respuesta aquí...")
    if user_input:
        st.session_state.historial.append({"tipo": "user", "contenido": user_input})
        st.session_state.reacciones.append(user_input)
        
        if st.session_state.esperando_respuesta:
            st.session_state.esperando_respuesta = False
            st.session_state.contador += 1
            st.session_state.mostrada_noticia = False
            st.rerun()
        else:
            analisis_reaccion = cadena_reaccion.run(reaccion=user_input)
            
            if "INSUFICIENTE" in analisis_reaccion:
                # Extraer solo la pregunta (eliminando "INSUFICIENTE" y cualquier texto adicional)
                pregunta_seguimiento = analisis_reaccion.replace("INSUFICIENTE", "").strip()
                # Eliminar saltos de línea y quedarse solo con la primera línea (la pregunta)
                pregunta_seguimiento = pregunta_seguimiento.split("\n")[0].strip()
                
                with st.chat_message("bot", avatar="🤖"):
                    st.write(pregunta_seguimiento)
                st.session_state.historial.append({"tipo": "bot", "contenido": pregunta_seguimiento})
                st.session_state.esperando_respuesta = True
            else:
                with st.chat_message("bot", avatar="🤖"):
                    st.write("Ok, pasemos a la siguiente pregunta.")
                st.session_state.historial.append({"tipo": "bot", "contenido": "Ok, pasemos a la siguiente pregunta."})
                
                st.session_state.contador += 1
                st.session_state.mostrada_noticia = False
                st.session_state.esperando_respuesta = False
                st.rerun()
else:
    analisis_total = "\n".join(st.session_state.reacciones)
    perfil = cadena_perfil.run(analisis=analisis_total)
    with st.chat_message("bot", avatar="🤖"):
        st.write(f"**Perfil del inversor:** {perfil}")
    st.session_state.historial.append({"tipo": "bot", "contenido": f"**Perfil del inversor:** {perfil}"})

    puntuaciones = {
        "Ambiental": int(re.search(r"Ambiental: (\d+)", perfil).group(1)),
        "Social": int(re.search(r"Social: (\d+)", perfil).group(1)),
        "Gobernanza": int(re.search(r"Gobernanza: (\d+)", perfil).group(1)),
        "Riesgo": int(re.search(r"Riesgo: (\d+)", perfil).group(1)),
    }

    categorias = list(puntuaciones.keys())
    valores = list(puntuaciones.values())

    fig, ax = plt.subplots()
    ax.bar(categorias, valores)
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
    fila.extend(valores)
    sheet.append_row(fila)

    st.success("Respuestas y perfil guardados en Google Sheets en una misma fila.")
