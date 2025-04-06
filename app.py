import streamlit as st
from langchain import LLMChain, PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import re

load_dotenv()

# Configuración avanzada del LLM
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Datos de contexto ESG
TEMAS_ESG = {
    "Ambiental": [
        "emisiones carbono", "energías renovables", "gestión residuos",
        "biodiversidad", "huella ecológica"
    ],
    "Social": [
        "derechos empleados", "diversidad inclusión", "relación comunidades",
        "seguridad producto", "derechos humanos"
    ],
    "Gobernanza": [
        "ética corporativa", "transparencia", "estructura consejo",
        "remuneración ejecutivos", "políticas anticorrupción"
    ],
    "Riesgo": [
        "volatilidad mercado", "regulaciones", "disrupción tecnológica",
        "continuidad negocio", "crisis financieras"
    ]
}

# Sistema de conversación por pasos
ESTADOS_CONVERSACION = {
    "INICIO": {
        "mensaje": "Hola! Soy tu asesor ESG. Analizaré tu perfil de inversión mediante 5 noticias. ¿Preparado para comenzar?",
        "transicion": "PRIMERA_NOTICIA"
    },
    "PRIMERA_NOTICIA": {
        "noticia": "Repsol entre las 50 empresas con mayor responsabilidad histórica en el calentamiento global",
        "pregunta_base": "¿Qué opinas sobre esta noticia?",
        "profundizar": {
            "Ambiental": "¿Cómo valoras que una empresa energética gestione su transición ecológica?",
            "Social": "¿Crees que deberían compensar a las comunidades afectadas por su impacto ambiental?",
            "Gobernanza": "¿Qué medidas de transparencia esperarías en su reporting de sostenibilidad?"
        }
    },
    # ... (otros estados para cada noticia)
}

# Plantillas de conversación dinámicas
plantilla_dialogo = """
Como analista ESG experto, estás manteniendo una conversación natural con un inversor sobre:

Noticia actual: {noticia_actual}
Historial reciente: {historial_chat}

Objetivo: {objetivo_actual}

Instrucciones:
1. Mantén un tono profesional pero cercano
2. Adapta tu respuesta al último mensaje del usuario
3. Haz máximo 1 pregunta por turno
4. Profundiza en aspectos ESG no cubiertos
5. Usa transiciones naturales entre temas

Respuesta esperada (solo texto, sin formato):
"""

if "estado" not in st.session_state:
    st.session_state.estado = "INICIO"
    st.session_state.historial = []
    st.session_state.esg_scores = {"Ambiental": 0, "Social": 0, "Gobernanza": 0, "Riesgo": 0}
    st.session_state.contexto_actual = {}

# Interfaz mejorada
st.title("💬 Diálogo ESG Personalizado")
st.caption("Conversación inteligente para analizar tu perfil de inversión")

# Mostrar historial de chat
for msg in st.session_state.historial:
    avatar = "🤖" if msg["rol"] == "analista" else "👤"
    with st.chat_message(msg["rol"], avatar=avatar):
        st.write(msg["contenido"])

# Lógica de conversación
if st.session_state.estado == "INICIO":
    with st.chat_message("analista", avatar="🤖"):
        st.write(ESTADOS_CONVERSACION["INICIO"]["mensaje"])
    st.session_state.historial.append({
        "rol": "analista",
        "contenido": ESTADOS_CONVERSACION["INICIO"]["mensaje"]
    })
    st.session_state.estado = "PRIMERA_NOTICIA"

# Manejar respuestas del usuario
if user_input := st.chat_input("Escribe tu respuesta..."):
    # Guardar respuesta del usuario
    st.session_state.historial.append({
        "rol": "usuario",
        "contenido": user_input
    })
    
    # Analizar respuesta y generar réplica
    contexto = {
        "noticia_actual": ESTADOS_CONVERSACION[st.session_state.estado].get("noticia", ""),
        "historial_chat": "\n".join([f"{m['rol']}: {m['contenido']}" for m in st.session_state.historial[-4:]]),
        "objetivo_actual": f"Profundizar en aspectos {list(TEMAS_ESG.keys())} de la noticia"
    }
    
    with st.spinner("Analizando tu respuesta..."):
        respuesta = llm.invoke(plantilla_dialogo.format(**contexto))
    
    # Mostrar respuesta del analista
    with st.chat_message("analista", avatar="🤖"):
        st.write(respuesta)
    
    st.session_state.historial.append({
        "rol": "analista",
        "contenido": respuesta
    })

    # Actualizar puntuaciones ESG (ejemplo simplificado)
    if "ambiental" in respuesta.lower():
        st.session_state.esg_scores["Ambiental"] += 10
    elif "social" in respuesta.lower():
        st.session_state.esg_scores["Social"] += 10

# Mostrar resultados finales
if st.session_state.estado == "FIN":
    st.subheader("📊 Tu Perfil ESG Final")
    
    # Gráfico radial
    categorias = list(st.session_state.esg_scores.keys())
    valores = list(st.session_state.esg_scores.values())
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(categorias, valores, 'teal', alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticklabels(categorias)
    st.pyplot(fig)
    
    # Explicación textual
    st.write("""
    **Análisis detallado:**
    - **Prioridad Ambiental:** {ambiental}/100
    - **Sensibilidad Social:** {social}/100
    - **Exigencia Gobernanza:** {gobernanza}/100
    - **Tolerancia Riesgo:** {riesgo}/100
    """.format(**st.session_state.esg_scores))
