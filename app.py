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
    temperature=0.7,  # Aumentamos ligeramente la temperatura para respuestas más naturales
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
Eres un analista financiero experto en ESG (Environmental, Social, Governance). 
El usuario ha respondido a la noticia con el siguiente comentario:

Noticia: {noticia}
Comentario: {reaccion}

Analiza el sentimiento y la preocupación expresada considerando estos aspectos:
1. Clasifica la preocupación principal (Ambiental, Social, Gobernanza o Riesgo)
2. Evalúa la profundidad del comentario (superficial, moderado, profundo)
3. Identifica emociones subyacentes (preocupación, indiferencia, aprobación, etc.)

Si el comentario es superficial (menos de 15 palabras o sin justificación), genera UNA SOLA pregunta de seguimiento natural para profundizar en el tema. La pregunta debe ser abierta y relacionada con la categoría detectada.

Si el comentario es adecuado (más de 15 palabras con alguna justificación), ofrece un breve reconocimiento y pasa a la siguiente noticia.

Ejemplo de respuesta para comentario superficial:
"Interesante perspectiva. ¿Podrías contarme más sobre cómo crees que esto afectará a [aspecto relacionado con la categoría] a largo plazo?"

Ejemplo de respuesta para comentario adecuado:
"Gracias por compartir tu análisis detallado. Veo que te preocupa especialmente [categoría]. Pasemos a la siguiente noticia."

Respuesta:
"""

prompt_reaccion = PromptTemplate(template=plantilla_reaccion, input_variables=["noticia", "reaccion"])
cadena_reaccion = LLMChain(llm=llm, prompt=prompt_reaccion)

plantilla_perfil = """
Analiza las siguientes respuestas de un inversor a diversas noticias financieras:

{analisis}

Genera un perfil detallado del inversor considerando:
1. Preferencias ESG (Ambiental, Social, Gobernanza)
2. Tolerancia al riesgo
3. Estilo de inversión (conservador, moderado, agresivo)
4. Preocupaciones principales

Asigna puntuaciones de 0-100 para cada dimensión ESG y riesgo, donde 100 es máxima preocupación/aversión.

Formato de salida:
**Perfil del Inversor**
- Ambiental: [puntuación]/100 - [breve explicación]
- Social: [puntuación]/100 - [breve explicación]
- Gobernanza: [puntuación]/100 - [breve explicación]
- Riesgo: [puntuación]/100 - [breve explicación]
**Estilo de inversión**: [descripción]
"""

prompt_perfil = PromptTemplate(template=plantilla_perfil, input_variables=["analisis"])
cadena_perfil = LLMChain(llm=llm, prompt=prompt_perfil)

# Inicialización del estado de la sesión
if "historial" not in st.session_state:
    st.session_state.historial = []
    st.session_state.contador = 0
    st.session_state.reacciones = []
    st.session_state.esperando_respuesta = False

st.title("🤖 Analizador ESG para Inversores")

# Mostrar historial de conversación
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["tipo"], avatar=mensaje.get("avatar", None)):
        st.markdown(mensaje["contenido"])

# Lógica principal de la conversación
if st.session_state.contador < len(noticias):
    noticia_actual = noticias[st.session_state.contador]
    
    # Mostrar noticia si es el inicio o se acaba de avanzar
    if not st.session_state.esperando_respuesta:
        with st.chat_message("assistant", avatar="🤖"):
            mensaje_noticia = st.markdown(f"📰 **Noticia {st.session_state.contador + 1}/{len(noticias)}:**\n\n*{noticia_actual}*\n\n¿Qué opinas sobre esta noticia?")
        st.session_state.historial.append({
            "tipo": "assistant", 
            "contenido": f"📰 **Noticia {st.session_state.contador + 1}/{len(noticias)}:**\n\n*{noticia_actual}*\n\n¿Qué opinas sobre esta noticia?",
            "avatar": "🤖"
        })
        st.session_state.esperando_respuesta = True
    
    # Procesar respuesta del usuario
    if user_input := st.chat_input("Escribe tu respuesta aquí..."):
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        st.session_state.historial.append({
            "tipo": "user", 
            "contenido": user_input,
            "avatar": "👤"
        })
        st.session_state.reacciones.append(user_input)
        
        # Analizar la respuesta
        with st.spinner("Analizando tu respuesta..."):
            respuesta_bot = cadena_reaccion.run(noticia=noticia_actual, reaccion=user_input)
        
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(respuesta_bot)
        st.session_state.historial.append({
            "tipo": "assistant", 
            "contenido": respuesta_bot,
            "avatar": "🤖"
        })
        
        # Determinar si avanzamos o esperamos más input
        if "siguiente noticia" in respuesta_bot.lower():
            st.session_state.contador += 1
            st.session_state.esperando_respuesta = False
        st.rerun()

else:  # Todas las noticias procesadas - generar perfil
    analisis_total = "\n\n".join([
        f"Noticia: {noticias[i]}\nRespuesta: {st.session_state.reacciones[i]}" 
        for i in range(len(noticias))
    ])
    
    with st.spinner("Generando tu perfil de inversor..."):
        perfil = cadena_perfil.run(analisis=analisis_total)
    
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(f"## 📊 Tu Perfil de Inversor ESG\n\n{perfil}")
    st.session_state.historial.append({
        "tipo": "assistant", 
        "contenido": f"## 📊 Tu Perfil de Inversor ESG\n\n{perfil}",
        "avatar": "🤖"
    })

    # Extraer puntuaciones para el gráfico
    puntuaciones = {
        "Ambiental": int(re.search(r"Ambiental: (\d+)", perfil).group(1)),
        "Social": int(re.search(r"Social: (\d+)", perfil).group(1)),
        "Gobernanza": int(re.search(r"Gobernanza: (\d+)", perfil).group(1)),
        "Riesgo": int(re.search(r"Riesgo: (\d+)", perfil).group(1)),
    }

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    categorias = list(puntuaciones.keys())
    valores = list(puntuaciones.values())
    bars = ax.bar(categorias, valores, color=['#4CAF50', '#2196F3', '#9C27B0', '#FF9800'])
    
    ax.set_ylabel("Puntuación (0-100)", fontsize=12)
    ax.set_title("Perfil ESG y de Riesgo", fontsize=14, pad=20)
    ax.set_ylim(0, 100)
    
    # Añadir valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    st.pyplot(fig)

    # Guardar en Google Sheets
    try:
        creds_json_str = st.secrets["gcp_service_account"]
        creds_json = json.loads(creds_json_str)
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)
        client = gspread.authorize(creds)
        
        sheet = client.open('BBDD_RESPUESTAS').sheet1
        fila = st.session_state.reacciones[:]
        fila.extend(valores)
        sheet.append_row(fila)
        
        st.success("✅ Tus respuestas y perfil han sido guardados exitosamente.")
    except Exception as e:
        st.error(f"⚠️ Error al guardar los datos: {str(e)}")
