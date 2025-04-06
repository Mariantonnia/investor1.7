import streamlit as st
import random
from langchain import LLMChain, PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import re
import json

# Configuración inicial
load_dotenv()

# Configuración avanzada del modelo LLM
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.7,  # Más creatividad en las respuestas
    max_tokens=1024,
)

# Temas de conversación naturales con variantes
temas_conversacion = {
    "experiencia": [
        "He ayudado a muchas personas con sus inversiones, ¿cómo ha sido tu experiencia hasta ahora?",
        "Cada persona tiene una relación única con el dinero, ¿cómo describirías la tuya?",
        "¿Qué te motivó a empezar a interesarte por las inversiones?"
    ],
    "mercados": [
        "Los mercados han estado interesantes últimamente, ¿has notado algo que te llame la atención?",
        "Cuando ves noticias financieras, ¿qué tipo de información suele captar tu interés?",
        "En tiempos de volatilidad, ¿cómo sueles reaccionar?"
    ],
    "valores": [
        "Más allá de los rendimientos, ¿hay algo que siempre buscas en tus inversiones?",
        "Si una empresa pudiera representar tus valores, ¿cómo sería?",
        "¿Qué prácticas empresariales te hacen sentir más confiado al invertir?"
    ],
    "objetivos": [
        "Imaginemos el futuro, ¿cómo te gustaría que tus inversiones contribuyeran a tus planes?",
        "Cuando piensas en seguridad financiera, ¿qué es lo más importante para ti?",
        "¿Qué papel deberían jugar tus inversiones en tu vida dentro de 10 años?"
    ],
    "preocupaciones": [
        "¿Qué tipo de riesgos financieros te quitan más el sueño?",
        "En materia de inversiones, ¿cuáles son tus principales inquietudes?",
        "¿Hay algún escenario de mercado que te preocupe especialmente?"
    ]
}

# Plantillas mejoradas para conversación natural
plantilla_dialogo = """
Como asistente financiero experto, mantén una conversación natural para entender el perfil de inversión. 

Contexto:
{historial}

Último mensaje del cliente: {input}

Instrucciones:
1. Muestra empatía y comprensión genuina
2. Profundiza en aspectos relevantes (ESG, riesgo, objetivos)
3. Haz preguntas abiertas cuando sea necesario
4. Mantén un tono profesional pero cercano
5. Adapta el lenguaje al nivel del cliente
6. Evita lenguaje técnico sin explicación
7. No des respuestas genéricas o de cuestionario

Responde únicamente con el texto de tu respuesta (sin prefijos como "Asistente:").
"""

plantilla_analisis = """
Analiza la conversación para determinar el perfil del inversor:

{conversacion}

Extrae:
1. Aversión al riesgo (0-100)
2. Preferencias ESG (Ambiental, Social, Gobernanza - cada 0-100)
3. Estilo de inversión (texto descriptivo)
4. Conocimiento financiero (básico, intermedio, avanzado)
5. Objetivos principales (crecimiento, seguridad, impacto)
6. Horizonte temporal (corto, medio, largo plazo)

Formato de salida (JSON válido):
{
    "riesgo": 0-100,
    "ambiental": 0-100,
    "social": 0-100,
    "gobernanza": 0-100,
    "estilo": "texto descriptivo",
    "conocimiento": "nivel",
    "objetivos": ["lista", "de", "objetivos"],
    "horizonte": "plazo"
}
"""

# Configuración de cadenas LangChain
prompt_dialogo = PromptTemplate(template=plantilla_dialogo, input_variables=["historial", "input"])
cadena_dialogo = LLMChain(llm=llm, prompt=prompt_dialogo)

prompt_analisis = PromptTemplate(template=plantilla_analisis, input_variables=["conversacion"])
cadena_analisis = LLMChain(llm=llm, prompt=prompt_analisis)

# Inicialización del estado de la sesión
def inicializar_sesion():
    if "historial" not in st.session_state:
        st.session_state.historial = []
        st.session_state.conversacion = []
        st.session_state.etapa = "inicio"
        st.session_state.temas_abordados = set()
        st.session_state.perfil = None
        st.session_state.mostrar_analisis = False

# Función para iniciar conversación de forma natural
def iniciar_dialogo():
    temas_disponibles = [t for t in temas_conversacion if t not in st.session_state.temas_abordados]
    
    if not temas_disponibles:
        return "Hemos cubierto varios temas importantes. ¿Hay algún aspecto particular de tus inversiones que quieras comentar?"
    
    tema_elegido = random.choice(temas_disponibles)
    st.session_state.temas_abordados.add(tema_elegido)
    return random.choice(temas_conversacion[tema_elegido])

# Función para generar gráfico de perfil
def generar_grafico_perfil(perfil):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico de barras ESG
    categorias_esg = ["Ambiental", "Social", "Gobernanza"]
    valores_esg = [perfil["ambiental"], perfil["social"], perfil["gobernanza"]]
    ax1.bar(categorias_esg, valores_esg, color=['#2e8b57', '#4682b4', '#6a5acd'])
    ax1.set_ylim(0, 100)
    ax1.set_title("Preferencias ESG")
    ax1.set_ylabel("Puntuación")
    
    # Gráfico de radar
    categorias_radar = ['Riesgo', 'Ambiental', 'Social', 'Gobernanza']
    valores_radar = [perfil["riesgo"], perfil["ambiental"], perfil["social"], perfil["gobernanza"]]
    valores_radar += valores_radar[:1]  # Para cerrar el círculo
    
    angulos = [n / float(len(categorias_radar)) * 2 * 3.1416 for n in range(len(categorias_radar))]
    angulos += angulos[:1]
    
    ax2 = plt.subplot(122, polar=True)
    ax2.plot(angulos, valores_radar, color='#ff7f0e', linewidth=2)
    ax2.fill(angulos, valores_radar, color='#ff7f0e', alpha=0.25)
    ax2.set_xticks(angulos[:-1])
    ax2.set_xticklabels(categorias_radar)
    ax2.set_title("Perfil Completo", y=1.1)
    ax2.set_rlabel_position(30)
    
    plt.tight_layout()
    return fig

# Interfaz de la aplicación
def main():
    st.set_page_config(page_title="Asesor Financiero AI", page_icon="💼")
    st.title("💬 Asesor de Inversiones Personalizado")
    st.caption("Un chatbot conversacional para entender tu perfil de inversor")
    
    inicializar_sesion()
    
    # Mostrar historial de chat
    for mensaje in st.session_state.historial:
        with st.chat_message(mensaje["role"]):
            st.write(mensaje["content"])
    
    # Iniciar conversación
    if st.session_state.etapa == "inicio":
        mensaje_inicial = (
            "¡Hola! Soy tu asesor financiero virtual. "
            "Vamos a conocernos mejor a través de una conversación natural sobre tus preferencias de inversión. "
            "Habla libremente sobre lo que consideres importante."
        )
        
        with st.chat_message("assistant"):
            st.write(mensaje_inicial)
        
        st.session_state.historial.append({"role": "assistant", "content": mensaje_inicial})
        st.session_state.conversacion.append(f"Asesor: {mensaje_inicial}")
        st.session_state.etapa = "dialogo"
    
    # Manejo de la interacción del usuario
    if user_input := st.chat_input("Escribe tu mensaje..."):
        st.session_state.historial.append({"role": "user", "content": user_input})
        st.session_state.conversacion.append(f"Cliente: {user_input}")
        
        # Generar respuesta conversacional
        with st.spinner("Pensando..."):
            if len(st.session_state.historial) < 4:
                respuesta = iniciar_dialogo()
            else:
                historial_texto = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.historial[-6:]])
                respuesta = cadena_dialogo.run(historial=historial_texto, input=user_input)
            
            st.session_state.historial.append({"role": "assistant", "content": respuesta})
            st.session_state.conversacion.append(f"Asesor: {respuesta}")
            
            # Verificar si tenemos suficiente información para el análisis
            if len(st.session_state.historial) >= 8 and not st.session_state.mostrar_analisis:
                with st.spinner("Analizando tu perfil..."):
                    try:
                        conversacion_texto = "\n".join(st.session_state.conversacion)
                        perfil_json = cadena_analisis.run(conversacion=conversacion_texto)
                        st.session_state.perfil = json.loads(perfil_json)
                        st.session_state.mostrar_analisis = True
                        
                        # Mensaje de resumen
                        resumen = (
                            f"Basado en nuestra conversación, veo que tienes un perfil {st.session_state.perfil['estilo'].lower()} "
                            f"con conocimiento {st.session_state.perfil['conocimiento']}. "
                            f"Tus principales objetivos son: {', '.join(st.session_state.perfil['objetivos'])}."
                        )
                        
                        st.session_state.historial.append({"role": "assistant", "content": resumen})
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error al analizar el perfil: {str(e)}")
    
    # Mostrar análisis si está listo
    if st.session_state.mostrar_analisis and st.session_state.perfil:
        with st.expander("🔍 Ver análisis detallado", expanded=True):
            st.subheader("Tu Perfil de Inversión")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nivel de Riesgo", f"{st.session_state.perfil['riesgo']}/100")
                st.write(f"**Estilo:** {st.session_state.perfil['estilo']}")
                st.write(f"**Conocimiento:** {st.session_state.perfil['conocimiento'].capitalize()}")
            
            with col2:
                st.metric("Horizonte Temporal", st.session_state.perfil["horizonte"].capitalize())
                st.write(f"**Objetivos:** {', '.join(st.session_state.perfil['objetivos'])}")
            
            st.pyplot(generar_grafico_perfil(st.session_state.perfil))
            
            st.write("**Recomendaciones iniciales:**")
            if st.session_state.perfil["riesgo"] < 30:
                st.write("- Considera carteras diversificadas con bajo volatility")
            elif st.session_state.perfil["riesgo"] > 70:
                st.write("- Podrías explorar activos de mayor crecimiento con gestión activa del riesgo")
            else:
                st.write("- Un enfoque balanceado podría alinearse con tu perfil")
            
            if any(val > 70 for val in [st.session_state.perfil[k] for k in ["ambiental", "social", "gobernanza"]]):
                st.write("- Las inversiones con impacto ESG podrían ser importantes para ti")

if __name__ == "__main__":
    main()
