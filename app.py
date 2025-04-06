import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

# Configuración del LLM
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.7,
    max_tokens=1024,
)

# Sistema de diálogo mejorado
class ESGConversationManager:
    def __init__(self):
        self.states = {
            "start": {
                "message": "¡Hola! Soy tu asesor ESG personal. Analizaremos tu perfil de inversión mediante algunas noticias relevantes. ¿Estás listo para comenzar?",
                "transition": "show_news_1"
            },
            "show_news_1": {
                "news": "Repsol entre las 50 empresas con mayor responsabilidad histórica en el calentamiento global",
                "transition": "ask_opinion_1"
            },
            "ask_opinion_1": {
                "question": "¿Qué opinas sobre esta noticia?",
                "transition": "analyze_response_1"
            },
            "analyze_response_1": {
                "follow_up": "¿Cómo valoras el compromiso ambiental de las empresas energéticas?",
                "transition": "ask_measures_1"
            },
            "ask_measures_1": {
                "question": "¿Qué medidas crees que deberían tomar para mejorar su impacto?",
                "transition": "show_news_2"
            },
            "show_news_2": {
                "news": "Amancio Ortega crea un fondo de 100 millones para afectados por inundaciones",
                "transition": "ask_opinion_2"
            },
            "ask_opinion_2": {
                "question": "¿Qué te parece esta iniciativa?",
                "transition": "analyze_response_2"
            },
            "analyze_response_2": {
                "follow_up": "¿Cómo valoras la responsabilidad social de los grandes fortunas?",
                "transition": "final_question"
            },
            "final_question": {
                "question": "¿Crees que este tipo de acciones deberían ser más comunes?",
                "transition": "show_results"
            }
        }
        self.current_state = "start"
        self.scores = {"Ambiental": 0, "Social": 0, "Gobernanza": 0, "Riesgo": 0}

    def get_current_step(self):
        return self.states.get(self.current_state, {})
    
    def move_to_next_state(self):
        self.current_state = self.states[self.current_state].get("transition", "show_results")

# Plantilla de análisis mejorada
analysis_template = """
Como analista ESG experto, analiza esta conversación:

Noticia: {news}
Respuesta del usuario: {user_input}
Contexto previo: {history}

Instrucciones:
1. Identifica la categoría ESG principal (Ambiental, Social, Gobernanza o Riesgo)
2. Analiza el sentimiento (positivo/neutral/negativo)
3. Genera una respuesta natural que:
   - Reconozca el comentario del usuario
   - Profundice en aspectos no mencionados
   - Haga solo UNA pregunta relevante
   - Mantenga un tono profesional pero cercano

Ejemplo de formato:
[Análisis breve] [Pregunta de seguimiento]

No repitas la noticia en tu respuesta.
"""

# Configuración de Streamlit
st.set_page_config(page_title="💬 Asesor ESG Conversacional", layout="wide")
st.title("💬 Asesor ESG Conversacional")
st.caption("Diálogo inteligente para analizar tu perfil de inversión responsable")

# Inicialización COMPLETA del estado
if "conversation" not in st.session_state:
    st.session_state.conversation = ESGConversationManager()
    st.session_state.chat_history = []
    st.session_state.waiting_for_input = False
    st.session_state.news_responses = []  # Inicialización añadida aquí

# Mostrar historial de chat
for msg in st.session_state.chat_history:
    avatar = "🤖" if msg["role"] == "analyst" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Manejo de la conversación
cm = st.session_state.conversation
current_step = cm.get_current_step()

if not st.session_state.waiting_for_input:
    if current_step:
        if "message" in current_step:
            # Mensaje inicial
            with st.chat_message("analyst", avatar="🤖"):
                st.write(current_step["message"])
            st.session_state.chat_history.append({
                "role": "analyst",
                "content": current_step["message"]
            })
            cm.move_to_next_state()
            st.session_state.waiting_for_input = True
        
        elif "news" in current_step:
            # Mostrar noticia
            with st.chat_message("analyst", avatar="🤖"):
                st.markdown(f"📰 **Noticia:** {current_step['news']}")
            st.session_state.chat_history.append({
                "role": "analyst",
                "content": f"📰 **Noticia:** {current_step['news']}"
            })
            cm.move_to_next_state()
            st.rerun()
        
        elif "question" in current_step:
            # Hacer pregunta
            with st.chat_message("analyst", avatar="🤖"):
                st.write(current_step["question"])
            st.session_state.chat_history.append({
                "role": "analyst",
                "content": current_step["question"]
            })
            st.session_state.waiting_for_input = True

# Procesar input del usuario
if user_input := st.chat_input("Escribe tu respuesta..."):
    # Guardar respuesta del usuario
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    st.session_state.news_responses.append(user_input)  # Ahora sí existe
    st.session_state.waiting_for_input = False
    
    # Solo analizar respuestas a preguntas, no a noticias
    if "question" in cm.get_current_step() or "follow_up" in cm.get_current_step():
        # Preparar contexto para el análisis
        context = {
            "news": next((step["news"] for step in cm.states.values() if "news" in step and step["news"] in "\n".join([m["content"] for m in st.session_state.chat_history[-3:]])), ""),
            "user_input": user_input,
            "history": "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-3:]])
        }
        
        # Generar respuesta analizada
        with st.spinner("Analizando tu respuesta..."):
            try:
                prompt = PromptTemplate(template=analysis_template, input_variables=["news", "user_input", "history"])
                chain = LLMChain(llm=llm, prompt=prompt)
                response = chain.run(**context)
                
                # Actualizar puntuaciones
                response_lower = response.lower()
                if "ambiental" in response_lower:
                    cm.scores["Ambiental"] += 15
                elif "social" in response_lower:
                    cm.scores["Social"] += 15
                elif "gobernanza" in response_lower:
                    cm.scores["Gobernanza"] += 15
                elif "riesgo" in response_lower:
                    cm.scores["Riesgo"] += 10
                
                # Mostrar respuesta
                with st.chat_message("analyst", avatar="🤖"):
                    st.write(response)
                st.session_state.chat_history.append({
                    "role": "analyst",
                    "content": response
                })
                
            except Exception as e:
                st.error(f"Error al generar respuesta: {str(e)}")
                st.stop()
    
    cm.move_to_next_state()
    st.rerun()

# Mostrar resultados finales
if cm.current_state == "show_results":
    st.subheader("📊 Tu Perfil ESG Completo")
    
    # Gráfico de radar
    categories = list(cm.scores.keys())
    values = [min(100, score) for score in cm.scores.values()]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(categories + [categories[0]], values + [values[0]], color='teal', linewidth=2)
    ax.fill(categories + [categories[0]], values + [values[0]], color='teal', alpha=0.25)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_ylim(0, 100)
    ax.grid(True)
    st.pyplot(fig)
    
    # Explicación detallada
    st.markdown("""
    ## 📝 Análisis de tu perfil ESG:
    
    **Preocupación Ambiental:** {}/100 - {}
    
    **Sensibilidad Social:** {}/100 - {}
    
    **Exigencia en Gobernanza:** {}/100 - {}
    
    **Tolerancia al Riesgo:** {}/100 - {}
    """.format(
        cm.scores["Ambiental"],
        "Muy alta" if cm.scores["Ambiental"] > 70 else "Moderada" if cm.scores["Ambiental"] > 40 else "Baja",
        cm.scores["Social"],
        "Muy alta" if cm.scores["Social"] > 70 else "Media" if cm.scores["Social"] > 40 else "Baja",
        cm.scores["Gobernanza"],
        "Muy exigente" if cm.scores["Gobernanza"] > 70 else "Moderada" if cm.scores["Gobernanza"] > 40 else "Flexible",
        cm.scores["Riesgo"],
        "Muy conservador" if cm.scores["Riesgo"] > 70 else "Moderado" if cm.scores["Riesgo"] > 40 else "Arriesgado"
    ))
