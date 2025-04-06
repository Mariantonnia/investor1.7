import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from typing import Dict, List

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
            "START": {
                "message": "¡Hola! Soy tu asesor ESG personal. Analizaremos tu perfil de inversión mediante algunas noticias relevantes. ¿Estás listo para comenzar?",
                "transition": "NEWS_1"
            },
            "NEWS_1": {
                "news": "Repsol entre las 50 empresas con mayor responsabilidad histórica en el calentamiento global",
                "questions": [
                    "¿Qué opinas sobre esta noticia?",
                    "¿Cómo valoras el compromiso ambiental de las empresas energéticas?",
                    "¿Qué medidas crees que deberían tomar para mejorar su impacto?"
                ],
                "categories": ["Ambiental", "Social", "Gobernanza"],
                "transition": "NEWS_2"
            },
            "NEWS_2": {
                "news": "Amancio Ortega crea un fondo de 100 millones para afectados por inundaciones",
                "questions": [
                    "¿Qué te parece esta iniciativa?",
                    "¿Cómo valoras la responsabilidad social de los grandes fortunas?",
                    "¿Crees que este tipo de acciones deberían ser más comunes?"
                ],
                "categories": ["Social", "Gobernanza"],
                "transition": "RESULTS"
            }
        }
        self.question_index = 0
        self.scores = {"Ambiental": 0, "Social": 0, "Gobernanza": 0, "Riesgo": 0}
    
    def get_current_state(self):
        return self.states.get(st.session_state.current_state, {})
    
    def next_question(self):
        state = self.get_current_state()
        questions = state.get("questions", [])
        if self.question_index < len(questions):
            question = questions[self.question_index]
            self.question_index += 1
            return question
        else:
            st.session_state.current_state = state.get("transition", "RESULTS")
            self.question_index = 0
            return None

# Plantilla de análisis mejorada
analysis_template = """
Como analista ESG experto, analiza esta conversación:

Noticia: {news}
Última interacción:
Usuario: {user_input}
Contexto previo: {history}

Instrucciones:
1. Identifica la categoría ESG principal (Ambiental, Social, Gobernanza o Riesgo)
2. Analiza el sentimiento (positivo/neutral/negativo)
3. Genera una respuesta natural que:
   - Reconozca el comentario del usuario
   - Profundice en aspectos no mencionados
   - Haga solo UNA pregunta relevante
   - Mantenga un tono profesional pero cercano

Formato de respuesta:
[Análisis breve] [Pregunta de seguimiento]

Ejemplo:
"Entiendo tu preocupación por el aspecto ambiental. Esto es crucial porque afecta directamente al cambio climático. ¿Has considerado también cómo impacta esto en las comunidades locales?"
"""

# Configuración inicial de Streamlit
st.set_page_config(page_title="💬 Asesor ESG Conversacional", layout="wide")
st.title("💬 Asesor ESG Conversacional")
st.caption("Diálogo inteligente para analizar tu perfil de inversión responsable")

# Inicialización de estado
if "current_state" not in st.session_state:
    st.session_state.current_state = "START"
    st.session_state.chat_history = []
    st.session_state.conversation = ESGConversationManager()

# Mostrar historial de chat
for msg in st.session_state.chat_history:
    avatar = "🤖" if msg["role"] == "analyst" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Manejo de la conversación
cm = st.session_state.conversation
current_state = cm.get_current_state()

if current_state:
    if "message" in current_state and not st.session_state.chat_history:
        # Mensaje inicial
        with st.chat_message("analyst", avatar="🤖"):
            st.write(current_state["message"])
        st.session_state.chat_history.append({
            "role": "analyst",
            "content": current_state["message"]
        })
    else:
        # Continuar con preguntas sobre la noticia actual
        next_question = cm.next_question()
        if next_question:
            message = f"📰 Noticia: {current_state['news']}\n\n{next_question}"
            with st.chat_message("analyst", avatar="🤖"):
                st.write(message)
            st.session_state.chat_history.append({
                "role": "analyst",
                "content": message
            })

# Procesar input del usuario
if user_input := st.chat_input("Escribe tu respuesta..."):
    # Guardar respuesta del usuario
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Preparar contexto para el análisis
    context = {
        "news": cm.get_current_state().get("news", ""),
        "user_input": user_input,
        "history": "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-3:]])
    }
    
    # Generar respuesta analizada
    with st.spinner("Analizando tu respuesta..."):
        try:
            prompt = PromptTemplate(template=analysis_template, input_variables=["news", "user_input", "history"])
            chain = LLMChain(llm=llm, prompt=prompt)
            response = chain.run(**context)
            
            # Extraer el contenido si es un AIMessage
            response_content = response.content if isinstance(response, AIMessage) else str(response)
            
            # Actualizar puntuaciones basado en la categoría detectada
            response_lower = response_content.lower()
            if "ambiental" in response_lower:
                cm.scores["Ambiental"] += 10
            elif "social" in response_lower:
                cm.scores["Social"] += 10
            elif "gobernanza" in response_lower:
                cm.scores["Gobernanza"] += 10
            elif "riesgo" in response_lower:
                cm.scores["Riesgo"] += 5
            
            # Mostrar respuesta
            with st.chat_message("analyst", avatar="🤖"):
                st.write(response_content)
            st.session_state.chat_history.append({
                "role": "analyst",
                "content": response_content
            })
            
        except Exception as e:
            st.error(f"Error al generar respuesta: {str(e)}")
            st.stop()

# Mostrar resultados finales
if st.session_state.current_state == "RESULTS":
    st.subheader("📊 Tu Perfil ESG Completo")
    
    # Gráfico de radar mejorado
    categories = list(cm.scores.keys())
    values = list(cm.scores.values())
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(categories + [categories[0]], values + [values[0]], color='teal', linewidth=2)
    ax.fill(categories + [categories[0]], values + [values[0]], color='teal', alpha=0.25)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_yticks([25, 50, 75, 100])
    ax.grid(True)
    st.pyplot(fig)
    
    # Explicación detallada
    st.markdown("""
    ## 📝 Análisis de tu perfil:
    
    | Dimensión       | Puntuación | Interpretación |
    |-----------------|------------|----------------|
    | **Ambiental**   | {}/100     | {}             |
    | **Social**      | {}/100     | {}             |
    | **Gobernanza**  | {}/100     | {}             |
    | **Riesgo**      | {}/100     | {}             |
    """.format(
        cm.scores["Ambiental"],
        "Muy consciente" if cm.scores["Ambiental"] > 70 else "Moderada" if cm.scores["Ambiental"] > 40 else "Baja",
        cm.scores["Social"],
        "Muy alta" if cm.scores["Social"] > 70 else "Media" if cm.scores["Social"] > 40 else "Baja",
        cm.scores["Gobernanza"],
        "Muy exigente" if cm.scores["Gobernanza"] > 70 else "Moderada" if cm.scores["Gobernanza"] > 40 else "Flexible",
        cm.scores["Riesgo"],
        "Muy conservador" if cm.scores["Riesgo"] > 70 else "Moderado" if cm.scores["Riesgo"] > 40 else "Arriesgado"
    ))
