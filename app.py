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
        self.news_items = [
            "Repsol entre las 50 empresas con mayor responsabilidad histórica en el calentamiento global",
            "Amancio Ortega crea un fondo de 100 millones para afectados por inundaciones"
        ]
        self.current_news_index = 0
        self.questions_for_news = [
            ["¿Qué opinas sobre esta noticia?", "¿Cómo valoras el compromiso ambiental de las empresas energéticas?", "¿Qué medidas crees que deberían tomar para mejorar su impacto?"],
            ["¿Qué te parece esta iniciativa?", "¿Cómo valoras la responsabilidad social de los grandes fortunas?", "¿Crees que este tipo de acciones deberían ser más comunes?"]
        ]
        self.current_question_index = 0
        self.scores = {"Ambiental": 0, "Social": 0, "Gobernanza": 0, "Riesgo": 0}
        self.state = "start"  # start -> show_news -> ask_question -> analyze -> next_question_or_news -> results

    def get_current_news(self):
        return self.news_items[self.current_news_index]
    
    def get_current_question(self):
        return self.questions_for_news[self.current_news_index][self.current_question_index]
    
    def move_to_next_question(self):
        self.current_question_index += 1
        if self.current_question_index >= len(self.questions_for_news[self.current_news_index]):
            self.current_question_index = 0
            self.current_news_index += 1
            if self.current_news_index >= len(self.news_items):
                self.state = "results"
            else:
                self.state = "show_news"
        else:
            self.state = "ask_question"

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

# Inicialización completa del estado
if "conversation" not in st.session_state:
    st.session_state.conversation = ESGConversationManager()
    st.session_state.chat_history = []
    st.session_state.news_responses = []
    st.session_state.waiting_for_response = False

# Mostrar historial de chat
for msg in st.session_state.chat_history:
    avatar = "🤖" if msg["role"] == "analyst" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Manejo de la conversación
cm = st.session_state.conversation

if cm.state == "start":
    with st.chat_message("analyst", avatar="🤖"):
        st.write("¡Hola! Soy tu asesor ESG personal. Analizaremos tu perfil de inversión mediante algunas noticias relevantes. ¿Estás listo para comenzar?")
    st.session_state.chat_history.append({
        "role": "analyst",
        "content": "¡Hola! Soy tu asesor ESG personal. Analizaremos tu perfil de inversión mediante algunas noticias relevantes. ¿Estás listo para comenzar?"
    })
    cm.state = "waiting_start_confirmation"
    st.session_state.waiting_for_response = True

elif cm.state == "show_news":
    news = cm.get_current_news()
    with st.chat_message("analyst", avatar="🤖"):
        st.markdown(f"📰 **Noticia:** {news}")
    st.session_state.chat_history.append({
        "role": "analyst",
        "content": f"📰 **Noticia:** {news}"
    })
    cm.state = "ask_question"
    st.rerun()

elif cm.state == "ask_question":
    question = cm.get_current_question()
    with st.chat_message("analyst", avatar="🤖"):
        st.write(question)
    st.session_state.chat_history.append({
        "role": "analyst",
        "content": question
    })
    cm.state = "waiting_response"
    st.session_state.waiting_for_response = True

# Procesar input del usuario
if user_input := st.chat_input("Escribe tu respuesta..."):
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    st.session_state.news_responses.append(user_input)
    st.session_state.waiting_for_response = False
    
    if cm.state == "waiting_start_confirmation":
        cm.state = "show_news"
        st.rerun()
    
    elif cm.state == "waiting_response":
        # Analizar respuesta
        context = {
            "news": cm.get_current_news(),
            "user_input": user_input,
            "history": "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-3:]])
        }
        
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
                
                # Mover a siguiente pregunta o noticia
                cm.move_to_next_question()
                st.rerun()
                
            except Exception as e:
                st.error(f"Error al generar respuesta: {str(e)}")
                st.stop()

# Mostrar resultados finales
if cm.state == "results":
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
