import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
import json
import re
from openai import OpenAI

# ============================================================
# 1. CONFIGURACIÓN Y ESTILOS
# ============================================================
st.set_page_config(page_title="MentorAI - Orientación Vocacional", page_icon="🧠", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    .stApp { background-color: #F8FAFC !important; font-family: 'Inter', sans-serif; }
    html, body, [class*="st-"] { color: #334155 !important; }
    .stChatInput textarea { border-radius: 12px !important; }
    .stChatMessage { border-radius: 12px !important; padding: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. CARGA DEL MOTOR XGBOOST LOCAL
# ============================================================
@st.cache_resource
def load_mentor_engine():
    model_path = "modelos/motor_completo.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        class MockEngine:
            def recommend(self, perfil, top_k=5, include_details=True):
                import random
                carreras = [
                    "Datos e IA",
                    "Tecnología Core",
                    "Diseño UX/UI",
                    "Negocios Tech",
                    "Marketing Digital"
                ]
                random.shuffle(carreras)
                return [
                    {
                        "rank": i + 1,
                        "carrera": c,
                        "confidence": round(random.uniform(70, 95) - (i * 3), 1)
                    }
                    for i, c in enumerate(carreras[:top_k])
                ]
        return MockEngine()

engine = load_mentor_engine()

# ============================================================
# 3. CONFIGURACIÓN DEL LLM (Google Gemini)
# ============================================================
API_KEY = st.secrets.get("GEMINI_API_KEY", "")

if not API_KEY:
    st.error("⚠️ No se encontró la API Key de Gemini. Configúrala en los Secrets de Streamlit como `GEMINI_API_KEY`.")
    st.stop()

client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
Eres MentorAI, un orientador vocacional empático y conversacional.
Tu objetivo es evaluar al usuario en 13 áreas (del 1 al 10), además de conocer su 'age' (edad) y 'education' (1=Secundaria, 2=Universidad, 3=Maestría, 4=Doctorado).

Las 13 áreas son: analytical, logical_reasoning, problem_solving, creativity, design, communication, empathy, social, teamwork, leadership, technology, business, stress_tolerance.

REGLAS:
1. NO hagas una lista aburrida de preguntas. Haz preguntas conversacionales e indaga sobre sus gustos. Haz máximo 2 o 3 preguntas a la vez.
2. Ve estimando internamente su puntaje del 1 al 10 en cada área según lo que responda.
3. CUANDO YA TENGAS SUFICIENTE INFORMACIÓN para estimar los 15 valores, DEJA DE HABLAR normalmente.
4. Tu ÚLTIMO mensaje debe ser ÚNICAMENTE un bloque de código JSON con esta estructura exacta (sin texto antes ni después):

```json
{
  "analytical": 8,
  "logical_reasoning": 7,
  "problem_solving": 9,
  "creativity": 4,
  "design": 3,
  "communication": 8,
  "empathy": 9,
  "social": 8,
  "teamwork": 10,
  "leadership": 7,
  "technology": 6,
  "business": 5,
  "stress_tolerance": 8,
  "age": 22,
  "education": 2
}
"""

# ============================================================
# 4. INICIALIZACIÓN DEL ESTADO DE LA SESIÓN
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "assistant",
            "content": (
                "¡Hola! Soy MentorAI 🧠. Cuéntame un poco sobre ti: "
                "¿qué actividades o materias disfrutas más?"
            )
        }
    ]
    st.session_state.finished = False

# ============================================================
# 5. INTERFAZ DE USUARIO
# ============================================================
st.title("🧠 MentorAI - Orientación Vocacional")
st.markdown("Chatea conmigo para descubrir tu carrera ideal.")
st.markdown("---")

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input(
    "Escribe tu respuesta..." if not st.session_state.finished else "Evaluación completada."
):
    if not st.session_state.finished:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                response = client.chat.completions.create(
                    model="gemini-3.5-flash",
                    messages=st.session_state.messages,
                    temperature=0.7
                )
                llm_reply = response.choices[0].message.content
                st.session_state.messages.append(
                    {"role": "assistant", "content": llm_reply}
                )

                json_match = re.search(
                    r'json\s*(\{.*?\})\s*', llm_reply, re.DOTALL
                )
                if json_match:
                    st.session_state.finished = True
                    message_placeholder.markdown(
                        "✅ ¡Perfil completado! Procesando resultados..."
                    )
                    try:
                        datos_llm = json.loads(json_match.group(1))
                        edu_map = {1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9}

                        perfil_usuario = {
                            "analytical": datos_llm.get("analytical", 5) / 10.0,
                            "logical_reasoning": datos_llm.get("logical_reasoning", 5) / 10.0,
                            "problem_solving": datos_llm.get("problem_solving", 5) / 10.0,
                            "creativity": datos_llm.get("creativity", 5) / 10.0,
                            "design": datos_llm.get("design", 5) / 10.0,
                            "communication": datos_llm.get("communication", 5) / 10.0,
                            "empathy": datos_llm.get("empathy", 5) / 10.0,
                            "social": datos_llm.get("social", 5) / 10.0,
                            "teamwork": datos_llm.get("teamwork", 5) / 10.0,
                            "leadership": datos_llm.get("leadership", 5) / 10.0,
                            "technology": datos_llm.get("technology", 5) / 10.0,
                            "business": datos_llm.get("business", 5) / 10.0,
                            "stress_tolerance": datos_llm.get("stress_tolerance", 5) / 10.0,
                            "education": edu_map.get(
                                int(datos_llm.get("education", 2)), 0.5
                            ),
                            "age": min(datos_llm.get("age", 20) / 65.0, 1.0)
                        }

                        recomendaciones = engine.recommend(
                            perfil_usuario, top_k=3, include_details=True
                        )
                        df_res = pd.DataFrame(recomendaciones)
                        top_1 = df_res.iloc[0]

                        st.markdown("---")
                        st.markdown(
                            f"### 🎯 TU CARRERA IDEAL: "
                            f"{top_1['carrera'].replace('_', ' ').title()}"
                        )

                        fig = go.Figure(go.Bar(
                            x=df_res["confidence"],
                            y=df_res["carrera"].str.replace("_", " ").str.title(),
                            orientation="h",
                            marker=dict(
                                color=df_res["confidence"],
                                colorscale="Blues"
                            )
                        ))
                        fig.update_layout(
                            title="Afinidad Vocacional (%)",
                            yaxis=dict(autorange="reversed"),
                            height=300
                        )
                        st.plotly_chart(fig, width="stretch")

                    except Exception as e:
                        st.error(f"Error procesando JSON: {e}")
                else:
                    message_placeholder.markdown(llm_reply)

            except Exception as e:
                message_placeholder.error(f"Error de API: {e}")

# ============================================================
# REQUIREMENTS.TXT
# ============================================================
# Crea un archivo requirements.txt con el siguiente contenido:
#
# streamlit>=1.28.0
# pandas>=1.5.0
# numpy>=1.24.0
# plotly>=5.15.0
# joblib>=1.2.0
# openai>=1.0.0
#
# Solo necesitas la librería openai (NO hace falta instalar
# google-generativeai). La compatibilidad con Gemini se maneja
# a través del base_url de OpenAI.
#
# Además, configura los Secrets de Streamlit:
# Crea un archivo .streamlit/secrets.toml con:
#
# GEMINI_API_KEY = "tu-api-key-de-gemini-aqui"
