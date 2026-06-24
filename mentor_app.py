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
st.set_page_config(page_title="MentorAI", page_icon="🧠", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif !important; }

    .stApp {
        background: linear-gradient(135deg, #DBEAFE 0%, #EFF6FF 50%, #F0F4FF 100%);
    }

    .block-container {
        max-width: 720px !important;
        background: white !important;
        border-radius: 16px !important;
        padding: 0 !important;
        box-shadow: 0 8px 32px rgba(30, 41, 59, 0.08), 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-top: 1.5rem !important;
        margin-bottom: 1.5rem !important;
    }

    header { display: none !important; }
    footer { display: none !important; }
    #MainMenu { display: none !important; }
    .stDeployButton { display: none !important; }

    .chat-header {
        background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%);
        padding: 20px 24px;
        text-align: center;
        color: white;
    }
    .chat-header h1 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    .chat-header p {
        margin: 4px 0 0 0;
        font-size: 0.85rem;
        opacity: 0.85;
        font-weight: 400;
    }

    .chat-body {
        padding: 24px 20px 12px 20px;
    }

    div[data-testid="stChatMessage"] {
        padding: 10px 14px !important;
        margin-bottom: 6px !important;
        border: none !important;
        color: #000000 !important;
    }
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] div {
        color: #000000 !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: #EFF6FF !important;
        border-radius: 16px 16px 4px 16px !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: #F8FAFC !important;
        border-radius: 16px 16px 16px 4px !important;
    }

    div[data-testid="chatAvatarIcon-user"],
    div[data-testid="chatAvatarIcon-assistant"] {
        font-size: 1.5rem !important;
        background: none !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 32px !important;
        height: 32px !important;
    }

    .stChatInput textarea {
        border-radius: 14px !important;
        border: 1.5px solid #E2E8F0 !important;
        background: #F8FAFC !important;
        padding: 10px 16px !important;
        font-size: 0.9rem !important;
        color: #000000 !important;
    }
    .stChatInput textarea::placeholder {
        color: #6B7280 !important;
    }
    .stChatInput textarea:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12) !important;
    }

    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        height: 44px !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    hr { margin: 20px 0 !important; border-color: #F1F5F9 !important; }

    .stAlert {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER (TARJETA FLOTANTE)
# ============================================================
st.markdown("""
<div class="chat-header">
    <h1>🧠 MentorAI</h1>
    <p>Tu orientador vocacional inteligente</p>
</div>
<div class="chat-body">
""", unsafe_allow_html=True)

# ============================================================
# 2. CARGA DEL MOTOR XGBOOST
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
# 3. CONFIGURACIÓN GROQ
# ============================================================
API_KEY = st.secrets.get("GROQ_API_KEY", "")
if not API_KEY:
    st.error("⚠️ Configura GROQ_API_KEY en los Secrets de Streamlit.")
    st.stop()

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
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
# 4. ESTADO DE SESIÓN
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "assistant",
            "content": (
                "¡Hola! Soy MentorAI 🧠. Cuéntame un poco sobre ti: "
                "¿qué actividades o materias disfrutas más en tu día a día?"
            )
        }
    ]
    st.session_state.finished = False

# ============================================================
# 5. FUNCIÓN AUXILIAR
# ============================================================
def extract_json(text):
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            required_keys = {
                "analytical", "logical_reasoning", "problem_solving",
                "creativity", "design", "communication", "empathy",
                "social", "teamwork", "leadership", "technology",
                "business", "stress_tolerance", "age", "education"
            }
            if required_keys.issubset(data.keys()):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
    return None

# ============================================================
# 6. CHAT
# ============================================================
for msg in st.session_state.messages:
    if msg["role"] != "system":
        avatar = "👤" if msg["role"] == "user" else "🧠"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

if prompt := st.chat_input(
    "Escribe tu respuesta aquí..."
    if not st.session_state.finished
    else "La evaluación ha finalizado"
):
    if not st.session_state.finished:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🧠"):
            message_placeholder = st.empty()
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=st.session_state.messages,
                    temperature=0.7
                )
                llm_reply = response.choices[0].message.content

                datos_llm = extract_json(llm_reply)
                if datos_llm:
                    st.session_state.finished = True
                    with st.spinner("Analizando tus respuestas..."):
                        import time
                        time.sleep(0.5)
                    message_placeholder.success(
                        "✅ ¡Perfil completado! Generando tu mapa vocacional..."
                    )

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
                        "education": edu_map.get(int(datos_llm.get("education", 2)), 0.5),
                        "age": min(datos_llm.get("age", 20) / 65.0, 1.0)
                    }

                    recomendaciones = engine.recommend(
                        perfil_usuario, top_k=3, include_details=True
                    )
                    df_res = pd.DataFrame(recomendaciones)
                    top_1 = df_res.iloc[0]

                    st.markdown("---")
                    st.markdown(
                        f"<h2 style='text-align:center; color:#1E293B; font-size:1.3rem;'>"
                        f"🎯 Tu carrera ideal: "
                        f"<span style='color:#2563EB;'>{top_1['carrera'].replace('_', ' ').title()}</span>"
                        f"</h2>",
                        unsafe_allow_html=True
                    )

                    c1, c2 = st.columns(2, gap="medium")

                    with c1:
                        labels = [
                            "Analítico", "Razonamiento", "Problemas",
                            "Creatividad", "Diseño", "Comunicación",
                            "Empatía", "Social", "Trabajo en Equipo",
                            "Liderazgo", "Tecnología", "Negocios",
                            "Tolerancia al Estrés"
                        ]
                        values = [
                            datos_llm.get("analytical", 5),
                            datos_llm.get("logical_reasoning", 5),
                            datos_llm.get("problem_solving", 5),
                            datos_llm.get("creativity", 5),
                            datos_llm.get("design", 5),
                            datos_llm.get("communication", 5),
                            datos_llm.get("empathy", 5),
                            datos_llm.get("social", 5),
                            datos_llm.get("teamwork", 5),
                            datos_llm.get("leadership", 5),
                            datos_llm.get("technology", 5),
                            datos_llm.get("business", 5),
                            datos_llm.get("stress_tolerance", 5)
                        ]
                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values + [values[0]],
                            theta=labels + [labels[0]],
                            fill="toself",
                            fillcolor="rgba(37, 99, 235, 0.2)",
                            line=dict(color="#2563EB", width=2.5),
                            name="Tu perfil"
                        ))
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 10],
                                    tickfont=dict(size=9, color="#94A3B8"),
                                    gridcolor="#E2E8F0"
                                ),
                                angularaxis=dict(
                                    tickfont=dict(size=9, color="#475569")
                                ),
                                bgcolor="rgba(0,0,0,0)"
                            ),
                            paper_bgcolor="rgba(0,0,0,0)",
                            showlegend=False,
                            height=320,
                            margin=dict(l=40, r=40, t=10, b=10)
                        )
                        st.plotly_chart(fig_radar, width="stretch")

                    with c2:
                        fig_bar = go.Figure(go.Bar(
                            x=df_res["confidence"],
                            y=df_res["carrera"].str.replace("_", " ").str.title(),
                            orientation="h",
                            marker=dict(
                                color=df_res["confidence"],
                                colorscale=[[0, "#BFDBFE"], [1, "#2563EB"]]
                            ),
                            text=df_res["confidence"].apply(lambda v: f"{v}%"),
                            textposition="outside",
                            textfont=dict(size=12, color="#1E293B")
                        ))
                        fig_bar.update_layout(
                            yaxis=dict(
                                autorange="reversed",
                                tickfont=dict(size=12, color="#1E293B")
                            ),
                            xaxis=dict(range=[0, 100], visible=False),
                            height=240,
                            margin=dict(l=10, r=40, t=10, b=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            showlegend=False
                        )
                        st.plotly_chart(fig_bar, width="stretch")

                    st.markdown("---")
                    if st.button(
                        "🔄 Volver a comenzar",
                        use_container_width=True,
                        type="primary"
                    ):
                        for key in list(st.session_state.keys()):
                            del st.session_state[key]
                        st.session_state.finished = False
                        st.session_state.messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "assistant", "content": "¡Hola! Soy MentorAI 🧠. Cuéntame un poco sobre ti: ¿qué actividades o materias disfrutas más?"}
                        ]
                        st.rerun()

                else:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": llm_reply}
                    )
                    message_placeholder.markdown(llm_reply)

            except Exception as e:
                message_placeholder.error(f"Error de API: {e}")

# ============================================================
# CERRAR DIVS DEL HTML
# ============================================================
st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# REQUIREMENTS.TXT
# ============================================================
# Crea un archivo requirements.txt con:
#
# streamlit>=1.28.0
# pandas>=1.5.0
# numpy>=1.24.0
# plotly>=5.15.0
# joblib>=1.2.0
# openai>=1.0.0
#
# Secrets (.streamlit/secrets.toml):
# GROQ_API_KEY = "tu-api-key-de-groq-aqui"
