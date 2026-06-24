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
st.set_page_config(page_title="MentorAI - Tu guía vocacional", page_icon="🌟", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #1E293B !important;
    }

    .stApp {
        background: linear-gradient(135deg, #F0F4FF 0%, #F8FAFC 100%);
    }

    /* Main container */
    .main > .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Chat user bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background-color: #E0E7FF !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 10px 16px !important;
        margin-bottom: 8px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }

    /* Chat assistant bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background-color: #FFFFFF !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 10px 16px !important;
        margin-bottom: 8px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* Avatar styling */
    div[data-testid="chatAvatarIcon-user"] {
        background-color: #4F46E5 !important;
        color: white !important;
    }

    div[data-testid="chatAvatarIcon-assistant"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }

    /* Chat input */
    .stChatInput textarea {
        border-radius: 16px !important;
        border: 1.5px solid #E2E8F0 !important;
        background: white !important;
        padding: 12px 16px !important;
        font-family: 'Inter', sans-serif;
    }

    .stChatInput textarea:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* Success message */
    .stAlert {
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif;
    }

    /* Dividers */
    hr {
        margin-top: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        border-color: #E2E8F0 !important;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: white;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #F1F5F9;
    }
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
# 3. CONFIGURACIÓN DEL LLM (Groq)
# ============================================================
API_KEY = st.secrets.get("GROQ_API_KEY", "")

if not API_KEY:
    st.error("⚠️ No se encontró la API Key de Groq. Configúrala en los Secrets de Streamlit como `GROQ_API_KEY`.")
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
# 4. INICIALIZACIÓN DEL ESTADO DE LA SESIÓN
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "assistant",
            "content": (
                "¡Hola! Soy MentorAI 🌟. Cuéntame un poco sobre ti: "
                "¿qué actividades o materias disfrutas más en tu día a día?"
            )
        }
    ]
    st.session_state.finished = False

# ============================================================
# 5. FUNCIÓN AUXILIAR: EXTRAER JSON
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
# 6. ENCABEZADO
# ============================================================
col_logo, col_text = st.columns([1, 4])
with col_logo:
    st.markdown(
        "<div style='font-size:3.5rem; line-height:1; margin-top:8px;'>🌟</div>",
        unsafe_allow_html=True
    )
with col_text:
    st.markdown(
        "<h1 style='margin:0; padding:0; font-weight:700; color:#1E293B;'>MentorAI</h1>"
        "<p style='margin:0; padding:0; font-size:1rem; color:#64748B;'>"
        "Descubre tu vocación a través de una conversación inteligente</p>",
        unsafe_allow_html=True
    )
st.markdown("---")

# ============================================================
# 7. CHAT
# ============================================================
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input(
    "Escribe tu respuesta aquí..."
    if not st.session_state.finished
    else "La evaluación ha finalizado"
):
    if not st.session_state.finished:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
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
                        f"<h2 style='text-align:center; color:#1E293B;'>"
                        f"🎯 Tu carrera ideal: "
                        f"<span style='color:#3B82F6;'>{top_1['carrera'].replace('_', ' ').title()}</span>"
                        f"</h2>",
                        unsafe_allow_html=True
                    )

                    c1, c2 = st.columns(2, gap="medium")

                    with c1:
                        st.markdown(
                            "<p style='font-weight:600; color:#475569; "
                            "margin-bottom:6px;'>📊 Perfil de habilidades</p>",
                            unsafe_allow_html=True
                        )
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
                            fillcolor="rgba(59, 130, 246, 0.25)",
                            line=dict(color="#3B82F6", width=2.5),
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
                                    tickfont=dict(size=9, color="#64748B")
                                ),
                                bgcolor="rgba(0,0,0,0)"
                            ),
                            paper_bgcolor="rgba(0,0,0,0)",
                            showlegend=False,
                            height=340,
                            margin=dict(l=50, r=50, t=10, b=10)
                        )
                        st.plotly_chart(fig_radar, width="stretch")

                    with c2:
                        st.markdown(
                            "<p style='font-weight:600; color:#475569; "
                            "margin-bottom:6px;'>🏆 Top carreras recomendadas</p>",
                            unsafe_allow_html=True
                        )
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
                            textfont=dict(size=13, color="#1E293B", family="Inter")
                        ))
                        fig_bar.update_layout(
                            yaxis=dict(
                                autorange="reversed",
                                tickfont=dict(size=13, color="#1E293B", family="Inter")
                            ),
                            xaxis=dict(range=[0, 100], visible=False),
                            height=260,
                            margin=dict(l=10, r=40, t=10, b=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            showlegend=False
                        )
                        st.plotly_chart(fig_bar, width="stretch")

                    st.markdown("---")
                    st.markdown(
                        "<p style='text-align:center; color:#94A3B8; font-size:0.9rem;'>"
                        "¿No estás convencido? Puedes reiniciar la conversación y "
                        "explorar nuevas perspectivas.</p>",
                        unsafe_allow_html=True
                    )
                    if st.button(
                        "🔄 Volver a intentar",
                        use_container_width=True,
                        type="primary"
                    ):
                        for key in list(st.session_state.keys()):
                            del st.session_state[key]
                        st.rerun()

                else:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": llm_reply}
                    )
                    message_placeholder.markdown(llm_reply)

            except Exception as e:
                message_placeholder.error(f"Error de API: {e}")

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
