import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ============================================================
# 1. CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="MentorAI | Descubre tu Camino",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos profesionales (Light Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .stApp { background-color: #F8FAFC; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #0F172A !important; }
    .subtitle { color: #64748B; font-size: 1.1rem; margin-bottom: 2rem; }
    
    /* Botón principal */
    div.stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%) !important;
        color: white !important; font-weight: 800 !important; font-size: 1.1rem !important;
        border: none !important; border-radius: 12px !important; padding: 0.75rem 2rem !important;
        width: 100% !important; transition: all 0.3s ease; box-shadow: 0 4px 10px rgba(37, 99, 235, 0.2);
    }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(37, 99, 235, 0.3); }
    
    /* Tarjetas de resultados */
    .result-card {
        background: white; border: 1px solid #E2E8F0; border-radius: 12px;
        padding: 1.5rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .rank-badge {
        background: #EEF2FF; color: #4F46E5; font-weight: 800; padding: 0.2rem 0.8rem;
        border-radius: 999px; font-size: 0.9rem; margin-bottom: 0.5rem; display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. CARGA DEL MODELO MOTOR COMPLETO
# ============================================================
@st.cache_resource
def load_mentor_engine():
    model_path = "modelos/motor_completo.joblib"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.sidebar.error(f"Error cargando modelo: {e}")
            return None
    else:
        # MOCK ENGINE: Solo para que puedas ver la UI mientras subes tu modelo real
        class MockEngine:
            def recommend(self, perfil, top_k=5, include_details=True):
                import random
                carreras = ["Datos e IA", "Tecnología Core", "Diseño UX/UI", "Negocios Tech", "Marketing", "Ciberseguridad", "Finanzas"]
                random.shuffle(carreras)
                return [{"rank": i+1, "carrera": c, "confidence": round(random.uniform(60, 95) - (i*5), 1)} for i, c in enumerate(carreras[:top_k])]
        return MockEngine()

engine = load_mentor_engine()

# ============================================================
# 3. INTERFAZ PRINCIPAL
# ============================================================
st.title("🧠 MentorAI")
st.markdown("<p class='subtitle'>Evaluación vocacional potenciada por Inteligencia Artificial híbrida (XGBoost + Redes O*NET)</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# 4. FORMULARIO DE EVALUACIÓN (13 Rasgos Base)
# ============================================================
st.markdown("### 📊 Construye tu Perfil")
st.write("Evalúa tus habilidades e intereses en una escala del 1 (Nada) al 10 (Excelente/Me apasiona).")

with st.form("mentor_form"):
    c1, c2, c3 = st.columns(3, gap="large")
    
    with c1:
        st.markdown("#### 🧠 Cognitivo")
        analytical = st.slider("Capacidad Analítica", 1, 10, 5)
        logical_reasoning = st.slider("Razonamiento Lógico", 1, 10, 5)
        problem_solving = st.slider("Resolución de Problemas", 1, 10, 5)
        
        st.markdown("#### 🎨 Creativo")
        creativity = st.slider("Creatividad e Innovación", 1, 10, 5)
        design = st.slider("Diseño Visual y Estética", 1, 10, 5)

    with c2:
        st.markdown("#### 🤝 Social y Equipo")
        communication = st.slider("Comunicación (Oral/Escrita)", 1, 10, 5)
        empathy = st.slider("Empatía / Inteligencia Emocional", 1, 10, 5)
        social = st.slider("Habilidades Interpersonales", 1, 10, 5)
        teamwork = st.slider("Trabajo en Equipo", 1, 10, 5)
        leadership = st.slider("Liderazgo y Dirección", 1, 10, 5)

    with c3:
        st.markdown("#### 💻 Técnico y Negocios")
        technology = st.slider("Aptitud Tecnológica", 1, 10, 5)
        business = st.slider("Visión de Negocio", 1, 10, 5)
        
        st.markdown("#### 🛡️ Personal")
        stress_tolerance = st.slider("Tolerancia al Estrés", 1, 10, 5)
        
        st.markdown("#### 🎓 Perfil General")
        age = st.number_input("Edad", min_value=15, max_value=80, value=20)
        edu_str = st.selectbox("Nivel Educativo Actual", ["Secundaria", "Bachiller / Universitario", "Maestría", "Doctorado"])

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("Analizar mi Perfil Vocacional 🚀")

# ============================================================
# 5. INFERENCIA Y RESULTADOS
# ============================================================
if submit:
    # Mapeo de Educación según documentación: high_school=0.3, bachelor=0.5, master=0.7, phd=0.9
    edu_map = {"Secundaria": 0.3, "Bachiller / Universitario": 0.5, "Maestría": 0.7, "Doctorado": 0.9}
    
    # Normalización de variables [0, 1]
    perfil_usuario = {
        "analytical": analytical / 10.0,
        "logical_reasoning": logical_reasoning / 10.0,
        "problem_solving": problem_solving / 10.0,
        "creativity": creativity / 10.0,
        "design": design / 10.0,
        "communication": communication / 10.0,
        "empathy": empathy / 10.0,
        "social": social / 10.0,
        "teamwork": teamwork / 10.0,
        "leadership": leadership / 10.0,
        "technology": technology / 10.0,
        "business": business / 10.0,
        "stress_tolerance": stress_tolerance / 10.0,
        "education": edu_map[edu_str],
        "age": min(age / 65.0, 1.0) # Normalización básica de edad
    }

    st.markdown("---")
    st.markdown("### 🎯 Resultados de MentorAI")
    
    with st.spinner("Computando modelos XGBoost, similitud coseno y diversificando con MMR..."):
        if engine:
            try:
                # Llamada al motor completo como indica tu README
                recomendaciones = engine.recommend(perfil_usuario, top_k=5, include_details=True)
                
                # Desempaquetar resultados para visualización
                df_res = pd.DataFrame(recomendaciones)
                
                # Mostrar el Top 1 Destacado
                top_1 = df_res.iloc[0]
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #EEF2FF, #E0E7FF); border: 2px solid #C7D2FE; border-radius: 16px; padding: 2rem; text-align: center; margin-bottom: 2rem;'>
                    <h4 style='color: #4F46E5; margin:0;'>✨ TU CARRERA IDEAL ✨</h4>
                    <h1 style='color: #1E293B; margin: 0.5rem 0;'>{top_1['carrera'].replace('_', ' ').title()}</h1>
                    <p style='color: #475569; font-size: 1.1rem; margin:0;'>Confianza del modelo: <b>{top_1['confidence']}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Gráfico de barras horizontal para el top 5
                fig = go.Figure(go.Bar(
                    x=df_res['confidence'], 
                    y=df_res['carrera'].str.replace('_', ' ').str.title(),
                    orientation='h',
                    marker=dict(
                        color=df_res['confidence'],
                        colorscale='Blues',
                        line=dict(color='#2563EB', width=1)
                    ),
                    text=[f"{v}%" for v in df_res['confidence']],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Afinidad por Macro-Categorías Vocacionales",
                    xaxis=dict(title="Nivel de Compatibilidad (%)", range=[0, 100]),
                    yaxis=dict(autorange="reversed"),
                    height=350,
                    margin=dict(l=0, r=0, t=40, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error en la inferencia del modelo: {str(e)}\n\nAsegúrate de que la clase Motor tenga el método 'recommend' activo.")
        else:
            st.error("El motor no se pudo cargar. Revisa los archivos .joblib en la carpeta 'modelos/'.")