# ============================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ============================================================
st.set_page_config(
    page_title="MentorAI | Descubre tu Camino",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos profesionales (Light Theme forzado)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    /* Fondo general */
    .stApp { background-color: #F8FAFC !important; font-family: 'Inter', sans-serif; }
    
    /* FORZAR COLOR DE TEXTO OSCURO PARA EVITAR INVISIBILIDAD EN MODO OSCURO */
    html, body, [class*="st-"] { color: #334155 !important; }
    h1, h2, h3, h4, h5, h6 { color: #0F172A !important; font-family: 'Inter', sans-serif; }
    p, span, label, div[data-testid="stMarkdownContainer"] { color: #475569 !important; }
    .subtitle { color: #64748B !important; font-size: 1.1rem; margin-bottom: 2rem; }
    
    /* Botón principal */
    div.stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%) !important;
        border: none !important; border-radius: 12px !important; padding: 0.75rem 2rem !important;
        width: 100% !important; transition: all 0.3s ease; box-shadow: 0 4px 10px rgba(37, 99, 235, 0.2);
    }
    /* Asegurar que el texto del botón siga siendo blanco */
    div.stButton > button p, div.stButton > button span { 
        color: white !important; font-weight: 800 !important; font-size: 1.1rem !important; 
    }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(37, 99, 235, 0.3); }
    
    /* Tarjetas de resultados */
    .result-card {
        background: white; border: 1px solid #E2E8F0; border-radius: 12px;
        padding: 1.5rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)