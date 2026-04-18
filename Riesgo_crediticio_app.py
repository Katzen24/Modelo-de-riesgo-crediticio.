import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Simulador de Riesgo Crediticio",
    page_icon="🏦",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

directorio_app = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def cargar_recursos():
    ruta_modelo = os.path.join(directorio_app, 'modelo_riesgo.pkl')
    ruta_columnas = os.path.join(directorio_app, 'columnas_modelo.pkl')
    
    try:
        model = joblib.load(ruta_modelo)
        cols = joblib.load(ruta_columnas)
        return model, cols
    except FileNotFoundError:
        st.error(f"No se encontraron los archivos en {directorio_app}. Asegúrate de ejecutar el entrenamiento primero.")
        return None, None

modelo, columnas_entrenamiento = cargar_recursos()

st.title("🏦 Sistema de Evaluación de Créditos")
st.markdown("---")
st.info("Introduce los datos del cliente para evaluar la probabilidad de impago.")

if modelo is not None:
    st.subheader("⚙️ Configuración de Tolerancia al Riesgo")
    umbral_porcentaje = st.slider(
        "Umbral de Rechazo (%)", 
        min_value=10, max_value=90, value=50, step=5
    )
    umbral = umbral_porcentaje / 100.0
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        monto = st.number_input("Monto del Préstamo ($)", min_value=500, max_value=40000, value=10000)
        ingresos = st.number_input("Ingresos Anuales ($)", min_value=1000, value=50000)
        tasa = st.slider("Tasa de Interés Anual (%)", 5.0, 35.0, 12.0)
        plazo = st.selectbox("Plazo (Meses)", [36, 60])
        verificacion = st.selectbox("Estado de Verificación", ["Not Verified", "Source Verified", "Verified"])

    with col2:
        grado = st.slider("Grado de Riesgo (1=A, 7=G)", 1, 7, 2)
        dti = st.slider("Relación Deuda/Ingreso (DTI)", 0.0, 100.0, 15.0)
        antiguedad = st.slider("Antigüedad Laboral (Años)", 0, 10, 5)
        vivienda = st.selectbox("Tipo de Vivienda", ["MORTGAGE", "RENT", "OWN", "ANY"])

    st.markdown("---")

    if st.button("Evaluar Solicitud de Crédito"):
        datos_cliente = pd.DataFrame(0, index=[0], columns=columnas_entrenamiento)
        
        datos_cliente['loan_amnt'] = monto
        datos_cliente['annual_inc'] = ingresos
        datos_cliente['int_rate'] = tasa
        datos_cliente['term'] = plazo
        datos_cliente['grade'] = grado
        datos_cliente['dti'] = dti
        datos_cliente['emp_length'] = antiguedad
        
        tasa_mensual = (tasa / 100) / 12
        if tasa_mensual > 0:
            cuota = monto * (tasa_mensual * (1 + tasa_mensual)**plazo) / ((1 + tasa_mensual)**plazo - 1)
        else:
            cuota = monto / plazo
        datos_cliente['installment'] = cuota
        
        col_vivienda = f"home_ownership_{vivienda}"
        if col_vivienda in datos_cliente.columns:
            datos_cliente[col_vivienda] = 1
            
        col_verif = f"verification_status_{verificacion}"
        if col_verif in datos_cliente.columns:
            datos_cliente[col_verif] = 1

        probabilidad = modelo.predict_proba(datos_cliente)[0][1]
        
        st.subheader("Resultado de la Evaluación:")
        
        if probabilidad > umbral:
            st.error("CRÉDITO RECHAZADO")
            st.warning(f"Riesgo de Default detectado: **{probabilidad:.2%}** (Supera el límite de {umbral_porcentaje}%)")
        else:
            st.success("CRÉDITO APROBADO")
            st.info(f"Probabilidad de impago baja: **{probabilidad:.2%}** (Dentro del límite seguro)")

        st.progress(probabilidad)

        st.markdown("---")
        st.subheader("📊 ¿Qué factores pesaron más en esta decisión?")
        st.write("El modelo considera múltiples variables, pero estas son las que definen principalmente el perfil de riesgo:")
        
        importancias = modelo.feature_importances_
        
        diccionario_nombres = {
            'int_rate': 'Tasa de Interés',
            'dti': 'Relación Deuda/Ingreso',
            'annual_inc': 'Ingresos Anuales',
            'installment': 'Cuota Mensual',
            'loan_amnt': 'Monto del Préstamo',
            'emp_length': 'Antigüedad Laboral',
            'term': 'Plazo (Meses)',
            'grade': 'Grado de Riesgo (Interno)'
        }
        
        nombres_mostrar = [diccionario_nombres.get(col, col.replace('_', ' ').title()) for col in columnas_entrenamiento]
        
        df_importancias = pd.DataFrame({
            'Factor': nombres_mostrar,
            'Nivel de Impacto (%)': importancias * 100
        }).sort_values(by='Nivel de Impacto (%)', ascending=True).tail(7)
        
        st.bar_chart(df_importancias.set_index('Factor'), horizontal=True)