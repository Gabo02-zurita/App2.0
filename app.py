import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# --- Configuración Inicial de la Página ---
st.set_page_config(
    page_title="Dinámica de Rotación de Sólidos Rígidos",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚙️ Aplicación Interactiva: Dinámica del Sólido Rígido")
st.markdown("""
Esta herramienta modela y explica fenómenos clave de la rotación, utilizando la física clásica y visualizaciones interactivas.
""")

# --------------------------------------------------------------------------------------
# 1. Funciones Físicas de Soporte
# --------------------------------------------------------------------------------------

def momento_inercia(forma, masa, dimension):
    """Calcula el momento de inercia I. 'dimension' es R para discos/cilindros o L para varilla."""
    R = dimension
    M = masa
    
    # Fórmulas de Momento de Inercia (I) para el eje de rotación central:
    if forma == "Anillo/Cilindro Hueco":
        return M * R**2
    elif forma == "Cilindro Sólido/Disco":
        return 0.5 * M * R**2
    elif forma == "Varilla (Eje central)":
        return (1/12) * M * R**2
    elif forma == "Esfera Sólida":
        return 0.4 * M * R**2 # (2/5)MR^2
    return 0

def factor_inercial(forma):
    """Retorna el factor C = I / (M*R^2) para rodadura."""
    if forma == "Esfera Sólida":
        return 0.4  # C = 2/5
    elif forma == "Cilindro Sólido/Disco":
        return 0.5  # C = 1/2
    elif forma == "Anillo/Cilindro Hueco":
        return 1.0  # C = 1

# --------------------------------------------------------------------------------------
# 2. Organización de la Aplicación en Pestañas
# --------------------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Torque y $\mathbf{I}$ Variable",
    "2. Masa Colgante (Polea)",
    "3. Conservación del Momento Angular",
    "4. Rodadura en Plano Inclinado (Caso Extendido)"
])

# ====================================================================
# --- PESTAÑA 1: Torque Fijo y Momento de Inercia Variable ---
# ====================================================================
with tab1:
    st.header("1. Rotación con Torque Constante ($\mathbf{\\tau} = I \\alpha$)")
    st.markdown("""
    Explore cómo el **Momento de Inercia ($I$)** de un objeto influye en su **Aceleración Angular ($\alpha$)**
    cuando se aplica un **Torque ($\tau$)** constante.
    """)

    col_c1, col_c2 = st.columns([1, 2])

    with col_c1:
        st.subheader("Controles Físicos")
        forma_t1 = st.selectbox(
            "Seleccione la Forma del Sólido:",
            ["Cilindro Sólido/Disco", "Anillo/Cilindro Hueco", "Varilla (Eje central)"]
        )
        masa_t1 = st.number_input("Masa (M, kg)", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
        radio_t1 = st.number_input("Dimensión (R/L, m)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, help="Radio para cilindro/disco o Longitud para varilla.")
        torque_t1 = st.number_input("Torque Aplicado ($\\tau$, N·m)", min_value=0.1, max_value=50.0, value=5.0, step=0.5)
        tiempo_t1 = st.slider("Tiempo de Simulación (t, s)", 1.0, 10.0, 5.0, 0.5)

    # --- Cálculos ---
    I_t1 = momento_inercia(forma_t1, masa_t1, radio_t1)
    alfa_t1 = torque_t1 / I_t1 # Aceleración angular cte: alpha = tau / I
    
    tiempo_eje = np.linspace(0, tiempo_t1, 100)
    omega_t = alfa_t1 * tiempo_eje
    angulo_t = 0.5 * alfa_t1 * tiempo_eje**2
    vueltas_t = angulo_t / (2 * np.pi)

    with col_c2:
        st.subheader("Resultados y Fundamentos Teóricos")
        
        st.info(f"**Momento de Inercia ($I$):** ${I_t1:.4f}$ kg·m²")
        st.info(f"**Aceleración Angular ($\\alpha$):** ${alfa_t1:.4f}$ rad/s²")
        
        st.markdown(f"""
        **Cálculos a $t = {tiempo_t1}s$:**
        - **Velocidad Angular Final ($\omega$):** ${omega_t[-1]:.2f}$ rad/s
        - **Ángulo Girado ($\\theta$):** ${angulo_t[-1]:.2f}$ rad
        - **Número de Vueltas:** ${vueltas_t[-1]:.2f}$ vueltas
        
        ---
        
        **Explicación Física:**
        La **Segunda Ley de Newton para la Rotación** ($\mathbf{\\tau} = I \\alpha$) establece que
        un mayor momento de inercia ($I$) resulta en una menor aceleración angular ($\alpha$)
        para el mismo torque aplicado ($\tau$). $I$ es una medida de la resistencia de un
        objeto a cambiar su estado de movimiento rotacional.
        """)
    
    # --- Visualización Gráfica ---
    st.subheader("Visualización del Movimiento Cinemático")
    
    fig_t1 = go.Figure()
    fig_t1.add_trace(go.Scatter(x=tiempo_eje, y=omega_t, mode='lines', name='Velocidad Angular ($\omega$)', line=dict(color='blue')))
    fig_t1.add_trace(go.Scatter(x=tiempo_eje, y=angulo_t, mode='lines', name='Ángulo Girado ($\\theta$)', line=dict(color='red', dash='dash')))
    
    fig_t1.update_layout(
        xaxis_title="Tiempo (s)",
        yaxis_title="Valor",
        title="$\mathbf{\\omega}$ y $\mathbf{\\theta}$ vs. Tiempo (Movimiento Uniformemente Acelerado)",
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig_t1, use_container_width=True)

# ====================================================================
# --- PESTAÑA 2: Masa Colgante (Polea) ---
# ====================================================================
with tab2:
    st.header("2. Cilindro Giratorio con Masa Colgante")
    st.markdown("""
    Modelo de una masa colgante que, al caer, desenrolla un cable y provoca la rotación
    de un cilindro fijo (análogo a una máquina de Atwood rotacional).
    El movimiento acopla la dinámica **traslacional** y **rotacional**.
    """)

    col_m1, col_m2 = st.columns([1, 2])
    g_m2 = 9.81
    
    with col_m1:
        st.subheader("Parámetros del Sistema")
        masa_cilindro_m2 = st.number_input("Masa del Cilindro (M, kg)", min_value=0.5, max_value=10.0, value=4.0, step=0.5)
        radio_cilindro_m2 = st.number_input("Radio del Cilindro (R, m)", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
        masa_colgante_m2 = st.number_input("Masa Colgante (m, kg)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        altura_m2 = st.slider("Altura de Caída (h, m)", 0.5, 5.0, 2.0, 0.5)

    # --- Cálculos ---
    # Asumimos Cilindro Sólido: I = 0.5 * M * R^2
    I_m2 = 0.5 * masa_cilindro_m2 * radio_cilindro_m2**2
    
    # Ecuación de la aceleración lineal (a):
    # a = g / (1 + I / (m * R^2))
    a_m2 = g_m2 / (1 + I_m2 / (masa_colgante_m2 * radio_cilindro_m2**2))
    alpha_m2 = a_m2 / radio_cilindro_m2
    
    # Tiempo de caída: t = sqrt(2h / a)
    tiempo_caida_m2 = np.sqrt(2 * altura_m2 / a_m2)
    
    # Tensión: T = m(g - a)
    T_m2 = masa_colgante_m2 * (g_m2 - a_m2)
    
    with col_m2:
        st.subheader("Resultados de la Dinámica")
        
        st.metric("Aceleración Lineal de la Masa ($a$)", f"{a_m2:.4f} m/s²")
        st.metric("Aceleración Angular del Cilindro ($\\alpha$)", f"{alpha_m2:.4f} rad/s²")
        st.metric("Tensión del Cable ($T$)", f"{T_m2:.4f} N")
        st.metric("Tiempo Total de Caída", f"{tiempo_caida_m2:.2f} s")
        
        st.markdown(f"""
        **Fundamentos:**
        1. **Traslación ($m$):** $mg - T = ma$.
        2. **Rotación ($M$):** $T R = I \\alpha$.
        3. **Restricción (No desliza):** $a = \\alpha R$.
        
        La aceleración ($a$) es siempre menor que $g$ porque la **Tensión ($T$)** hace un trabajo negativo sobre la masa
        y un trabajo positivo sobre el cilindro (generando la rotación). La energía potencial se convierte en energía
        cinética **traslacional** y **rotacional**.
        """)

# ====================================================================
# --- PESTAÑA 3: Conservación del Momento Angular ---
# ====================================================================
with tab3:
    st.header("3. Conservación del Momento Angular ($\mathbf{L} = I \\omega$)")
    st.markdown("""
    Modelación de un patinador o bailarín que cambia su distribución de masa (brazos) para
    demostrar cómo la **Velocidad Angular ($\omega$)** debe cambiar para mantener el
    **Momento Angular ($\mathbf{L}$) constante**.
    """)

    col_p1, col_p2 = st.columns([1, 2])

    with col_p1:
        st.subheader("Parámetros del Patinador")
        I_inicial = st.number_input("Momento de Inercia Inicial ($I_{in}$, brazos abiertos)", min_value=5.0, max_value=30.0, value=15.0, step=0.5)
        I_final = st.number_input("Momento de Inercia Final ($I_{fin}$, brazos cerrados)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
        omega_inicial = st.number_input("Velocidad Angular Inicial ($\omega_{in}$, rad/s)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    # --- Cálculos ---
    # L_inicial = L_final  =>  I_inicial * omega_inicial = I_final * omega_final
    omega_final = (I_inicial * omega_inicial) / I_final
    
    # Energía Cinética: Ek = 0.5 * I * omega^2
    Ek_inicial = 0.5 * I_inicial * omega_inicial**2
    Ek_final = 0.5 * I_final * omega_final**2
    
    aumento_velocidad = ((omega_final / omega_inicial) - 1) * 100

    with col_p2:
        st.subheader("Resultados de la Conservación")
        
        st.metric("Momento Angular ($L$)", f"{(I_inicial * omega_inicial):.2f} kg·m²/s (Constante)")
        st.metric("Velocidad Angular Final ($\omega_{fin}$)", f"{omega_final:.2f} rad/s")
        st.metric("Aumento de Velocidad", f"{aumento_velocidad:.1f} %")
        st.metric("Trabajo Realizado ($\\Delta E_k$)", f"{(Ek_final - Ek_inicial):.2f} J")
        
        st.markdown("""
        **Principio Físico:**
        Cuando el torque externo neto es cero ($\\tau_{ext} = 0$), el **Momento Angular ($\mathbf{L}$) se conserva**.
        Al disminuir el Momento de Inercia ($I$) (ej. brazos cerrados), la velocidad angular ($\omega$)
        debe aumentar para compensar.
        
        **¡Ojo con la Energía!** La **Energía Cinética de Rotación ($E_k$) NO se conserva**; el aumento de $E_k$
        proviene del trabajo interno realizado por el patinador al acercar sus brazos.
        """)
        
    # --- Visualización (Plotly para énfasis) ---
    st.subheader("Visualización Comparativa $I$ vs. $\\omega$")
    fig_p = go.Figure(data=[
        go.Bar(name='Inicial', x=['Inercia $I$ (kg·m²)', 'Velocidad $\\omega$ (rad/s)', 'Energía $E_k$ (J)'], y=[I_inicial, omega_inicial, Ek_inicial], marker_color='blue'),
        go.Bar(name='Final', x=['Inercia $I$ (kg·m²)', 'Velocidad $\\omega$ (rad/s)', 'Energía $E_k$ (J)'], y=[I_final, omega_final, Ek_final], marker_color='red')
    ])
    fig_p.update_layout(barmode='group', title="Comparación de Estados Inicial y Final", height=400)
    st.plotly_chart(fig_p, use_container_width=True)

# ====================================================================
# --- PESTAÑA 4: Caso Extendido: Rodadura en Plano Inclinado ---
# ====================================================================
with tab4:
    st.header("4. Rodadura de Formas en un Plano Inclinado (La Carrera)")
    st.markdown("""
    Simulación que compara la velocidad de objetos con diferentes geometrías (y, por tanto, diferentes
    momentos de inercia) que ruedan sin deslizar por un plano inclinado.
    """)

    col_i1, col_i2 = st.columns([1, 2])
    g_i4 = 9.81
    
    with col_i1:
        st.subheader("Parámetros del Plano")
        angulo_inclinacion = st.slider("Ángulo de Inclinación ($\\theta$, grados)", 5, 60, 30, 1)
        longitud_plano = st.slider("Longitud del Plano Inclinado (L, m)", 1.0, 10.0, 5.0, 0.5)
        st.markdown(f"**Altura Inicial ($h$):** ${longitud_plano * np.sin(np.radians(angulo_inclinacion)):.2f}$ m")
    
    # --- Cálculos ---
    # a = g * sin(theta) / (1 + C), donde C = I / (M * R^2) es el factor inercial.
    
    formas_i4 = ["Esfera Sólida", "Cilindro Sólido/Disco", "Anillo/Cilindro Hueco"]
    resultados_i4 = []
    theta_rad = np.radians(angulo_inclinacion)

    for nombre in formas_i4:
        C = factor_inercial(nombre)
        a_i4 = (g_i4 * np.sin(theta_rad)) / (1 + C)
        t_i4 = np.sqrt(2 * longitud_plano / a_i4)
        v_final_i4 = a_i4 * t_i4
        
        resultados_i4.append({
            "Forma": nombre,
            "Factor Inercial (C)": C,
            "Aceleración ($a$, m/s²)": a_i4,
            "Tiempo de Recorrido ($t$, s)": t_i4,
            "Velocidad Final ($v$, m/s)": v_final_i4
        })

    df_resultados = pd.DataFrame(resultados_i4)
    df_resultados_sorted = df_resultados.sort_values(by="Tiempo de Recorrido ($t$, s)")
    
    with col_i2:
        st.subheader("Tabla de Resultados y Ganador")
        st.dataframe(df_resultados_sorted.set_index('Forma'))

        st.markdown(f"""
        **Explicación Fundamental:**
        El objeto que gana la carrera es la **{df_resultados_sorted.iloc[0]['Forma']}** (Factor $C={df_resultados_sorted.iloc[0]['Factor Inercial (C)']:.1f}$),
        ya que tiene la mayor aceleración lineal.
        
        **¿Por qué?**
        La Energía Potencial Inicial ($E_p = mgh$) se divide en Energía Cinética Traslacional ($E_{k,trasl}$)
        y Rotacional ($E_{k,rot}$). El factor inercial $C$ determina qué parte de la energía se 'gasta' en rotación.
        **A menor $C$ (masa más cerca del eje)**, más energía se dirige al movimiento lineal,
        ¡lo que resulta en mayor aceleración y menor tiempo!
        """)
        
    # --- Visualización de la Comparación de Tiempos (Plotly) ---
    st.subheader("Gráfica: ¿Quién llega primero?")
    fig_bar = px.bar(
        df_resultados_sorted,
        x="Forma",
        y="Tiempo de Recorrido ($t$, s)",
        title="Comparación del Tiempo Total de Rodadura",
        color="Tiempo de Recorrido ($t$, s)",
        color_continuous_scale=px.colors.sequential.Inferno_r,
        labels={"Tiempo de Recorrido ($t$, s)": "Tiempo (s)"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)
