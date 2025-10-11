import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time 

# ----------------- Fórmulas de Momento de Inercia (Capítulo 9) -----------------
def calcular_momento_inercia(forma, masa, radio, longitud=None):
    """Calcula el momento de inercia Icm para diferentes geometrías respecto a un eje que pasa por el centro de masa."""
    if forma == "Disco/Cilindro Sólido":
        return 0.5 * masa * radio**2  # I_cm = (1/2)MR^2
    elif forma == "Cilindro Hueco (Anillo)":
        return masa * radio**2        # I_cm = MR^2
    elif forma == "Esfera Sólida":
        return (2/5) * masa * radio**2  # I_cm = (2/5)MR^2
    elif forma == "Varilla Delgada (L)":
        # Se asume que el radio 'R' del input es en realidad la longitud 'L' para este caso simple
        # Aunque esto es técnicamente I = (1/12)ML^2
        return (1/12) * masa * radio**2 if radio is not None else 0
    return 0

# ----------------- Funciones de Cálculo para las Simulaciones -----------------

def simular_torque(I, tau, t_max, dt=0.05):
    """Simulación 1: Dinámica y Cinemática Rotacional (Cap. 9 y 10)."""
    if I == 0:
        return pd.DataFrame(), 0.0

    # Dinámica Rotacional (Cap. 10): tau = I * alpha
    alfa = tau / I  # Aceleración angular constante
    
    tiempo = np.arange(0, t_max + dt, dt) 
    
    # Cinemática Rotacional (Cap. 9)
    omega = alfa * tiempo               # omega = omega_0 + alpha * t (con omega_0 = 0)
    theta = 0.5 * alfa * tiempo**2      # theta = theta_0 + 0.5 * alpha * t^2 (con theta_0, omega_0 = 0)
    vueltas = theta / (2 * np.pi)       # Número de vueltas
    
    df = pd.DataFrame({
        'Tiempo (s)': tiempo,
        'Velocidad Angular (rad/s)': omega,
        'Ángulo Girado (rad)': theta,
        'Número de Vueltas': vueltas,
        'Aceleración Angular (rad/s^2)': [alfa] * len(tiempo)
    })
    return df, alfa

def simular_masa_colgante(m_masa, R_cil, M_cil, t_max, dt=0.05):
    """Simulación 2: Energía y Dinámica (Cap. 9 y 10)."""
    g = 9.81  # Aceleración de la gravedad
    
    if m_masa <= 0 or R_cil <= 0 or M_cil <= 0:
        return pd.DataFrame(), 0, 0, 0

    I_cil = 0.5 * M_cil * R_cil**2  # Cilindro Sólido
    
    # Cálculo de la aceleración lineal de la masa 'a'
    a = g / (1 + I_cil / (m_masa * R_cil**2))
    alfa = a / R_cil 
    T = m_masa * (g - a) # Tensión
    
    tiempo = np.arange(0, t_max + dt, dt)
    h = 0.5 * a * tiempo**2  # Distancia que cae la masa
    
    K_rot = 0.5 * I_cil * (alfa * tiempo)**2  # Energía Rotacional (Cap. 9)
    K_tras = 0.5 * m_masa * (a * tiempo)**2  # Energía Traslacional
    
    # Energía Potencial Gravitacional (Cap. 9)
    h_max_caida = 0.5 * a * t_max**2
    U_grav_actual = m_masa * g * (h_max_caida - h)
    
    df = pd.DataFrame({
        'Tiempo (s)': tiempo,
        'Energía Rotacional (J)': K_rot,
        'Energía Traslacional (J)': K_tras,
        'Energía Potencial (J)': U_grav_actual,
        'Energía Total (J)': K_rot + K_tras + U_grav_actual
    })
    return df, a, alfa, T

def simular_rodadura(forma, masa, radio, angulo_deg, altura, t_max, dt=0.05):
    """Simulación de Rodadura sin Deslizamiento (Caso Extendido 2, Cap. 10)."""
    g = 9.81
    angulo_rad = np.radians(angulo_deg)
    
    I_cm = calcular_momento_inercia(forma, masa, radio)
    
    if forma == "Varilla Delgada (L)":
         st.warning("La Varilla Delgada no suele rodar. Usaremos la Esfera Sólida como caso de ejemplo para esta simulación.")
         I_cm = (2/5) * masa * radio**2
         
    if I_cm == 0 or masa == 0:
        return pd.DataFrame(), 0, 0, 0
    
    # Coeficiente c (I_cm = c * M * R^2)
    c = I_cm / (masa * radio**2)
    
    # Aceleración del Centro de Masa (Fórmula de Energía/Dinámica - Cap. 10)
    a_cm = (g * np.sin(angulo_rad)) / (1 + c)
    alfa = a_cm / radio
    
    # Tiempo para recorrer la distancia S
    S = altura / np.sin(angulo_rad)
    
    # Condición de detención/finalización
    t_fin = np.sqrt(2 * S / a_cm) if a_cm > 0 else t_max
    
    tiempo = np.arange(0, min(t_max, t_fin) + dt, dt)
    
    v_cm = a_cm * tiempo
    distancia = 0.5 * a_cm * tiempo**2
    
    df = pd.DataFrame({
        'Tiempo (s)': tiempo,
        'Velocidad CM (m/s)': v_cm,
        'Distancia Recorrida (m)': distancia,
        'Aceleración CM (m/s^2)': [a_cm] * len(tiempo),
        'Aceleración Angular (rad/s^2)': [alfa] * len(tiempo)
    })
    return df, a_cm, c, S


# ----------------- Funciones de Visualización 3D (Se mantienen) -----------------

def create_cylinder_mesh(radius, height, num_segments=50):
    """Crea una malla para un cilindro 3D."""
    z = np.linspace(-height/2, height/2, 2)
    theta = np.linspace(0, 2*np.pi, num_segments)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    return x_grid, y_grid, z_grid

def get_rotated_cylinder_data(x_base, y_base, z_base, angle):
    """Rota las coordenadas de un cilindro alrededor del eje Z."""
    Rz = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    coords = np.vstack([x_base.flatten(), y_base.flatten(), z_base.flatten()])
    rotated_coords = Rz @ coords
    x_rotated = rotated_coords[0].reshape(x_base.shape)
    y_rotated = rotated_coords[1].reshape(y_base.shape)
    z_rotated = rotated_coords[2].reshape(z_base.shape)
    return x_rotated, y_rotated, z_rotated

# ----------------- Configuración de la Interfaz Streamlit -----------------

st.set_page_config(layout="wide", page_title="Rotación de Cuerpos Rígidos (Sears Zemansky)")

st.title("Asistente Interactivo: Rotación de Cuerpos Rígidos 🌀")
st.write("Simulaciones basadas en los Capítulos 9 (Cinemática y Energía) y 10 (Dinámica y Momento Angular) de *Física Universitaria* (Sears, Zemansky, Freedman).")


# Selector de simulación en la barra lateral
opcion = st.sidebar.selectbox(
    "Selecciona la Simulación:",
    (
        "📚 Conceptos Fundamentales (Cap. 9 y 10)",
        "1️⃣ Dinámica Rotacional con Torque (Cap. 10)",
        "2️⃣ Energía: Masa Colgante en Cilindro Fijo (Cap. 9)",
        "3️⃣ Conservación del Momento Angular (Cap. 10)",
        "4️⃣ Casos Extendidos y Rodadura (Cap. 10)"
    )
)

# ----------------- Contenido de las Secciones -----------------

if opcion == "📚 Conceptos Fundamentales (Cap. 9 y 10)":
    st.header("Conceptos Clave de Dinámica y Cinemática Rotacional")
    st.markdown("""
    | Concepto | Lineal | Rotacional (Cap. 9) | Dinámica (Cap. 10) |
    | :--- | :---: | :---: | :---: |
    | Cantidad Base | Masa ($m$) | **Momento de Inercia** ($I$) | **Momento de Inercia** ($I$) |
    | Segunda Ley | $\\sum F = ma$ | N/A | **$\\sum \\tau = I \\alpha$** |
    | Energía Cinética | $K = 1/2 m v^2$ | **$K = 1/2 I \\omega^2$** | $K_{{Total}} = K_{{tras}} + K_{{rot}}$ |
    | Momento/Impulso | $p = mv$ | **Momento Angular** ($L = I \\omega$) | $L = I \\omega$ |
    """)
    st.info("¡Usa el menú lateral para seleccionar una simulación y experimentar virtualmente!")

# ---------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "1️⃣ Dinámica Rotacional con Torque (Cap. 10)":
    st.header("1. Torque y Cinemática Rotacional")
    st.markdown("Aplica un torque constante a una geometría. Analizamos la aceleración angular ($\\alpha$) y las variables de **Cinemática Rotacional** (Cap. 9).")

    # Controles de entrada
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forma = st.selectbox(
            "Selecciona la Forma (Determina $I$):",
            ("Disco/Cilindro Sólido", "Cilindro Hueco (Anillo)", "Esfera Sólida", "Varilla Delgada (L)")
        )
    with col2:
        masa = st.number_input("Masa ($M$, kg):", 0.1, 10.0, 2.0, 0.1)
        radio = st.number_input("Radio/Longitud ($R$ o $L$, m):", 0.1, 1.0, 0.5, 0.05)
    with col3:
        torque = st.number_input("Torque Aplicado ($\\tau$, N·m):", 0.1, 5.0, 1.0, 0.1)
        t_max = st.number_input("Tiempo de Simulación ($t_{max}$, s):", 1.0, 20.0, 5.0, 0.5)
        
    # --- Cálculos y Resultados ---
    I_cm = calcular_momento_inercia(forma, masa, radio) 
    
    if I_cm <= 0:
        st.error("Error: Momento de Inercia no válido. Asegúrate de que la masa y el radio/longitud sean positivos.")
    else:
        df_sim, alfa = simular_torque(I_cm, torque, t_max)
        
        st.markdown("---")
        st.subheader("Resultados Clave (Dinámica Rotacional)")
        st.latex(f"\\text{{Momento de Inercia: }} I = {I_cm:.4f} \\, \\text{{kg}} \\cdot \\text{{m}}^2")
        st.latex(f"\\text{{Aceleración Angular: }} \\alpha = \\frac{{\\tau}}{{I}} = {alfa:.4f} \\, \\text{{rad/s}}^2")

        # --- Visualización 3D (Solo para cilindro) ---
        if "Cilindro" in forma or "Disco" in forma:
            st.subheader("Visualización 3D Animada de la Rotación")
            
            # Usamos una altura pequeña para simular un disco/cilindro
            height = radio * 0.1 
            x_base, y_base, z_base = create_cylinder_mesh(radio, height)
            
            animation_placeholder = st.empty()
            
            if st.button("▶️ Iniciar Animación 3D", key="anim3d_1"):
                # Animación simplificada
                animation_steps = 100
                time_steps = np.linspace(0, t_max, animation_steps)
                
                for i in range(animation_steps):
                    current_time = time_steps[i]
                    current_theta = 0.5 * alfa * current_time**2
                    
                    x_rot, y_rot, z_rot = get_rotated_cylinder_data(x_base, y_base, z_base, current_theta)
                    
                    fig_3d = go.Figure(data=[
                        go.Surface(x=x_rot, y=y_rot, z=z_rot, colorscale='Plasma', opacity=0.8, showscale=False)
                    ])
                    
                    fig_3d.update_layout(
                        title=f"Tiempo: {current_time:.2f} s | Ángulo Girado: {np.degrees(current_theta) % 360:.1f}°",
                        scene_aspectmode='cube',
                        scene=dict(
                            xaxis=dict(range=[-radio*1.2, radio*1.2], visible=False),
                            yaxis=dict(range=[-radio*1.2, radio*1.2], visible=False),
                            zaxis=dict(range=[-height, height], visible=False),
                            aspectratio=dict(x=1, y=1, z=0.1),
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                        ),
                        margin=dict(l=0, r=0, b=0, t=40),
                        height=400
                    )
                    
                    animation_placeholder.plotly_chart(fig_3d, use_container_width=True)
                    time.sleep(t_max / animation_steps / 4) # Velocidad de simulación

        # --- Gráficas de Cinemática (Cap. 9) ---
        st.subheader("Gráficas de Cinemática Rotacional (Cap. 9)")
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
             fig_omega = px.line(
                df_sim, 
                x='Tiempo (s)', 
                y='Velocidad Angular (rad/s)', 
                title='Velocidad Angular ($\\omega$)'
            )
             st.plotly_chart(fig_omega, use_container_width=True)
        
        with col_g2:
             fig_theta = px.line(
                df_sim, 
                x='Tiempo (s)', 
                y='Ángulo Girado (rad)', 
                title='Ángulo Girado ($\\theta$)'
            )
             st.plotly_chart(fig_theta, use_container_width=True)

# ---------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "2️⃣ Energía: Masa Colgante en Cilindro Fijo (Cap. 9)":
    st.header("2. Energía en Rotación: Cilindro Fijo con Masa Colgante")
    st.markdown("La **Energía Potencial Gravitacional** ($U=Mgy_{cm}$) se transforma en **Energía Cinética de Traslación** y **Energía Cinética Rotacional** ($K=1/2 I \\omega^2$).")
    
    # Controles de entrada
    col1, col2 = st.columns(2)
    with col1:
        M_cil = st.number_input("Masa del Cilindro ($M_{cil}$, kg):", 0.1, 10.0, 2.0, 0.1, key="M_cil")
        R_cil = st.number_input("Radio del Cilindro ($R_{cil}$, m):", 0.05, 1.0, 0.2, 0.05, key="R_cil")
    with col2:
        m_masa = st.number_input("Masa Colgante ($m_{masa}$, kg):", 0.01, 5.0, 1.0, 0.01, key="m_masa")
        t_max = st.number_input("Tiempo de Simulación ($t_{max}$, s):", 0.5, 10.0, 3.0, 0.5, key="t_max_2")

    if R_cil <= 0 or m_masa <= 0 or M_cil <= 0:
        st.error("Todas las masas y el radio deben ser positivos para la simulación.")
    else:
        # --- Cálculos ---
        df_ener, a, alfa, T = simular_masa_colgante(m_masa, R_cil, M_cil, t_max)
        I_cil = 0.5 * M_cil * R_cil**2

        st.markdown("---")
        st.subheader("Análisis Dinámico y de Energía")
        st.latex(f"\\text{{Momento de Inercia del Cilindro: }} I = {I_cil:.4f} \\, \\text{{kg}} \\cdot \\text{{m}}^2")
        st.latex(f"\\text{{Aceleración Lineal de la masa: }} a = {a:.4f} \\, \\text{{m/s}}^2")

        # --- Visualización de Energía ---
        fig_ener = px.line(
            df_ener, 
            x='Tiempo (s)', 
            y=['Energía Rotacional (J)', 'Energía Traslacional (J)', 'Energía Potencial (J)', 'Energía Total (J)'], 
            title='Conversión y Conservación de Energía',
            labels={'value': 'Energía (J)', 'variable': 'Tipo de Energía'}
        )
        st.plotly_chart(fig_ener, use_container_width=True)
        
        st.subheader("Animación de la Masa Colgante")
        
        animation_placeholder = st.empty()
        
        if st.button("▶️ Iniciar Animación Masa Colgante", key="anim3d_2"):
            animation_steps = 50
            time_steps = np.linspace(0, t_max, animation_steps)
            
            for i in range(animation_steps):
                current_time = time_steps[i]
                h_caida = 0.5 * a * current_time**2
                current_theta = alfa * current_time # Ángulo girado

                # Coordenadas del cilindro
                height = R_cil * 0.2 
                x_base, y_base, z_base = create_cylinder_mesh(R_cil, height)
                x_rot, y_rot, z_rot = get_rotated_cylinder_data(x_base, y_base, z_base, current_theta)
                
                # Coordenadas de la masa (punto)
                x_masa = R_cil 
                y_masa = -R_cil - h_caida # Posición de la masa colgante

                fig_3d = go.Figure(data=[
                    # Cilindro
                    go.Surface(x=x_rot, y=y_rot, z=z_rot, colorscale='Blues', opacity=0.8, showscale=False),
                    # Masa
                    go.Scatter3d(x=[x_masa], y=[y_masa], z=[0], mode='markers', marker=dict(size=8, color='red'), name='Masa')
                ])
                
                fig_3d.update_layout(
                    title=f"Caída: {h_caida:.2f} m | Tensión: {T:.2f} N",
                    scene_aspectmode='cube',
                    scene=dict(
                        xaxis=dict(range=[-R_cil*1.5, R_cil*1.5], visible=False),
                        yaxis=dict(range=[-R_cil*2 - 0.5, R_cil], visible=False),
                        zaxis=dict(range=[-height, height], visible=False),
                        aspectratio=dict(x=1, y=1, z=0.5),
                        camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
                    ),
                    margin=dict(l=0, r=0, b=0, t=40),
                    height=500
                )
                
                animation_placeholder.plotly_chart(fig_3d, use_container_width=True)
                time.sleep(t_max / animation_steps / 2)


# ---------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "3️⃣ Conservación del Momento Angular (Cap. 10)":
    st.header("3. Conservación del Momento Angular ($L = I\\omega$)")
    st.markdown("Simulación del efecto **Patinador**. Si el **torque externo neto es cero**, el Momento Angular se conserva: $L_i = L_f$. Al reducir el radio de giro, el **Momento de Inercia ($I$)** disminuye y la **Velocidad Angular ($\\omega$)** debe aumentar para compensar.")
    
    # Controles de entrada
    col1, col2 = st.columns(2)
    with col1:
        M_cuerpo = st.number_input("Masa del Patinador (kg):", 40.0, 100.0, 60.0, 1.0)
        I_cuerpo = st.number_input("Momento de Inercia del Cuerpo Central ($I_{cm}$, kg·m²):", 0.5, 5.0, 1.0, 0.1, help="Parte del cuerpo cuya inercia no cambia.")
        R_ext_i = st.number_input("Radio Inicial de Brazos ($R_i$, m):", 0.5, 2.0, 1.5, 0.1)
    with col2:
        w_i = st.number_input("Velocidad Angular Inicial ($\\omega_i$, rad/s):", 0.1, 5.0, 1.0, 0.1)
        R_ext_f = st.number_input("Radio Final de Brazos ($R_f$, m):", 0.1, 1.0, 0.5, 0.1)
        
    st.markdown("---")

    # --- Cálculos ---
    # Se modelan los brazos/piernas como dos masas puntuales m/2
    m_brazos = M_cuerpo * 0.2 # 20% de la masa total en los brazos/extremidades
    
    # Momento de Inercia Inicial
    I_i = I_cuerpo + m_brazos * R_ext_i**2
    L_i = I_i * w_i
    
    # Momento de Inercia Final
    I_f = I_cuerpo + m_brazos * R_ext_f**2
    
    # Velocidad Angular Final (Conservación de L: Lf = Li => If * wf = Li)
    w_f = L_i / I_f

    # Cambio de Energía Cinética Rotacional
    K_i = 0.5 * I_i * w_i**2
    K_f = 0.5 * I_f * w_f**2
    delta_K = K_f - K_i

    st.subheader("Resultados de la Conservación de $L$")
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.markdown("**Estado Inicial (Brazos Abiertos):**")
        st.latex(f"I_{{i}} = {I_i:.4f} \\, \\text{{kg}} \\cdot \\text{{m}}^2")
        st.latex(f"L_{{i}} = I_{{i}} \\omega_{{i}} = {L_i:.4f} \\, \\text{{kg}} \\cdot \\text{{m}}^2/\\text{{s}}")
        st.latex(f"K_{{i}} = \\frac{{1}}{{2}} I_{{i}} \\omega_{{i}}^2 = {K_i:.4f} \\, \\text{{J}}")
        
    with col_res2:
        st.markdown("**Estado Final (Brazos Cerrados):**")
        st.latex(f"I_{{f}} = {I_f:.4f} \\, \\text{{kg}} \\cdot \\text{{m}}^2")
        st.latex(f"\\omega_{{f}} = \\frac{{L_{{i}}}}{{I_{{f}}}} = {w_f:.4f} \\, \\text{{rad/s}}")
        st.latex(f"K_{{f}} = \\frac{{1}}{{2}} I_{{f}} \\omega_{{f}}^2 = {K_f:.4f} \\, \\text{{J}}")
    
    st.info(f"El momento angular se conserva ($L_i = L_f$), pero la velocidad angular **aumenta** de {w_i:.2f} rad/s a **{w_f:.2f} rad/s**.")
    st.warning(f"El cambio en la Energía Cinética Rotacional es $\\Delta K = K_f - K_i = {delta_K:.2f}$ J. Esta energía es el **trabajo interno** realizado por el patinador al jalar sus brazos hacia adentro.")


# ------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "4️⃣ Casos Extendidos y Rodadura (Cap. 10)":
    st.header("4. Casos Extendidos: Rodadura, Energía y Sistemas Acoplados")
    st.markdown("Análisis de situaciones compuestas donde se combinan conceptos de inercia, energía y dinámica.")
    
    tab1, tab2, tab3 = st.tabs(["Rodadura en Plano Inclinado", "Eje con Discos Acoplados", "Análisis de Energía"])
    
    # --- Pestaña 1: Rodadura en Plano Inclinado (Cap. 10) ---
    with tab1:
        st.subheader("Rodadura Sin Deslizamiento")
        st.markdown("Compara la **aceleración del centro de masa ($a_{cm}$)** de diferentes geometrías que ruedan por un plano inclinado.")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            forma_rod = st.selectbox(
                "Selecciona la Forma:",
                ("Disco/Cilindro Sólido", "Cilindro Hueco (Anillo)", "Esfera Sólida"), key="forma_rod"
            )
            masa_rod = st.number_input("Masa ($M$, kg):", 0.1, 10.0, 1.0, 0.1, key="masa_rod")
            radio_rod = st.number_input("Radio ($R$, m):", 0.05, 1.0, 0.1, 0.05, key="radio_rod")
        
        with col_r2:
            angulo_rod = st.slider("Ángulo del Plano ($\\theta$, grados):", 5, 90, 30, 1, key="angulo_rod")
            altura_rod = st.number_input("Altura Vertical ($h$, m):", 0.1, 5.0, 1.0, 0.1, key="altura_rod")
            t_max_rod = st.number_input("Tiempo Máximo a Graficar (s):", 1.0, 10.0, 3.0, 0.5, key="t_max_rod")
            
        if st.button("Simular Rodadura", key="sim_rod"):
            df_rod, a_cm, c, S = simular_rodadura(forma_rod, masa_rod, radio_rod, angulo_rod, altura_rod, t_max_rod)
            
            st.markdown("---")
            st.subheader("Resultados de la Dinámica de Rodadura")
            st.latex(f"\\text{{Momento de Inercia: }} I_{{cm}} = {c:.2f} M R^2")
            st.latex(f"\\text{{Aceleración del Centro de Masa: }} a_{{cm}} = \\frac{{g \\sin \\theta}}{{1 + c}} = {a_cm:.4f} \\, \\text{{m/s}}^2")
            st.latex(f"\\text{{Distancia Total del Plano: }} S = {S:.4f} \\, \\text{{m}}")
            
            fig_rod = px.line(
                df_rod, 
                x='Tiempo (s)', 
                y=['Velocidad CM (m/s)', 'Distancia Recorrida (m)'], 
                title='Rodadura en el Plano Inclinado'
            )
            st.plotly_chart(fig_rod, use_container_width=True)

    # --- Pestaña 2: Eje con Discos Acoplados (Cap. 10) ---
    with tab2:
        st.subheader("Colisión Angular Inelástica (Acoplamiento de Discos)")
        st.markdown("Un disco que gira se acopla con otro disco inicialmente en reposo, formando un solo cuerpo. Se **conserva el Momento Angular**, pero se **pierde Energía Cinética** (colisión inelástica).")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Disco 1 (Inicialmente Girando)")
            M1 = st.number_input("Masa $M_1$ (kg):", 0.1, 10.0, 5.0, 0.1, key="M1")
            R1 = st.number_input("Radio $R_1$ (m):", 0.1, 1.0, 0.5, 0.1, key="R1")
            w1_i = st.number_input("Vel. Angular Inicial $\\omega_{1i}$ (rad/s):", 0.1, 10.0, 4.0, 0.1, key="w1i")
        
        with col_c2:
            st.markdown("#### Disco 2 (Inicialmente en Reposo)")
            M2 = st.number_input("Masa $M_2$ (kg):", 0.1, 10.0, 3.0, 0.1, key="M2")
            R2 = st.number_input("Radio $R_2$ (m):", 0.1, 1.0, 0.4, 0.1, key="R2")
            w2_i = 0.0
            st.markdown(f"Vel. Angular Inicial $\\omega_{{2i}}$: **{w2_i} rad/s**")
            
        if st.button("Calcular Acoplamiento", key="calc_coupling"):
            # Momento de Inercia de Discos Sólidos (I = 1/2 MR^2)
            I1 = 0.5 * M1 * R1**2
            I2 = 0.5 * M2 * R2**2
            
            # Momento Angular Inicial (L_i = I1*w1_i + I2*w2_i)
            L_i = I1 * w1_i
            
            # Momento de Inercia Final (I_f = I1 + I2)
            I_f = I1 + I2
            
            # Conservación de Momento Angular (L_f = L_i => I_f * w_f = L_i)
            w_f = L_i / I_f
            
            # Energía Cinética (K = 1/2 I w^2)
            K_i = 0.5 * I1 * w1_i**2
            K_f = 0.5 * I_f * w_f**2
            K_perdida = K_i - K_f
            
            st.markdown("---")
            st.subheader("Resultados del Acoplamiento")
            st.latex(f"\\text{{Momento de Inercia Total Final: }} I_{{f}} = I_{{1}} + I_{{2}} = {I_f:.4f} \\, \\text{{kg}} \\cdot \\text{{m}}^2")
            st.latex(f"\\text{{Velocidad Angular Final (Conservación de L): }} \\omega_{{f}} = \\frac{{L_{{i}}}}{{I_{{f}}}} = {w_f:.4f} \\, \\text{{rad/s}}")
            st.latex(f"\\text{{Pérdida de Energía Cinética: }} \\Delta K = K_{{i}} - K_{{f}} = {K_perdida:.4f} \\, \\text{{J}}")

    # --- Pestaña 3: Cálculo de Energías en Rotación (Cap. 9) ---
    with tab3:
        st.subheader("Cálculo de Energía Cinética Total ($K_{{Total}} = K_{{tras}} + K_{{rot}}$)")
        st.markdown("Calcula la energía cinética total de un cuerpo que tiene movimiento de traslación y rotación, como en el caso de la rodadura (Cap. 9, Ecuación 9.17).")
        
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            forma_e = st.selectbox("Forma del Objeto:", ("Disco/Cilindro Sólido", "Esfera Sólida"), key="forma_e")
            M_e = st.number_input("Masa ($M$, kg):", 0.1, 10.0, 5.0, 0.1, key="M_e")
            R_e = st.number_input("Radio ($R$, m):", 0.1, 1.0, 0.2, 0.1, key="R_e")
        with col_e2:
            v_cm_e = st.number_input("Velocidad de Traslación ($v_{cm}$, m/s):", 0.1, 10.0, 2.0, 0.1, key="v_cm_e")
            # Para rodadura, omega = v/R. Se da la opción de forzar rodadura.
            rodadura = st.checkbox("Rodadura Pura ($v_{cm} = R\\omega$)", value=True)
            if rodadura:
                w_e = v_cm_e / R_e
                st.markdown(f"Vel. Angular ($\\omega$): **{w_e:.2f} rad/s**")
            else:
                w_e = st.number_input("Velocidad Angular ($\\omega$, rad/s):", 0.1, 20.0, 5.0, 0.1, key="w_e")
                
        if st.button("Calcular Energías", key="calc_energies"):
            I_e = calcular_momento_inercia(forma_e, M_e, R_e)
            
            K_tras = 0.5 * M_e * v_cm_e**2
            K_rot = 0.5 * I_e * w_e**2
            K_total = K_tras + K_rot
            
            st.markdown("---")
            st.subheader("Componentes de la Energía Cinética")
            st.latex(f"I = {I_e:.4f} \\, \\text{{kg}} \\cdot \\text{{m}}^2")
            st.latex(f"K_{{tras}} = \\frac{{1}}{{2}} M v_{{cm}}^2 = {K_tras:.4f} \\, \\text{{J}}")
            st.latex(f"K_{{rot}} = \\frac{{1}}{{2}} I \\omega^2 = {K_rot:.4f} \\, \\text{{J}}")
            st.latex(f"K_{{Total}} = K_{{tras}} + K_{{rot}} = {K_total:.4f} \\, \\text{{J}}")

