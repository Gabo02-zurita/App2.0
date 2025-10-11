import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time 

# ----------------- CSS para el Fondo y Estilo General -----------------
st.markdown(
    """
    <style>
    /* Estilo del cuerpo para el fondo */
    body {
        background-image: url('https://www.transparenttextures.com/patterns/circles.png'); /* Patrón sutil */
        background-color: #f0f2f6; /* Color de fallback */
        background-size: cover;
        background-attachment: fixed;
        color: #333333; /* Color de texto principal */
    }
    .stApp {
        background-image: url('https://www.transparenttextures.com/patterns/circles.png'); 
        background-size: cover;
        background-attachment: fixed;
    }

    /* Estilo para el título principal */
    .stApp .st-emotion-cache-10q0ysb { /* Clase para el h1 de Streamlit */
        color: #2a5a9d; /* Un azul más oscuro */
        text-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Estilo para los headers (h2, h3) */
    h2, h3 {
        color: #3a7bd5; /* Un azul vibrante */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Estilo para el sidebar */
    .st-emotion-cache-vk330y { /* Contenedor del sidebar */
        background-color: rgba(255, 255, 255, 0.85); /* Fondo blanco semitransparente */
        border-right: 1px solid #ddd;
        box-shadow: 2px 0px 5px rgba(0,0,0,0.05);
    }
    .st-emotion-cache-vk330y .st-emotion-cache-1jmve37 { /* Título del selectbox */
        color: #2a5a9d;
        font-weight: bold;
    }
    
    /* Estilo para las tarjetas de información (st.info, st.warning, st.success) */
    .stAlert {
        border-radius: 10px;
        background-color: rgba(240, 248, 255, 0.9); /* Un tono muy claro de azul */
        color: #333333;
        border-left: 5px solid #3a7bd5;
    }
    .stAlert > div > span { /* Texto dentro de st.info */
        color: #333333 !important;
    }
    .stAlert.st-emotion-cache-1c9k0u9 { /* Para info */
        background-color: rgba(230, 240, 255, 0.9);
        border-left: 5px solid #3a7bd5;
    }
     .stAlert.st-emotion-cache-q8spsl { /* Para success */
        background-color: rgba(220, 255, 220, 0.9);
        border-left: 5px solid #28a745;
    }
    .stAlert.st-emotion-cache-e3g6i8 { /* Para warning */
        background-color: rgba(255, 248, 220, 0.9);
        border-left: 5px solid #ffc107;
    }

    /* Estilo de los botones */
    .stButton > button {
        background-color: #3a7bd5; /* Azul vibrante */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #2a5a9d; /* Azul más oscuro al pasar el mouse */
        color: white;
    }

    /* Estilo para tablas y dataframes */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-radius: 8px;
        overflow: hidden;
    }
    .dataframe th {
        background-color: #3a7bd5;
        color: white;
        padding: 10px;
        text-align: left;
    }
    .dataframe td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
    }
    .dataframe tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    .dataframe tr:hover {
        background-color: #e6f2ff;
    }

    /* Estilo para los tabs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.1rem;
        font-weight: bold;
        color: #2a5a9d;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(240, 242, 246, 0.8);
        border-radius: 8px 8px 0 0;
        margin: 0 5px;
        padding: 10px 15px;
        border-bottom: 3px solid transparent;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: white;
        border-bottom: 3px solid #3a7bd5; /* Indicador azul para el tab activo */
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- Fórmulas de Momento de Inercia (Capítulo 9) -----------------
def calcular_momento_inercia(forma, masa, radio, longitud=None):
    """Calcula el momento de inercia Icm para diferentes geometrías respecto a un eje que pasa por el centro de masa."""
    # Coeficiente c, donde I = c * M * R^2 (o L^2 para varilla)
    coeficientes = {
        "Disco/Cilindro Sólido": 0.5,
        "Cilindro Hueco (Anillo)": 1.0,
        "Esfera Sólida": 0.4, # 2/5
        "Varilla Delgada (L)": 1/12 
    }
    c = coeficientes.get(forma, 0)

    if forma == "Varilla Delgada (L)":
        return c * masa * radio**2 # radio se usa como longitud L
    return c * masa * radio**2

# ----------------- Funciones de Cálculo para las Simulaciones -----------------

def simular_torque(I, tau, t_max, dt=0.05):
    """Simulación 1: Dinámica y Cinemática Rotacional."""
    if I == 0:
        return pd.DataFrame(), 0.0

    alfa = tau / I  # Aceleración angular constante
    
    tiempo = np.arange(0, t_max + dt, dt) 
    
    omega = alfa * tiempo               
    theta = 0.5 * alfa * tiempo**2      
    vueltas = theta / (2 * np.pi)       
    
    df = pd.DataFrame({
        'Tiempo (s)': tiempo,
        'Velocidad Angular (rad/s)': omega,
        'Ángulo Girado (rad)': theta,
        'Número de Vueltas': vueltas,
        'Aceleración Angular (rad/s^2)': [alfa] * len(tiempo)
    })
    return df, alfa

def simular_masa_colgante(m_masa, R_cil, M_cil, t_max, dt=0.05):
    """Simulación 2: Energía y Dinámica."""
    g = 9.81  # Aceleración de la gravedad
    if m_masa <= 0 or R_cil <= 0 or M_cil <= 0:
        return pd.DataFrame(), 0, 0, 0

    I_cil = 0.5 * M_cil * R_cil**2  # Cilindro Sólido
    
    a = g / (1 + I_cil / (m_masa * R_cil**2))
    alfa = a / R_cil 
    T = m_masa * (g - a) 
    
    tiempo = np.arange(0, t_max + dt, dt)
    h = 0.5 * a * tiempo**2  
    
    K_rot = 0.5 * I_cil * (alfa * tiempo)**2  
    K_tras = 0.5 * m_masa * (a * tiempo)**2  
    
    # Se calcula la Energía Potencial respecto a la altura final (h_max_caida=0)
    h_max_caida = 0.5 * a * t_max**2
    U_grav_actual = m_masa * g * (h_max_caida - h)
    
    df = pd.DataFrame({
        'Tiempo (s)': tiempo,
        'Energía Rotacional (J)': K_rot,
        'Energía Traslacional (J)': K_tras,
        'Energía Potencial (J)': U_grav_actual,
        'Energía Total (J)': K_rot + K_tras + U_grav_actual,
        'Caída (m)': h # Para la animación
    })
    return df, a, alfa, T

# ----------------- Funciones de Animación 3D/Visualización -----------------

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
st.write("Simulaciones basadas en los Capítulos **9** y **10** de *Física Universitaria* (Sears, Zemansky, Freedman).")

# Selector de simulación en la barra lateral
opcion = st.sidebar.selectbox(
    "Selecciona la Simulación:",
    (
        "📚 Conceptos Fundamentales",
        "1️⃣ Dinámica y Cinemática Rotacional",
        "2️⃣ Energía: Cilindro con Masa Colgante",
        "3️⃣ Conservación del Momento Angular (Patinador)",
        "4️⃣ Casos Extendidos y Rodadura"
    )
)

# ----------------- Contenido de las Secciones -----------------

if opcion == "📚 Conceptos Fundamentales":
    st.header("Conceptos Clave de Dinámica y Cinemática Rotacional")
    st.markdown("""
    El movimiento rotacional es análogo al lineal. Las variables y principios clave son:

    | Concepto | Lineal | Rotacional (Cap. 9 y 10) | Principio |
    | :--- | :---: | :---: | :---: |
    | Cantidad Base | Masa ($m$) | **Momento de Inercia** ($I$) | Inercia |
    | Impulso/Fuerza | $\\sum F = ma$ | **$\\sum \\tau = I \\alpha$** | Dinámica |
    | Energía Cinética | $K = 1/2 m v^2$ | **$K = 1/2 I \\omega^2$** | Conservación de Energía |
    | Momento/Impulso | $p = mv$ | **Momento Angular** ($L = I \\omega$) | Conservación de Momento Angular |
    """)
    st.info("¡Usa el menú lateral para seleccionar una simulación y experimentar virtualmente!")

# ---------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "1️⃣ Dinámica y Cinemática Rotacional":
    st.header("1. Dinámica y Cinemática Rotacional: Torque Aplicado 🔄")
    st.markdown("Calcula el **Momento de Inercia ($I$)** y simula la rotación bajo un **torque ($\\tau$)** constante. Observa el cambio en $\\omega$, $\\theta$ y el número de vueltas en **tiempo real**.")

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
        st.error("Error: Momento de Inercia no válido.")
    else:
        df_sim, alfa = simular_torque(I_cm, torque, t_max)
        
        st.markdown("---")
        
        col_anim, col_datos = st.columns([1, 1])
        
        with col_datos:
            st.subheader("Métricas de Rotación")
            st.latex(f"I = {I_cm:.4f} \\, \\text{{kg}} \\cdot \\text{{m}}^2")
            st.latex(f"\\alpha = {alfa:.4f} \\, \\text{{rad/s}}^2")

            metric_placeholder = st.empty()
            
        with col_anim:
            st.subheader("Visualización Animada 3D")
            animation_placeholder = st.empty()
            
        if st.button("▶️ Iniciar Rotación Interactiva", key="anim3d_1"):
            height = radio * 0.1 
            x_base, y_base, z_base = create_cylinder_mesh(radio, height)
            
            animation_steps = 100
            time_steps = np.linspace(0, t_max, animation_steps)
            
            # Animación y métricas en tiempo real
            for i in range(animation_steps):
                current_time = time_steps[i]
                
                # Obtener valores cinemáticos interpolados
                omega_t = alfa * current_time
                theta_t = 0.5 * alfa * current_time**2
                vueltas_t = theta_t / (2 * np.pi)
                
                # --- Actualizar Métricas ---
                metric_placeholder.markdown(f"""
                <div style="padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
                    <p style="font-size: 16px;">
                        <strong>Tiempo:</strong> {current_time:.2f} s<br>
                        <strong>Velocidad Angular ($\omega$):</strong> {omega_t:.2f} rad/s<br>
                        <strong>Ángulo Girado ($\\theta$):</strong> {theta_t:.2f} rad<br>
                        <strong>Número de Vueltas:</strong> {vueltas_t:.2f} 
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # --- Actualizar Animación 3D ---
                if "Cilindro" in forma or "Disco" in forma:
                    x_rot, y_rot, z_rot = get_rotated_cylinder_data(x_base, y_base, z_base, theta_t)
                    
                    fig_3d = go.Figure(data=[
                        go.Surface(x=x_rot, y=y_rot, z=z_rot, colorscale='Plasma', opacity=0.8, showscale=False)
                    ])
                    
                    fig_3d.update_layout(
                        title=f"Rotación ({np.degrees(theta_t) % 360:.1f}°)",
                        scene_aspectmode='cube',
                        scene=dict(
                            xaxis=dict(range=[-radio*1.2, radio*1.2], visible=False),
                            yaxis=dict(range=[-radio*1.2, radio*1.2], visible=False),
                            zaxis=dict(range=[-height, height], visible=False),
                            aspectratio=dict(x=1, y=1, z=0.1),
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                        ),
                        margin=dict(l=0, r=0, b=0, t=40),
                        height=350
                    )
                    animation_placeholder.plotly_chart(fig_3d, use_container_width=True)
                else:
                    animation_placeholder.warning("Animación 3D solo disponible para cilindros/discos.")
                
                time.sleep(t_max / animation_steps / 2) 

        # --- Gráficas de Cinemática (Cap. 9) ---
        st.subheader("Gráficas de Cinemática Rotacional")
        
        fig_omega = px.line(
            df_sim, 
            x='Tiempo (s)', 
            y=['Velocidad Angular (rad/s)', 'Ángulo Girado (rad)'],
            title='Velocidad y Ángulo vs. Tiempo'
        )
        st.plotly_chart(fig_omega, use_container_width=True)


# ---------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "2️⃣ Energía: Cilindro con Masa Colgante":
    st.header("2. Conversión de Energía: Masa Colgante ⚡")
    st.markdown("Se analiza la transformación de **Energía Potencial ($U$)** en **Energía Cinética de Traslación ($K_{{tras}}$) y Rotacional ($K_{{rot}}$)**, demostrando la conservación de energía.")
    
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
        
        # --- Animación (Mejorada para visualizar el movimiento) ---
        st.subheader("Animación Interactiva del Sistema (Acción y Reacción)")
        col_anim_2, col_datos_2 = st.columns([1, 1])

        with col_datos_2:
             st.info(f"El cilindro recibe un **torque $\\tau = T\\cdot R_{{cil}}$** que lo hace girar, mientras que la masa cae con **aceleración $a$**.")
             st.markdown(f"**Tensión del Cable:** ${T:.2f}\\,\\text{{N}}$")
             st.markdown(f"**Aceleración de la Masa:** ${a:.2f}\\,\\text{{m/s}}^2$")
             metric_masa = st.empty()
        
        with col_anim_2:
            animation_placeholder = st.empty()
        
        if st.button("▶️ Iniciar Simulación de Caída", key="anim3d_2"):
            animation_steps = 50
            
            for i in range(animation_steps):
                # Aseguramos no salir del rango del DataFrame
                index = min(i * (len(df_ener) // animation_steps), len(df_ener) - 1)
                
                current_time = df_ener['Tiempo (s)'].iloc[index]
                h_caida = df_ener['Caída (m)'].iloc[index]
                
                # Cinemática para la rotación del cilindro
                theta_t = alfa * current_time * current_time * 0.5
                
                # --- Actualizar Métricas ---
                metric_masa.metric(label="Caída de la Masa (h)", value=f"{h_caida:.2f} m")

                # --- Actualizar Animación 3D ---
                height = R_cil * 0.2 
                x_base, y_base, z_base = create_cylinder_mesh(R_cil, height)
                x_rot, y_rot, z_rot = get_rotated_cylinder_data(x_base, y_base, z_base, theta_t)
                
                x_masa = R_cil 
                y_masa = -R_cil - h_caida # Posición de la masa colgante

                fig_3d = go.Figure(data=[
                    # Cilindro Rotando
                    go.Surface(x=x_rot, y=y_rot, z=z_rot, colorscale='Blues', opacity=0.8, showscale=False),
                    # Masa Colgante (Punto)
                    go.Scatter3d(x=[x_masa], y=[y_masa], z=[0], mode='markers', marker=dict(size=8, color='red'), name='Masa'),
                    # Cuerda (Línea)
                    go.Scatter3d(x=[R_cil, R_cil], y=[-R_cil, y_masa], z=[0, 0], mode='lines', line=dict(color='black', width=2), name='Cuerda')
                ])
                
                fig_3d.update_layout(
                    title="Cilindro Girando y Masa Cayendo",
                    scene_aspectmode='cube',
                    scene=dict(
                        xaxis=dict(range=[-R_cil*1.5, R_cil*1.5], visible=False),
                        yaxis=dict(range=[-R_cil*2 - 0.5, R_cil], visible=False),
                        zaxis=dict(range=[-height, height], visible=False),
                        aspectratio=dict(x=1, y=1, z=0.5),
                        camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
                    ),
                    margin=dict(l=0, r=0, b=0, t=40),
                    height=350
                )
                
                animation_placeholder.plotly_chart(fig_3d, use_container_width=True)
                time.sleep(t_max / animation_steps / 2)
        
        # --- Gráfica de Conservación de Energía ---
        st.subheader("Gráfica de Conservación de Energía")
        fig_ener = px.line(
            df_ener, 
            x='Tiempo (s)', 
            y=['Energía Rotacional (J)', 'Energía Traslacional (J)', 'Energía Potencial (J)', 'Energía Total (J)'], 
            title='Flujo de Energía en el Sistema',
            labels={'value': 'Energía (J)', 'variable': 'Tipo de Energía'},
            color_discrete_map={
                'Energía Rotacional (J)': 'blue',
                'Energía Traslacional (J)': 'orange',
                'Energía Potencial (J)': 'green',
                'Energía Total (J)': 'red'
            }
        )
        st.plotly_chart(fig_ener, use_container_width=True)


# ---------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "3️⃣ Conservación del Momento Angular (Patinador)":
    st.header("3. Conservación del Momento Angular (Patinador) ⛸️")
    st.markdown("La animación muestra cómo la **disminución del Momento de Inercia ($I$)** al acercar la masa al eje, provoca un **aumento de la Velocidad Angular ($\\omega$)** para mantener el Momento Angular ($L$) constante.")
    
    # Controles de entrada
    col1, col2 = st.columns(2)
    with col1:
        M_cuerpo = st.number_input("Masa del Patinador (kg):", 40.0, 100.0, 60.0, 1.0)
        I_cuerpo = st.number_input("Inercia del Cuerpo Central ($I_{cm}$, kg·m²):", 0.5, 5.0, 1.0, 0.1, help="Inercia del torso/cabeza (constante).")
        R_ext_i = st.slider("Radio Inicial de Brazos ($R_i$, m):", 0.5, 2.0, 1.5, 0.1)
    with col2:
        w_i = st.number_input("Velocidad Angular Inicial ($\\omega_i$, rad/s):", 0.1, 5.0, 1.0, 0.1)
        R_ext_f = st.slider("Radio Final de Brazos ($R_f$, m):", 0.1, 1.0, 0.5, 0.1)
        st.info("Asegúrate de que $R_f < R_i$ para ver el aumento de velocidad.")
        
    st.markdown("---")

    # --- Cálculos ---
    m_brazos = M_cuerpo * 0.2 
    
    I_i = I_cuerpo + m_brazos * R_ext_i**2
    L_i = I_i * w_i
    I_f = I_cuerpo + m_brazos * R_ext_f**2
    w_f = L_i / I_f

    # --- Animación del Patinador ---
    col_vis, col_calc = st.columns([1, 1])
    
    with col_vis:
        st.subheader("Animación: Conservación del Momento Angular")
        animation_placeholder_patinador = st.empty()
        
        if st.button("▶️ Iniciar Contracción de Brazos", key="anim_patinador"):
            
            steps = 50
            transition_time = 2.0 # Segundos para la transición
            
            # Interpolación lineal de R e I
            R_steps = np.linspace(R_ext_i, R_ext_f, steps)
            I_steps = I_cuerpo + m_brazos * R_steps**2
            
            # Cálculo de la nueva omega (L constante)
            w_steps = L_i / I_steps 
            
            # Simulación de la rotación
            theta_t = 0
            
            for i in range(steps):
                R_actual = R_steps[i]
                w_actual = w_steps[i]
                
                # Integrar el ángulo
                theta_t += w_actual * (transition_time / steps)
                
                # Coordenadas de la masa (brazos)
                x_mass = R_actual * np.cos(theta_t)
                y_mass = R_actual * np.sin(theta_t)
                
                fig = go.Figure(data=[
                    # Eje de rotación (Centro)
                    go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=15, color='darkgray'), name='Torso'),
                    # Masa Extrema (Brazos)
                    go.Scatter(x=[x_mass, -x_mass], y=[y_mass, -y_mass], mode='markers', marker=dict(size=10, color='red'), name='Brazos'),
                    # Brazos (Líneas)
                    go.Scatter(x=[0, x_mass, 0, -x_mass], y=[0, y_mass, 0, -y_mass], mode='lines', line=dict(color='lightgray', width=2), showlegend=False)
                ])
                
                fig.update_layout(
                    title=f"R={R_actual:.2f}m → $\\omega$={w_actual:.2f} rad/s",
                    xaxis=dict(range=[-R_ext_i*1.1, R_ext_i*1.1], title="Eje X (m)"),
                    yaxis=dict(range=[-R_ext_i*1.1, R_ext_i*1.1], title="Eje Y (m)", scaleanchor="x", scaleratio=1),
                    height=400,
                    showlegend=False
                )
                
                animation_placeholder_patinador.plotly_chart(fig, use_container_width=True)
                time.sleep(transition_time / steps)
                
            st.success(f"Transición completa. $\\omega$ final: {w_f:.2f} rad/s")

    with col_calc:
        st.subheader("Análisis Físico Detallado")
        st.markdown(f"""
        | **Variable** | **Inicial ($R_i={R_ext_i:.1f}$m)** | **Final ($R_f={R_ext_f:.1f}$m)** |
        | :---: | :---: | :---: |
        | **Inercia ($I$)** | ${I_i:.2f}\\,\\text{{kg}}\\cdot\\text{{m}}^2$ | $\mathbf{{{I_f:.2f}}}\\,\\mathbf{{\\text{{kg}}\\cdot\\text{{m}}^2}}$ |
        | **Velocidad ($\omega$)** | ${w_i:.2f}\\,\\text{{rad/s}}$ | $\\mathbf{{{w_f:.2f}}}\\;\\mathbf{{\\text{{rad/s}}}}$ |
        | **Momento Angular ($L$)** | ${L_i:.2f}\\,\\text{{kg}}\\cdot\\text{{m}}^2/\\text{{s}}$ | ${L_i:.2f}\\,\\text{{kg}}\\cdot\\text{{m}}^2/\\text{{s}}$ |
        """)
        
        K_i = 0.5 * I_i * w_i**2
        K_f = 0.5 * I_f * w_f**2
        delta_K = K_f - K_i
        st.info(f"Se realiza trabajo interno (músculos) para contraer los brazos, incrementando la Energía Cinética: $\\Delta K = +{delta_K:.4f}\\,\\text{{J}}$")
        
        fig_L = go.Figure(data=[
            go.Bar(name='Inercia (I)', x=['Inicial', 'Final'], y=[I_i, I_f], yaxis='y1', offsetgroup=1, marker_color='skyblue'),
            go.Bar(name='Velocidad (ω)', x=['Inicial', 'Final'], y=[w_i, w_f], yaxis='y2', offsetgroup=2, marker_color='lightcoral')
        ])
        fig_L.update_layout(
            title='Relación Inversa I vs. ω',
            yaxis=dict(title='Inercia (kg·m²)', side='left', showgrid=False),
            yaxis2=dict(title='Velocidad Angular (rad/s)', overlaying='y', side='right', showgrid=False)
        )
        st.plotly_chart(fig_L, use_container_width=True)


# ---------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "4️⃣ Casos Extendidos y Rodadura":
    st.header("4. Casos Extendidos: Rodadura, Energía y Sistemas Acoplados 🧩")
    
    tab1, tab2, tab3 = st.tabs(["Rodadura en Plano Inclinado", "Eje con Discos Acoplados", "Análisis de Energía Compuesta"])
    
    # --- Pestaña 1: Rodadura en Plano Inclinado ---
    with tab1:
        st.subheader("Carrera de Inercia: ¿Quién Rueda Más Rápido? 🥇")
        st.markdown("La animación simula una carrera entre tres objetos. El que tenga la menor **inercia relativa** ($c = I_{{cm}}/MR^2$) convierte más rápido $U$ en $K_{{tras}}$ y, por lo tanto, gana.")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            angulo_rod = st.slider("Ángulo del Plano ($\\theta$, grados):", 5, 90, 30, 1, key="angulo_rod")
            altura_rod = st.number_input("Altura Vertical ($h$, m):", 0.1, 5.0, 1.0, 0.1, key="altura_rod")
        
        g = 9.81
        angulo_rad = np.radians(angulo_rod)
        S_total = altura_rod / np.sin(angulo_rad)
        
        # Geometrías a comparar: c = I_cm / M R^2
        formas_c = {
            "Esfera Sólida (c=0.4)": 0.4,
            "Disco Sólido (c=0.5)": 0.5,
            "Cilindro Hueco (c=1.0)": 1.0,
        }
        
        datos_carrera = []
        carrera_params = {}
        for forma, c in formas_c.items():
            a_cm = (g * np.sin(angulo_rad)) / (1 + c)
            t_fin = np.sqrt(2 * S_total / a_cm) if a_cm > 0 else 999.0
            
            datos_carrera.append({
                'Forma': forma,
                'Factor c': c,
                'Aceleración ($a_{cm}$, m/s²)': a_cm,
                'Tiempo de Bajada (s)': t_fin
            })
            carrera_params[forma] = {'a': a_cm, 't_final': t_fin}
            
        df_carrera = pd.DataFrame(datos_carrera).sort_values(by='Tiempo de Bajada (s)', ascending=True).reset_index(drop=True)
        
        st.dataframe(df_carrera, use_container_width=True, hide_index=True)
        
        # --- Animación de la Carrera ---
        st.subheader("Simulación Visual de la Carrera")
        animation_placeholder_carrera = st.empty()
        
        if st.button("🏁 Iniciar Carrera", key="anim_carrera"):
            T_max_sim = df_carrera['Tiempo de Bajada (s)'].max() # Tiempo que tarda el más lento
            steps = 100
            time_steps = np.linspace(0, T_max_sim, steps)
            
            x_start = 0.0
            y_start = 0.0
            
            for t in time_steps:
                posiciones = {'Forma': [], 'Posición (X)': [], 'Posición (Y)': []}
                
                # Calcular la posición de cada cuerpo
                for forma, params in carrera_params.items():
                    a = params['a']
                    # Posición a lo largo del plano inclinado (S)
                    S_t = 0.5 * a * t**2 if t <= params['t_final'] else S_total 
                    
                    # Convertir S a coordenadas X e Y
                    X_t = x_start + S_t * np.cos(angulo_rad)
                    Y_t = y_start - S_t * np.sin(angulo_rad)
                    
                    posiciones['Forma'].append(forma)
                    posiciones['Posición (X)'].append(X_t)
                    posiciones['Posición (Y)'].append(Y_t)

                df_pos = pd.DataFrame(posiciones)
                
                # Crear la gráfica del plano inclinado y la posición de los cuerpos
                X_plano = [x_start, x_start + S_total * np.cos(angulo_rad)]
                Y_plano = [y_start, y_start - S_total * np.sin(angulo_rad)]
                
                fig_carrera = go.Figure()
                fig_carrera.add_trace(go.Scatter(x=X_plano, y=Y_plano, mode='lines', name='Plano Inclinado', line=dict(color='gray', width=3)))
                
                for forma in df_pos['Forma'].unique():
                    df_body = df_pos[df_pos['Forma'] == forma]
                    symbol = 'circle' if 'Esfera' in forma else 'square' if 'Disco' in forma else 'star'
                    # FIX: La línea incompleta estaba aquí. Se completa la expresión ternaria:
                    color = 'green' if 'Esfera' in forma else 'orange' if 'Disco' in forma else 'blue'
                    
                    fig_carrera.add_trace(go.Scatter(
                        x=df_body['Posición (X)'], 
                        y=df_body['Posición (Y)'], 
                        mode='markers', 
                        name=forma,
                        marker=dict(size=15, symbol=symbol, color=color)
                    ))
                
                fig_carrera.update_layout(
                    title=f"Tiempo: {t:.2f} s. Ganador Parcial: {df_carrera.iloc[0]['Forma']}",
                    xaxis=dict(range=[0, X_plano[1] * 1.1], title="Distancia Horizontal (m)", visible=False),
                    yaxis=dict(range=[Y_plano[1] * 1.1, 0], title="Altura (m)", scaleanchor="x", scaleratio=1),
                    height=450
                )
                
                animation_placeholder_carrera.plotly_chart(fig_carrera, use_container_width=True)
                time.sleep(T_max_sim / steps / 2)
            
            st.success(f"¡Carrera finalizada! El ganador es: **{df_carrera.iloc[0]['Forma']}**")

    # --- Pestaña 2: Eje con Discos Acoplados ---
    with tab2:
        st.subheader("Colisión Angular Inelástica (Acoplamiento de Discos) 🔗")
        st.markdown("El **Momento Angular se conserva** ($L_i=L_f$), pero la energía se pierde ($K_f < K_i$) debido a la fricción durante el acoplamiento.")
        
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
            
            # --- Cálculos de Acoplamiento ---
            I1 = 0.5 * M1 * R1**2
            I2 = 0.5 * M2 * R2**2
            L_i = I1 * w1_i
            I_f = I1 + I2
            w_f = L_i / I_f
            K_i = 0.5 * I1 * w1_i**2
            K_f = 0.5 * I_f * w_f**2
            K_perdida = K_i - K_f

            st.metric(label="Velocidad Angular Final ($\omega_f$)", value=f"{w_f:.3f} rad/s")
            st.metric(label="Pérdida de Energía Cinética ($\Delta K$)", value=f"{-K_perdida:.3f} J", delta=f"{-K_perdida:.3f} J (Perdida)")

    # --- Pestaña 3: Cálculo de Energías en Rotación ---
    with tab3:
        st.subheader("Cálculo de Energía Cinética Total ($K_{{Total}}$)")
        st.markdown("Separa las contribuciones de **traslación** ($K_{{tras}}$) y **rotación** ($K_{{rot}}$) para un cuerpo en movimiento (Cap. 9).")
        
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            forma_e = st.selectbox("Forma del Objeto:", ("Disco/Cilindro Sólido", "Esfera Sólida"), key="forma_e")
            M_e = st.number_input("Masa ($M$, kg):", 0.1, 10.0, 5.0, 0.1, key="M_e")
            R_e = st.number_input("Radio ($R$, m):", 0.1, 1.0, 0.2, 0.1, key="R_e")
        with col_e2:
            v_cm_e = st.number_input("Velocidad de Traslación ($v_{cm}$, m/s):", 0.1, 10.0, 2.0, 0.1, key="v_cm_e")
            # Rodadura Pura
            rodadura = st.checkbox("Rodadura Pura ($v_{cm} = R\\omega$)", value=True)
            if rodadura:
                w_e = v_cm_e / R_e
                st.markdown(f"Vel. Angular ($\\omega$): **{w_e:.2f} rad/s**")
            else:
                w_e = st.number_input("Velocidad Angular ($\\omega$, rad/s):", 0.1, 20.0, 5.0, 0.1, key="w_e")
                
        I_e = calcular_momento_inercia(forma_e, M_e, R_e)
        K_tras = 0.5 * M_e * v_cm_e**2
        K_rot = 0.5 * I_e * w_e**2
        K_total = K_tras + K_rot
        
        st.markdown("---")
        st.latex(f"I = {I_e:.4f} \\, \\text{{kg}} \\cdot \\text{{m}}^2")
        
        fig_K = px.pie(
            names=['Traslación', 'Rotación'], 
            values=[K_tras, K_rot], 
            title=f'Distribución de Energía Total ({K_total:.2f} J)'
        )
        st.plotly_chart(fig_K, use_container_width=True)
