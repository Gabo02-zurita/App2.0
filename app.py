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
                    # Cilind
