import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time 

# ----------------- Funciones de Cálculo para las Simulaciones -----------------

def calcular_momento_inercia(forma, masa, radio, longitud=None):
    """Calcula el momento de inercia I para diferentes geometrías."""
    if forma == "Disco/Cilindro Sólido":
        return 0.5 * masa * radio**2
    elif forma == "Cilindro Hueco (Anillo)":
        return masa * radio**2
    elif forma == "Esfera Sólida":
        return (2/5) * masa * radio**2
    return 0

def simular_torque(I, tau, t_max, dt=0.05):
    """Calcula variables cinemáticas para un torque constante."""
    if I == 0:
        return pd.DataFrame(), 0.0

    alfa = tau / I  # Aceleración angular constante
    
    tiempo = np.arange(0, t_max + dt, dt) 
    omega = alfa * tiempo  # Velocidad angular: omega = alfa * t
    theta = 0.5 * alfa * tiempo**2  # Ángulo girado: theta = 0.5 * alfa * t^2
    vueltas = theta / (2 * np.pi)  # Número de vueltas
    
    df = pd.DataFrame({
        'Tiempo (s)': tiempo,
        'Velocidad Angular (rad/s)': omega,
        'Ángulo Girado (rad)': theta,
        'Número de Vueltas': vueltas,
        'Aceleración Angular (rad/s^2)': [alfa] * len(tiempo)
    })
    return df, alfa

def simular_masa_colgante(m_masa, R_cil, M_cil, t_max, dt=0.05):
    """Simula masa colgante que desenrolla un cable de un cilindro."""
    g = 9.81  # Aceleración de la gravedad
    
    if m_masa <= 0 or R_cil <= 0:
        return pd.DataFrame(), 0, 0, 0

    I_cil = 0.5 * M_cil * R_cil**2  # Momento de inercia del cilindro
    
    # Cálculo de la aceleración lineal de la masa 'a'
    # a = g / (1 + I / (m*R^2))
    a = g / (1 + I_cil / (m_masa * R_cil**2))
    
    # Tensión del cable (T = m*(g-a))
    T = m_masa * (g - a)
    
    # Aceleración angular del cilindro: alfa = a / R
    alfa = a / R_cil 
    
    tiempo = np.arange(0, t_max + dt, dt)
    h = 0.5 * a * tiempo**2  # Distancia que cae la masa
    
    # Energías en función del tiempo
    K_rot = 0.5 * I_cil * (alfa * tiempo)**2  # Energía Cinética Rotacional
    K_tras = 0.5 * m_masa * (a * tiempo)**2  # Energía Cinética Traslacional
    
    # La energía potencial disminuye, tomando la referencia U=0 en el punto más bajo alcanzado
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

# ----------------- Funciones de Visualización 3D para Simulación 1 -----------------

def create_cylinder_mesh(radius, height, num_segments=50):
    """Crea una malla para un cilindro 3D (para Disco/Cilindro Sólido)."""
    z = np.linspace(-height/2, height/2, 2)
    theta = np.linspace(0, 2*np.pi, num_segments)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    return x_grid, y_grid, z_grid

def get_rotated_cylinder_data(x_base, y_base, z_base, angle):
    """Rota las coordenadas de un cilindro alrededor del eje Z."""
    # Matriz de rotación alrededor del eje Z
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

st.set_page_config(layout="wide", page_title="Rotación de Sólidos Rígidos")

st.title("Asistente Interactivo de Rotación de Sólidos Rígidos 🌀")
st.write("Esta aplicación te ayudará a modelar, visualizar y entender fenómenos de la dinámica de rotación.")

# Selector de simulación en la barra lateral
opcion = st.sidebar.selectbox(
    "Selecciona la Simulación:",
    (
        "📚 Introducción y Fundamentos",
        "1️⃣ Torque y Cinemática Rotacional",
        "2️⃣ Masa Colgante y Cilindro Fijo",
        "3️⃣ Conservación del Momento Angular",
        "4️⃣ Rodadura en Plano Inclinado (Extendido)"
    )
)

# ----------------- Contenido de las Secciones -----------------

if opcion == "📚 Introducción y Fundamentos":
    st.header("Conceptos Clave de la Dinámica de Rotación")
    st.markdown("""
    La **rotación de un sólido rígido** es el movimiento de un objeto alrededor de un eje. Los conceptos clave son:

    * **Momento de Inercia ($I$):** Resistencia al cambio en el movimiento rotacional (análogo a la masa).
    * **Torque ($\\tau$):** Fuerza que provoca el cambio en el movimiento rotacional.
    * **Segunda Ley de Newton para Rotación:** $\\tau = I \\alpha$ (fuerza = inercia × aceleración).
    * **Momento Angular ($L$):** Cantidad de rotación. Se conserva si el torque externo neto es cero.
    """)
    st.info("¡Usa el menú lateral para seleccionar una simulación y experimentar virtualmente!")

# ------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "1️⃣ Torque y Cinemática Rotacional":
    st.header("1. Simulación de Torque Constante y Cinemática 📈 (con Animación 3D)")
    st.markdown("Aplica un torque constante a una forma geométrica para observar cómo varían sus parámetros de rotación con el tiempo.")

    # Controles de entrada DIGITALIZADOS
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forma = st.selectbox(
            "Selecciona la Forma:",
            ("Disco/Cilindro Sólido", "Cilindro Hueco (Anillo)", "Esfera Sólida")
        )
    with col2:
        masa = st.number_input("Masa ($M$, kg):", 0.1, 10.0, 2.0, 0.1)
        radio = st.number_input("Radio ($R$, m):", 0.1, 1.0, 0.5, 0.05)
    with col3:
        torque = st.number_input("Torque Aplicado ($\\tau$, N·m):", 0.1, 5.0, 1.0, 0.1)
        t_max = st.number_input("Tiempo de Simulación ($t_{max}$, s):", 1.0, 20.0, 5.0, 0.5)
        
    # --- Cálculos y Resultados ---
    I = calcular_momento_inercia(forma, masa, radio) 
    
    if I <= 0:
        st.error("Error: Momento de Inercia no válido. Asegúrate de que la masa y el radio sean positivos.")
    else:
        df_sim, alfa = simular_torque(I, torque, t_max)
        
        st.markdown("---")
        st.subheader("Resultados Teóricos Clave")
        st.latex(f"I = {I:.4f} \\, \\text{{kg}} \\cdot \\text{{m}}^2")
        st.latex(f"\\tau = I \\alpha \\Rightarrow \\alpha = \\frac{{\\tau}}{{I}} = \\frac{{{torque:.2f}}}{{{I:.4f}}} = {alfa:.4f} \\, \\text{{rad/s}}^2")
        
        st.info(f"El objeto es un **{forma}** con un **Momento de Inercia ($I$)** de **{I:.4f} kg·m²**.")

        # --- Visualización 3D Interactiva del Objeto ---
        st.subheader("Visualización 3D Animada del Objeto Girando")
        
        # Crear la base del cilindro/disco
        height = radio * 0.1 
        x_base, y_base, z_base = create_cylinder_mesh(radio, height)
        
        # Contenedor para la animación 3D
        animation_placeholder = st.empty()
        
        if st.button("▶️ Iniciar Animación 3D", key="anim3d"):
            
            animation_steps = 100
            time_steps = np.linspace(0, t_max, animation_steps)
            
            for i in range(animation_steps):
                current_time = time_steps[i]
                # Ángulo girado hasta el momento actual
                current_theta = 0.5 * alfa * current_time**2
                
                x_rot, y_rot, z_rot = get_rotated_cylinder_data(x_base, y_base, z_base, current_theta)
                
                fig_3d = go.Figure(data=[
                    go.Surface(
                        x=x_rot, y=y_rot, z=z_rot, 
                        colorscale='Viridis', 
                        opacity=0.8,
                        showscale=False
                    )
                ])
                
                fig_3d.update_layout(
                    title=f"Tiempo: {current_time:.2f} s | Ángulo: {np.degrees(current_theta) % 360:.1f}°",
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
                time.sleep(t_max / animation_steps) 
        
        # --- Visualización de Gráficas (Plotly Interactivo) ---
        st.subheader("Gráficas Interactivas de Cinemática Rotacional")
        
        fig_omega = px.line(
            df_sim, 
            x='Tiempo (s)', 
            y='Velocidad Angular (rad/s)', 
            title=f'Velocidad Angular ($\omega$) vs. Tiempo ($\\alpha = {alfa:.4f}$ rad/s²)',
            labels={'Velocidad Angular (rad/s)': 'Velocidad Angular $\omega$ (rad/s)'}
        )
        fig_omega.update_layout(hovermode="x unified")
        st.plotly_chart(fig_omega, use_container_width=True)

        st.subheader("Explicación Física")
        st.markdown(f"""
        * **Aceleración Angular ($\\alpha$):** Es **constante** e igual a **{alfa:.4f} rad/s²**.
        * **Velocidad Angular ($\omega$):** Aumenta **linealmente** con el tiempo, ya que la aceleración es constante ($\\omega = \\alpha t$).
        * **Ángulo Girado ($\\theta$):** Aumenta **cuadráticamente** con el tiempo ($\\theta = \\frac{1}{2} \\alpha t^2$).
        """)

# ------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "2️⃣ Masa Colgante y Cilindro Fijo":
    st.header("2. Cilindro Fijo con Masa Colgante ⛓️")
    st.markdown("Analiza la conversión de la energía potencial en energía cinética de traslación y rotación en este sistema.")
    
    # Controles de entrada DIGITALIZADOS
    col1, col2 = st.columns(2)
    with col1:
        M_cil = st.number_input("Masa del Cilindro ($M_{cil}$, kg):", 0.1, 10.0, 2.0, 0.1)
        R_cil = st.number_input("Radio del Cilindro ($R_{cil}$, m):", 0.05, 1.0, 0.2, 0.05)
    with col2:
        m_masa = st.number_input("Masa Colgante ($m_{masa}$, kg):", 0.01, 5.0, 1.0, 0.01)
        t_max = st.number_input("Tiempo de Simulación ($t_{max}$, s):", 0.5, 10.0, 3.0, 0.5)

    if R_cil <= 0 or m_masa <= 0 or M_cil <= 0:
        st.error("Todas las masas y el radio deben ser positivos para la simulación.")
    else:
        # --- Cálculos y Resultados ---
        df_ener, a, alfa, T = simular_masa_colgante(m_masa, R_cil, M_cil, t_max)

        st.markdown("---")
        st.subheader("Resultados Teóricos Clave")
        st.latex(f"a = {a:.4f} \\, \\text{{m/s}}^2 \\quad | \\quad \\alpha = {alfa:.4f} \\, \\text{{rad/s}}^2 \\quad | \\quad T = {T:.4f} \\, \\text{{N}}")

        # --- Visualización de Energía ---
        st.subheader("Distribución de Energía vs. Tiempo")
        
        fig_ener = px.line(
            df_ener, 
            x='Tiempo (s)', 
            y=['Energía Rotacional (J)', 'Energía Traslacional (J)', 'Energía Potencial (J)', 'Energía Total (J)'], 
            title='Conversión de Energía Potencial a Cinética',
            labels={'value': 'Energía (J)', 'variable': 'Tipo de Energía'}
        )
        fig_ener.update_layout(hovermode="x unified")
        st.plotly_chart(fig_ener, use_container_width=True)
        
        st.subheader("Explicación Física y Energía")
        st.markdown("""
        * **Dinámica:** La tensión del cable ($T$) produce el torque en el cilindro, y la gravedad y la tensión actúan sobre la masa.
        * **Conservación de la Energía:** En ausencia de fricción, la **Energía Total** del sistema se **conserva (línea horizontal)**. La Energía Potencial de la masa se convierte en Cinética de traslación (masa) y Rotación (cilindro).
        """)

# ------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "3️⃣ Conservación del Momento Angular":
    st.header("3. Conservación del Momento Angular: El Patinador ⛸️ (Modelo de Masas Variables)")
    st.markdown("Modelo que simula al patinador variando su Momento de Inercia ($I$) al mover dos masas (brazos) con distancia radial ($r$) variable.")
    
    st.markdown("---")
    st.subheader("Configuración del Sistema (Valores Digitales)")
    
    col1, col2 = st.columns(2)
    with col1:
        # I_0 simula la inercia del cuerpo central (cabeza y torso)
        I_cuerpo = st.number_input("Inercia Fija del Cuerpo Central ($I_0$, kg·m²):", 0.1, 5.0, 0.5, 0.1, help="Inercia fija del cuerpo (base del patinador).")
        # Masa de los brazos/masas puntuales (m1 = m2)
        m_brazo = st.number_input("Masa de Cada Brazo/Masa ($m$, kg):", 0.1, 5.0, 1.5, 0.1, help="Masa de cada una de las dos partes móviles.")
    with col2:
        # Radio Inicial (brazos extendidos)
        r_ini = st.number_input("Distancia Radial Inicial ($r_{ini}$, m):", 0.1, 2.0, 1.0, 0.1, help="Distancia inicial de las masas al eje de giro.")
        # Velocidad Angular Inicial
        omega_ini = st.number_input("Velocidad Angular Inicial ($\\omega_{ini}$, rad/s):", 0.1, 5.0, 1.0, 0.1, help="Velocidad de giro inicial del sistema.")

    st.markdown("---")
    st.subheader("Experimentación Virtual (Arrastra el Deslizador)")
    
    # Control SLIDER para cambiar la distancia radial (Experimentación)
    r_final = st.slider("Distancia Radial Final ($r_{final}$, m):", 0.1, 2.0, 0.2, 0.05, help="Arrastra para simular recoger (valor pequeño) o extender (valor grande) los brazos.")

    # --- Cálculos Basados en el Modelo de Patinador ---
    
    # Modelo de Inercia: I = I_0 + 2 * m * r^2
    I_inicial = I_cuerpo + 2 * m_brazo * r_ini**2
    I_final = I_cuerpo + 2 * m_brazo * r_final**2
    
    # Momento Angular (L se conserva)
    L = I_inicial * omega_ini
    
    # Velocidad Angular Final (omega = L / I)
    omega_final = L / I_final if I_final != 0 else 0
    
    # Energías Cinéticas de Rotación (K = 0.5 * I * omega^2)
    K_ini = 0.5 * I_inicial * omega_ini**2
    K_final = 0.5 * I_final * omega_final**2

    st.markdown("---")
    st.subheader("Análisis de la Conservación")
    
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.info("Valores Iniciales y Momento Angular Conservado ($L$):")
        st.latex(f"I_{{ini}} = {I_inicial:.3f} \\, \\text{{kg}} \\cdot \\text{{m}}^2")
        st.latex(f"\\omega_{{ini}} = {omega_ini:.3f} \\, \\text{{rad/s}}")
        st.latex(f"L = I_{{ini}} \\omega_{{ini}} = {L:.3f} \\, \\text{{kg}} \\cdot \\text{{m}}^2/\\text{{s}}")

    with col_res2:
        st.success("Resultados al Mover los Brazos:")
        st.latex(f"I_{{final}} = {I_final:.3f} \\, \\text{{kg}} \\cdot \\text{{m}}^2")
        st.latex(f"\\omega_{{final}} = \\frac{{L}}{{I_{{final}}}} = {omega_final:.3f} \\, \\text{{rad/s}}")
        st.latex(f"K_{{final}} = \\frac{{1}}{{2}} I_{{final}} \\omega_{{final}}^2 = {K_final:.3f} \\, \\text{{J}}")

    st.markdown("---")
    st.subheader("Conclusión Física")

    if r_final < r_ini:
        st.error(f"**¡El Patinador Acelera!** $\\omega$ aumentó de ${omega_ini:.2f}$ rad/s a **${omega_final:.2f}$ rad/s**.")
        st.warning(f"La Energía Cinética aumentó: $K_{{final}} > K_{{inicial}}$. El trabajo realizado para acercar los brazos (fuerzas internas) se convierte en esta energía cinética rotacional extra.")
    elif r_final > r_ini:
        st.error(f"**¡El Patinador Frena!** $\\omega$ disminuyó de ${omega_ini:.2f}$ rad/s a **${omega_final:.2f}$ rad/s**.")
        st.warning(f"La Energía Cinética disminuyó: $K_{{final}} < K_{{inicial}}$. El sistema liberó energía para extender los brazos, lo que reduce la velocidad de rotación.")
    else:
        st.info(f"**El sistema se mantiene en equilibrio:** $I$ y $\\omega$ no cambian.")

# ------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "4️⃣ Rodadura en Plano Inclinado (Extendido)":
    st.header("4. Rodadura de Varias Formas por un Plano Inclinado ⛰️")
    st.markdown("Compara la aceleración lineal de diferentes sólidos rígidos rodando sin deslizar. Solo importa la distribución de la masa.")
    
    # Controles de entrada DIGITALIZADOS
    col1, col2 = st.columns(2)
    with col1:
        angulo = st.number_input("Ángulo de Inclinación ($\\theta$, grados):", 5, 85, 30)
        L_plano = st.number_input("Longitud del Plano Inclinado ($L$, m):", 0.5, 20.0, 5.0)
    with col2:
        g = st.number_input("Aceleración de la Gravedad ($g$, m/s²):", 0.1, 20.0, 9.81)

    if angulo <= 0 or angulo >= 90 or L_plano <= 0:
        st.error("El ángulo debe estar entre 1 y 89 grados, y la longitud debe ser positiva.")
    else:
        # Convertir ángulo a radianes
        theta_rad = np.deg2rad(angulo)

        # Constantes de Momento de Inercia (C = I / (m*R^2))
        formas_C = {
            "Esfera Sólida (C=0.4)": 0.4,
            "Disco/Cilindro Sólido (C=0.5)": 0.5,
            "Esfera Hueca (C≈0.667)": 2/3,
            "Cilindro Hueco (Anillo, C=1.0)": 1.0
        }
        
        resultados = []
        
        for forma, C in formas_C.items():
            # Aceleración lineal para rodadura pura: a = g*sin(theta) / (1 + C)
            a = (g * np.sin(theta_rad)) / (1 + C)
            
            if a > 1e-6: 
                # Tiempo para recorrer la distancia L: t = sqrt(2L / a)
                t = np.sqrt((2 * L_plano) / a)
                v_final = a * t
            else: 
                t = np.inf
                v_final = 0
            
            resultados.append({
                'Forma': forma,
                'Aceleración (a, m/s²)': a,
                'Tiempo de Descenso (t, s)': t,
                'Velocidad Final (v, m/s)': v_final
            })
            
        df_rodadura = pd.DataFrame(resultados).sort_values(by='Tiempo de Descenso (t, s)')

        st.markdown("---")
        st.subheader("Resultados de la Carrera (Ordenado por Tiempo de Descenso)")
        st.dataframe(df_rodadura.style.format({
            'Tiempo de Descenso (t, s)': lambda x: f"{x:.2f}" if x != np.inf else "∞",
            'Aceleración (a, m/s²)': "{:.2f}",
            'Velocidad Final (v, m/s)': "{:.2f}"
            }), hide_index=True, use_container_width=True)

        st.subheader("Explicación Física: El Rol de la Geometría 🏆")
        st.markdown(f"""
        * **Aceleración:** La fórmula $a = \\frac{{g \\sin(\\theta)}}{{1 + C}}$ muestra que cuanto **menor** es la constante $C = \\frac{{I}}{{MR^2}}$ de la forma, **mayor** es la aceleración lineal.
        * **Ganador:** La **Esfera Sólida** (C=0.4) gana porque concentra más masa cerca del eje, requiriendo menos energía para rotar y dejando más energía disponible para la traslación.
        * **Dependencia:** El tiempo y la aceleración **no dependen de la masa ni del radio** del objeto, solo de su *forma* ($C$) y del ángulo ($\theta$).
        """)
