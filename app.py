import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time 

# ----------------- Fórmulas de Momento de Inercia (Capítulo 9) -----------------
# Basadas en I = c * M * R^2
def calcular_momento_inercia(forma, masa, radio, longitud=None):
    """Calcula el momento de inercia Icm para diferentes geometrías respecto a un eje que pasa por el centro de masa."""
    # Los valores c (Icm / MR^2) son estándar y corresponden a la Tabla 9.2
    if forma == "Disco/Cilindro Sólido (c=0.5)":
        return 0.5 * masa * radio**2  # I_cm = (1/2)MR^2
    elif forma == "Cilindro Hueco (Anillo, c=1.0)":
        return masa * radio**2        # I_cm = MR^2
    elif forma == "Esfera Sólida (c=0.4)":
        return (2/5) * masa * radio**2  # I_cm = (2/5)MR^2
    return 0

# ----------------- Funciones de Cálculo para las Simulaciones -----------------

def simular_torque(I, tau, t_max, dt=0.05):
    """Calcula variables cinemáticas para un torque constante (Segunda Ley de Newton para Rotación, Cap. 10)."""
    if I == 0:
        return pd.DataFrame(), 0.0

    # Segunda Ley de Newton para la Rotación: tau = I * alpha [cite: 11]
    alfa = tau / I  # Aceleración angular constante
    
    tiempo = np.arange(0, t_max + dt, dt) 
    
    # Ecuaciones de Cinemática Rotacional con aceleración constante (análogos del Cap. 9)
    omega = alfa * tiempo               # omega = omega_0 + alpha * t (con omega_0 = 0)
    theta = 0.5 * alfa * tiempo**2      # theta = theta_0 + omega_0*t + 0.5 * alpha * t^2 (con theta_0, omega_0 = 0)
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
    """Simula masa colgante que desenrolla un cable de un cilindro usando el enfoque de Energía (Cap. 9)."""
    g = 9.81  # Aceleración de la gravedad
    
    if m_masa <= 0 or R_cil <= 0:
        return pd.DataFrame(), 0, 0, 0

    I_cil = 0.5 * M_cil * R_cil**2  # Momento de inercia del cilindro
    
    # Cálculo de la aceleración lineal de la masa 'a' (Resultado de aplicar tau=I*alpha y F=ma)
    # a = g / (1 + I / (m*R^2))
    a = g / (1 + I_cil / (m_masa * R_cil**2))
    
    # Tensión del cable (T = m*(g-a))
    T = m_masa * (g - a)
    
    # Aceleración angular del cilindro: alfa = a / R
    alfa = a / R_cil 
    
    tiempo = np.arange(0, t_max + dt, dt)
    h = 0.5 * a * tiempo**2  # Distancia que cae la masa
    
    # Energía Cinética Rotacional (K_rot = 0.5 * I * omega^2) [cite: 2]
    K_rot = 0.5 * I_cil * (alfa * tiempo)**2  
    # Energía Cinética Traslacional (K_tras = 0.5 * m * v^2)
    K_tras = 0.5 * m_masa * (a * tiempo)**2  
    
    # Energía Potencial Gravitacional (U = Mgy_cm, Cap. 9) [cite: 3]
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

# ----------------- Funciones de Visualización 3D (No modificadas) -----------------

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

st.title("Asistente Interactivo: Rotación de Cuerpos Rígidos (Cap. 9 y 10) 🌀")
st.write("Esta aplicación modela fenómenos de la dinámica de rotación basados en las ecuaciones de *Física Universitaria* (Sears, Zemansky, Freedman).")

# Selector de simulación en la barra lateral
opcion = st.sidebar.selectbox(
    "Selecciona la Simulación:",
    (
        "📚 Conceptos Fundamentales (Cap. 9 y 10)",
        "1️⃣ Torque y Dinámica Rotacional ($\\tau = I\\alpha$)",
        "2️⃣ Trabajo y Energía en Rotación (K = 1/2 I $\\omega^2$)",
        "3️⃣ Conservación del Momento Angular ($L = I\\omega$)",
        "4️⃣ Rodadura en Plano Inclinado (Energía y Aceleración)"
    )
)

# ----------------- Contenido de las Secciones -----------------

if opcion == "📚 Conceptos Fundamentales (Cap. 9 y 10)":
    st.header("Conceptos Clave de Dinámica y Cinemática Rotacional")
    st.markdown("""
    La **rotación de un cuerpo rígido** se describe mediante análogos a las cantidades lineales[cite: 1].

    * **Momento de Inercia ($I$):** Es la resistencia del cuerpo a los cambios en su movimiento rotacional[cite: 2].
        $$\\text{Definición: } I = \\sum_i m_i r_i^2$$
        Se mide en $\\text{kg} \\cdot \\text{m}^2$[cite: 2].

    * **Torque ($\\tau$):** Análogo rotacional de la fuerza[cite: 11]. Causa cambios en la velocidad angular.
        $$\\text{Segunda Ley de Newton para Rotación: } \\sum \\tau = I \\alpha$$

    * **Energía Cinética Rotacional ($K$):** La energía asociada a la rotación.
        $$K = \\frac{1}{2} I \\omega^2$$ [cite: 2]

    * **Momento Angular ($L$):** El análogo rotacional del momento lineal ($p$)[cite: 9].
        $$\\text{Para un cuerpo rígido: } L = I \\omega$$ [cite: 9]
        La conservación ocurre si el torque externo neto es cero: $L_{\\text{ini}} = L_{\\text{final}}$[cite: 10].
    """)
    st.info("¡Usa el menú lateral para seleccionar una simulación y experimentar virtualmente!")

# ------------------------------------------------------------
---
# ------------------------------------------------------------

elif opcion == "1️⃣ Torque y Dinámica Rotacional ($\\tau = I\\alpha$)":
    st.header("1. Dinámica Rotacional: Aplicación de Torque Constante")
    st.markdown("Aplica un torque constante a una geometría. Analizamos la aceleración angular ($\\alpha$) resultante y cómo cambian la velocidad y el ángulo (Cinemática Rotacional, Cap. 9)[cite: 1, 11].")

    # Controles de entrada DIGITALIZADOS
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forma = st.selectbox(
            "Selecciona la Forma (Determina $I$):",
            ("Disco/Cilindro Sólido (c=0.5)", "Cilindro Hueco (Anillo, c=1.0)", "Esfera Sólida (c=0.4)")
        )
    with col2:
        masa = st.number_input("Masa ($M$, kg):", 0.1, 10.0, 2.0, 0.1)
        radio = st.number_input("Radio ($R$, m):", 0.1, 1.0, 0.5, 0.05)
    with col3:
        torque = st.number_input("Torque Aplicado ($\\tau$, N·m):", 0.1, 5.0, 1.0, 0.1)
        t_max = st.number_input("Tiempo de Simulación ($t_{max}$, s):", 1.0, 20.0, 5.0, 0.5)
        
    # --- Cálculos y Resultados ---
    I_cm = calcular_momento_inercia(forma, masa, radio) 
    
    if I_cm <= 0:
        st.error("Error: Momento de Inercia no válido. Asegúrate de que la masa y el radio sean positivos.")
    else:
        df_sim, alfa = simular_torque(I_cm, torque, t_max)
        
        st.markdown("---")
        st.subheader("Resultados Clave (Dinámica Rotacional)")
        st.latex(f"\\text{{Momento de Inercia: }} I = {I_cm:.4f} \\, \\text{{kg}} \\cdot \\text{{m}}^2")
        st.latex(f"\\text{{Segunda Ley para Rotación: }} \\sum \\tau = I \\alpha")
        st.latex(f"\\alpha = \\frac{{\\tau}}{{I}} = \\frac{{{torque:.2f}}}{{{I_cm:.4f}}} = {alfa:.4f} \\, \\text{{rad/s}}^2")
        
        st.info(f"El objeto es un **{forma.split('(')[0].strip()}** con $\\alpha$ constante de **{alfa:.4f} rad/s²**.")

        # --- Visualización 3D Interactiva del Objeto ---
        st.subheader("Visualización 3D Animada de la Rotación")
        
        height = radio * 0.1 
        x_base, y_base, z_base = create_cylinder_mesh(radio, height)
        
        animation_placeholder = st.empty()
        
        if st.button("▶️ Iniciar Animación 3D", key="anim3d"):
            
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
                time.sleep(t_max / animation_steps) 
        
        # --- Visualización de Gráficas (Plotly Interactivo) ---
        st.subheader("Gráficas de Cinemática Rotacional (Cap. 9)")
        
        fig_omega = px.line(
            df_sim, 
            x='Tiempo (s)', 
            y='Velocidad Angular (rad/s)', 
            title=f'Velocidad Angular ($\\omega$) vs. Tiempo ($\\omega = \\alpha t$)',
            labels={'Velocidad Angular (rad/s)': 'Velocidad Angular $\\omega$ (rad/s)'}
        )
        fig_omega.update_layout(hovermode="x unified")
        st.plotly_chart(fig_omega, use_container_width=True)


# ------------------------------------------------------------
---
# ------------------------------------------------------------

elif opcion == "2️⃣ Trabajo y Energía en Rotación (K = 1/2 I $\\omega^2$)":
    st.header("2. Energía en Rotación: Cilindro Fijo con Masa Colgante")
    st.markdown("Analizamos la conversión de la **Energía Potencial Gravitacional** ($U=Mgy_{cm}$) en **Energía Cinética de Traslación** y **Energía Cinética Rotacional** ($K=1/2 I \\omega^2$)[cite: 2, 3].")
    
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
        st.subheader("Análisis Dinámico (Fuerzas y Aceleraciones)")
        st.latex(f"\\text{{Tensión del Cable: }} T = {T:.4f} \\, \\text{{N}}")
        st.latex(f"\\text{{Aceleración Lineal de la masa: }} a = {a:.4f} \\, \\text{{m/s}}^2")
        st.latex(f"\\text{{Aceleración Angular del cilindro: }} \\alpha = \\frac{{a}}{{R}} = {alfa:.4f} \\, \\text{{rad/s}}^2")

        # --- Visualización de Energía ---
        st.subheader("Distribución de Energía (Conservación)")
        
        fig_ener = px.line(
            df_ener, 
            x='Tiempo (s)', 
            y=['Energía Rotacional (J)', 'Energía Traslacional (J)', 'Energía Potencial (J)', 'Energía Total (J)'], 
            title='Conversión de Energía Potencial a Cinética',
            labels={'value': 'Energía (J)', 'variable': 'Tipo de Energía'}
        )
        fig_ener.update_layout(hovermode="x unified")
        st.plotly_chart(fig_ener, use_container_width=True)
        
        st.subheader("Explicación del Flujo de Energía")
        st.markdown("""
        * **Conservación de la Energía:** La Energía Total del sistema se **conserva (línea horizontal)** ya que el trabajo realizado por la fricción es despreciable en este sistema ideal.
        * **Transformación:** La **Energía Potencial Gravitacional** del bloque ($U = mgh$) [cite: 3] se transforma continuamente en:
            1.  Energía Cinética de Traslación del bloque ($K_{\\text{tras}} = 1/2 m v^2$).
            2.  Energía Cinética Rotacional del cilindro ($K_{\\text{rot}} = 1/2 I \\omega^2$)[cite: 2].
        """)

# ------------------------------------------------------------
---
# ------------------------------------------------------------

elif opcion == "3️⃣ Conservación del Momento Angular ($L = I\\omega$)":
    st.header("3. Conservación del Momento Angular: El Patinador o la Placa de Embrague")
    st.markdown("Modelamos un sistema donde el **Momento Angular ($L$) se conserva** porque el torque externo neto es cero ($\sum \\tau_{\\text{ext}} = 0$)[cite: 10]. Al cambiar el Momento de Inercia ($I$) por un movimiento interno (ej. brazos), la **Velocidad Angular ($\omega$)** debe cambiar para mantener $L = I\\omega$ constante[cite: 9].")
    
    st.markdown("---")
    st.subheader("Configuración del Sistema (Valores Digitales)")
    
    col1, col2 = st.columns(2)
    with col1:
        # I_0 simula la inercia del cuerpo central 
        I_cuerpo = st.number_input("Inercia Fija del Cuerpo Central ($I_0$, kg·m²):", 0.1, 5.0, 0.5, 0.1, help="Inercia fija (ej. torso del patinador, Cap. 10.6).")
        # Masa de los brazos/masas puntuales (m1 = m2)
        m_brazo = st.number_input("Masa de Cada Brazo/Masa ($m$, kg):", 0.1, 5.0, 1.5, 0.1, help="Masa de cada una de las dos partes móviles.")
    with col2:
        r_ini = st.number_input("Distancia Radial Inicial ($r_{\\text{ini}}$, m):", 0.1, 2.0, 1.0, 0.1, help="Distancia inicial de las masas al eje.")
        omega_ini = st.number_input("Velocidad Angular Inicial ($\\omega_{\\text{ini}}$, rad/s):", 0.1, 5.0, 1.0, 0.1, help="Velocidad de giro inicial del sistema.")

    st.markdown("---")
    st.subheader("Experimentación Virtual (Cambio de la Geometría)")
    
    # Control SLIDER para cambiar la distancia radial (Experimentación)
    r_final = st.slider("Distancia Radial Final ($r_{\\text{final}}$, m):", 0.1, 2.0, 0.2, 0.05, help="Simula recoger (valor pequeño) o extender (valor grande) los brazos.")

    # --- Cálculos Basados en el Modelo de Patinador ---
    
    # Modelo de Inercia: I = I_0 + 2 * m * r^2
    I_inicial = I_cuerpo + 2 * m_brazo * r_ini**2
    I_final = I_cuerpo + 2 * m_brazo * r_final**2
    
    # Momento Angular (L se conserva) [cite: 10]
    L = I_inicial * omega_ini
    
    # Velocidad Angular Final: I_ini * omega_ini = I_final * omega_final
    omega_final = L / I_final if I_final != 0 else 0
    
    # Energías Cinéticas de Rotación (K = 0.5 * I * omega^2) [cite: 2]
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
    st.subheader("Conclusión Física (Cap. 10.6)")

    if r_final < r_ini:
        st.error(f"**El Patinador Acelera:** $\\omega$ aumentó de ${omega_ini:.2f}$ rad/s a **${omega_final:.2f}$ rad/s**.")
        st.warning(f"El trabajo realizado por las fuerzas internas para acercar la masa incrementa la **Energía Cinética Rotacional** ($K_{{final}} > K_{{inicial}}$), aunque el Momento Angular se conserva[cite: 9].")
    elif r_final > r_ini:
        st.error(f"**El Patinador Frena:** $\\omega$ disminuyó de ${omega_ini:.2f}$ rad/s a **${omega_final:.2f}$ rad/s**.")
        st.warning(f"La extensión de los brazos reduce la Energía Cinética Rotacional del sistema ($K_{{final}} < K_{{inicial}}$).")
    else:
        st.info(f"**El sistema se mantiene en equilibrio:** $I$ y $\\omega$ no cambian.")


# ------------------------------------------------------------
---
# ------------------------------------------------------------

elif opcion == "4️⃣ Rodadura en Plano Inclinado (Energía y Aceleración)":
    st.header("4. Rodadura de Cuerpos Rígidos en un Plano Inclinado")
    st.markdown("Analizamos la rodadura sin deslizamiento, donde la energía total se conserva[cite: 8]. La **Energía Cinética Total** es la suma de la traslacional del centro de masa y la rotacional en torno al centro de masa: $K = K_{\\text{tras}} + K_{\\text{rot}}$[cite: 7].")
    
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

        # Constantes de Momento de Inercia (c = I_cm / (M*R^2))
        formas_C = {
            "Esfera Sólida (c=0.4)": 0.4,
            "Disco/Cilindro Sólido (c=0.5)": 0.5,
            "Esfera Hueca (c≈0.667)": 2/3,
            "Cilindro Hueco (Anillo, c=1.0)": 1.0
        }
        
        resultados = []
        
        for forma, C in formas_C.items():
            # Aceleración lineal para rodadura pura (Resultado del Cap. 10.3, Ejemplo 10.5)
            # a_cm = g * sin(theta) / (1 + c) [cite: 8]
            a = (g * np.sin(theta_rad)) / (1 + C)
            
            if a > 1e-6: 
                # Tiempo para recorrer la distancia L: t = sqrt(2L / a)
                t = np.sqrt((2 * L_plano) / a)
                # Velocidad final: v_cm = a * t
                v_final = a * t
            else: 
                t = np.inf
                v_final = 0
            
            resultados.append({
                'Forma': forma,
                'Aceleración a_cm (m/s²)': a,
                'Tiempo de Descenso t (s)': t,
                'Velocidad Final v_cm (m/s)': v_final
            })
            
        df_rodadura = pd.DataFrame(resultados).sort_values(by='Tiempo de Descenso t (s)')

        st.markdown("---")
        st.subheader("Resultados de la Carrera (Ordenado por Tiempo de Descenso)")
        st.dataframe(df_rodadura.style.format({
            'Tiempo de Descenso t (s)': lambda x: f"{x:.2f}" if x != np.inf else "∞",
            'Aceleración a_cm (m/s²)': "{:.2f}",
            'Velocidad Final v_cm (m/s)': "{:.2f}"
            }), hide_index=True, use_container_width=True)

        st.subheader("Explicación Física (Cap. 10.3) 🏆")
        st.markdown(f"""
        * **Aceleración del Centro de Masa ($a_{\\text{cm}}$):** Está dada por la relación:
        $$a_{{\\text{{cm}}}} = \\frac{{g \\sin(\\theta)}}{{1 + c}}$$
        Donde $c = I_{{\\text{{cm}}}} / MR^2$.
        * **Conclusión (Ejemplo 10.5):** El cuerpo con el valor de **$c$ más pequeño** llega primero[cite: 8]. Esto se debe a que una $c$ pequeña significa que una fracción menor de la energía potencial inicial se convierte en $K_{\\text{rot}}$ y una fracción mayor se convierte en $K_{\\text{tras}}$, lo que resulta en una mayor velocidad lineal $v_{\\text{cm}}$ y, por lo tanto, en un menor tiempo de descenso.
        * **Independencia:** La aceleración lineal **no depende de la masa $M$ ni del radio $R$** del objeto, solo de su forma geométrica ($c$).
        """)
