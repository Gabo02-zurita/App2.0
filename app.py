import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time # Para simular animaciones en Streamlit

# ----------------- Funciones de Cálculo para las Simulaciones -----------------

def calcular_momento_inercia(forma, masa, radio, longitud=None):
    """Calcula el momento de inercia I para diferentes geometrías."""
    if forma == "Disco/Cilindro Sólido":
        return 0.5 * masa * radio**2
    elif forma == "Cilindro Hueco (Anillo)":
        return masa * radio**2
    elif forma == "Varilla (Eje Central)":
        # Para la varilla, la longitud es crucial, si no se proporciona, devuelve 0 (error).
        return (1/12) * masa * longitud**2 if longitud else 0
    elif forma == "Esfera Sólida":
        return (2/5) * masa * radio**2
    return 0

def simular_torque(I, tau, t_max, dt=0.05):
    """Calcula variables cinemáticas para un torque constante."""
    alfa = tau / I  # Aceleración angular constante
    
    tiempo = np.arange(0, t_max + dt, dt) # Usar arange para mejor control del paso de tiempo
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
    I_cil = 0.5 * M_cil * R_cil**2  # Momento de inercia del cilindro
    g = 9.81  # Aceleración de la gravedad
    
    # Cálculo de la aceleración lineal de la masa 'a'
    # a = g / (1 + I / (m*R^2))
    # Aquí es importante manejar la división por cero si m_masa o R_cil son 0
    if m_masa == 0 or R_cil == 0:
        a = 0
    else:
        a = g / (1 + I_cil / (m_masa * R_cil**2))
    
    # Tensión del cable (T = m*(g-a))
    T = m_masa * (g - a)
    
    # Aceleración angular del cilindro: alfa = a / R
    alfa = a / R_cil if R_cil != 0 else 0
    
    # Energía: solo si la masa se mueve una distancia 'h'
    tiempo = np.arange(0, t_max + dt, dt)
    h = 0.5 * a * tiempo**2  # Distancia que cae la masa
    
    # Energías en función del tiempo
    K_rot = 0.5 * I_cil * (alfa * tiempo)**2  # Energía Cinética Rotacional (E_rot = 0.5*I*omega^2)
    K_tras = 0.5 * m_masa * (a * tiempo)**2  # Energía Cinética Traslacional (E_tras = 0.5*m*v^2)
    
    # La energía potencial disminuye, tomamos la referencia inicial de 0 en el punto más bajo.
    # U_grav = m_masa * g * (h_max - h)
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
    """Crea una malla para un cilindro 3D."""
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
    
    # Reshape x, y, z para aplicar la rotación
    coords = np.vstack([x_base.flatten(), y_base.flatten(), z_base.flatten()])
    rotated_coords = Rz @ coords
    
    # Volver a las dimensiones originales
    x_rotated = rotated_coords[0].reshape(x_base.shape)
    y_rotated = rotated_coords[1].reshape(y_base.shape)
    z_rotated = rotated_coords[2].reshape(z_base.shape)
    
    return x_rotated, y_rotated, z_rotated

# ----------------- Configuración de la Interfaz Streamlit -----------------

st.set_page_config(layout="wide", page_title="Rotación de Sólidos Rígidos")

# Título principal y explicación
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
    La **rotación de un sólido rígido** es el movimiento de un objeto en el que cada punto describe un círculo alrededor de un eje fijo. Los conceptos clave son:

    * **Momento de Inercia ($I$):** Es la resistencia del objeto a cambiar su estado de movimiento rotacional (análogo a la masa en la traslación). Depende de la masa y cómo está distribuida respecto al eje de giro.
    * **Torque ($\\tau$):** Es la fuerza rotacional neta que provoca el cambio en el movimiento rotacional (análogo a la fuerza $F$).
    * **Segunda Ley de Newton para Rotación:** $\\tau = I \\alpha$ (análogo a $F = ma$), donde $\\alpha$ es la aceleración angular.
    * **Momento Angular ($L$):** Medida de la "cantidad de rotación" de un objeto. Se conserva si el torque externo neto es cero ($\\tau_{neto} = 0$). ($L = I \\omega$).
    """)
    st.info("¡Usa el menú lateral para seleccionar una simulación y experimentar virtualmente!")

# ------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "1️⃣ Torque y Cinemática Rotacional":
    st.header("1. Simulación de Torque Constante y Cinemática 📈")
    st.markdown("Aplica un torque constante a una forma geométrica para observar cómo varían sus parámetros de rotación con el tiempo. ¡Ahora con visualización animada en 3D!")

    # Controles de entrada DIGITALIZADOS
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forma = st.selectbox(
            "Selecciona la Forma:",
            ("Disco/Cilindro Sólido", "Cilindro Hueco (Anillo)", "Esfera Sólida") # Eliminamos Varilla por simplicidad en la visualización 3D de un disco/cilindro
        )
    with col2:
        masa = st.number_input("Masa ($M$, kg):", 0.1, 10.0, 2.0, 0.1)
        radio = st.number_input("Radio ($R$, m):", 0.1, 1.0, 0.5, 0.05)
    with col3:
        torque = st.number_input("Torque Aplicado ($\\tau$, N·m):", 0.1, 5.0, 1.0, 0.1)
        t_max = st.number_input("Tiempo de Simulación ($t_{max}$, s):", 1.0, 20.0, 5.0, 0.5)
        
    # --- Cálculos y Resultados ---
    I = calcular_momento_inercia(forma, masa, radio) # Varilla fue eliminada del selectbox
    
    if I == 0:
        st.error("Error: Momento de Inercia no definido. Asegúrate de que el radio y la masa sean válidos.")
    else:
        df_sim, alfa = simular_torque(I, torque, t_max)
        
        st.markdown("---")
        st.subheader("Resultados Teóricos Clave")
        st.latex(f"I = {I:.4f} \\, \\text{{kg}} \\cdot \\text{{m}}^2")
        st.latex(f"\\tau = I \\alpha \\Rightarrow \\alpha = \\frac{{\\tau}}{{I}} = \\frac{{{torque:.2f}}}{{{I:.4f}}} = {alfa:.4f} \\, \\text{{rad/s}}^2")
        st.latex(f"\\omega(t) = \\alpha t \\quad | \\quad \\theta(t) = \\frac{{1}}{{2}} \\alpha t^2")
        
        st.info(f"El objeto es un **{forma}** con un **Momento de Inercia ($I$)** de **{I:.4f} kg·m²**.")

        # --- Visualización 3D Interactiva del Objeto ---
        st.subheader("Visualización 3D Animada del Objeto Girando")
        
        # Crear la base del cilindro/disco
        height = radio * 0.1 # Un disco es un cilindro muy corto
        x_base, y_base, z_base = create_cylinder_mesh(radio, height)
        
        # Contenedor para la animación 3D
        animation_placeholder = st.empty()
        
        if st.button("Iniciar Animación 3D"):
            # Ajustar la velocidad de la animación para que sea perceptible
            animation_steps = 100
            time_steps = np.linspace(0, t_max, animation_steps)
            
            for i in range(animation_steps):
                current_time = time_steps[i]
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
                    title=f"Tiempo: {current_time:.2f} s | Ángulo: {np.degrees(current_theta):.1f}°",
                    scene_aspectmode='cube',
                    scene=dict(
                        xaxis=dict(range=[-radio*1.2, radio*1.2]),
                        yaxis=dict(range=[-radio*1.2, radio*1.2]),
                        zaxis=dict(range=[-height, height]),
                        aspectratio=dict(x=1, y=1, z=1),
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                    ),
                    margin=dict(l=0, r=0, b=0, t=40),
                    height=400
                )
                
                animation_placeholder.plotly_chart(fig_3d, use_container_width=True)
                time.sleep(t_max / animation_steps) # Controla la velocidad de la animación
        
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

        fig_theta = px.line(
            df_sim, 
            x='Tiempo (s)', 
            y=['Ángulo Girado (rad)', 'Número de Vueltas'], 
            title='Ángulo Girado ($\\theta$) y Número de Vueltas vs. Tiempo',
            labels={'value': 'Magnitud'}
        )
        fig_theta.update_layout(hovermode="x unified")
        st.plotly_chart(fig_theta, use_container_width=True)

        st.subheader("Explicación Física")
        st.markdown(f"""
        * **Aceleración Angular ($\\alpha$):** Es **constante** e igual a **{alfa:.4f} rad/s²**, determinada por el torque aplicado ($\\tau$) y el momento de inercia ($I$).
        * **Velocidad Angular ($\omega$):** Aumenta **linealmente** con el tiempo, ya que la aceleración es constante ($\\omega = \\alpha t$).
        * **Ángulo Girado ($\\theta$):** Aumenta **cuadráticamente** con el tiempo, lo que se traduce en una curva parabólica en la gráfica ($\\theta = \\frac{1}{2} \\alpha t^2$).
        """)

# ------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "2️⃣ Masa Colgante y Cilindro Fijo":
    st.header("2. Cilindro Fijo con Masa Colgante ⛓️")
    st.markdown("Un cable enrollado alrededor de un cilindro fijo (eje de rotación) está unido a una masa que cae, analizando la dinámica y la energía del sistema.")
    
    # Controles de entrada DIGITALIZADOS
    col1, col2 = st.columns(2)
    with col1:
        M_cil = st.number_input("Masa del Cilindro ($M_{cil}$, kg):", 0.1, 10.0, 2.0, 0.1)
        R_cil = st.number_input("Radio del Cilindro ($R_{cil}$, m):", 0.05, 1.0, 0.2, 0.05)
    with col2:
        m_masa = st.number_input("Masa Colgante ($m_{masa}$, kg):", 0.01, 5.0, 1.0, 0.01)
        t_max = st.number_input("Tiempo de Simulación ($t_{max}$, s):", 0.5, 10.0, 3.0, 0.5)

    # Validaciones básicas para evitar divisiones por cero o resultados no físicos
    if R_cil <= 0:
        st.error("El radio del cilindro debe ser mayor que cero.")
    elif m_masa <= 0:
        st.error("La masa colgante debe ser mayor que cero.")
    else:
        # --- Cálculos y Resultados ---
        df_ener, a, alfa, T = simular_masa_colgante(m_masa, R_cil, M_cil, t_max)

        st.markdown("---")
        st.subheader("Resultados Teóricos Clave")
        st.latex(f"a = {a:.4f} \\, \\text{{m/s}}^2 \\quad | \\quad \\alpha = {alfa:.4f} \\, \\text{{rad/s}}^2 \\quad | \\quad T = {T:.4f} \\, \\text{{N}}")
        st.markdown(f"La **aceleración de la masa** es $a = {a:.4f} \\, \\text{{m/s}}^2$ y la **tensión del cable** es $T = {T:.4f} \\, \\text{{N}}$.")

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
        * **Dinámica:** La tensión del cable ($T$) genera el torque en el cilindro ($\\tau = T \\cdot R$), mientras que la gravedad y la tensión actúan sobre la masa ($m_{masa} g - T = m_{masa} a$).
        * **Conservación de la Energía:** Se asume que no hay fricción, por lo que la **Energía Total** del sistema se **conserva (línea horizontal)**. La **Energía Potencial** de la masa colgante ($U$) se transforma en **Energía Cinética Traslacional** de la masa ($K_{tras}$) y **Energía Cinética Rotacional** del cilindro ($K_{rot}$).
        """)

# ------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "3️⃣ Conservación del Momento Angular":
    st.header("3. Conservación del Momento Angular (El Patinador) ⛸️")
    st.markdown("Ilustra el principio de conservación del momento angular: si el torque externo neto es cero, el momento angular ($L$) se mantiene constante, incluso si el momento de inercia ($I$) cambia.")
    
    st.markdown("---")
    st.subheader("Variables del Patinador (o Plataforma Giratoria)")
    
    col1, col2 = st.columns(2)
    with col1:
        I_ini = st.number_input("Momento de Inercia Inicial ($I_{ini}$, kg·m²):", 0.1, 20.0, 5.0, 0.1, help="Patinador con brazos extendidos o masa distribuida lejos del eje.")
        omega_ini = st.number_input("Velocidad Angular Inicial ($\\omega_{ini}$, rad/s):", 0.1, 10.0, 1.0, 0.1, help="Velocidad de giro inicial.")
    with col2:
        I_final = st.number_input("Momento de Inercia Final ($I_{final}$, kg·m²):", 0.1, 20.0, 1.0, 0.1, help="Patinador con brazos recogidos o masa concentrada cerca del eje.")
    
    # Asegurarse de que I_final no sea cero para evitar división por cero
    if I_final <= 0:
        st.error("El Momento de Inercia Final no puede ser cero o negativo. Por favor, ingrese un valor positivo.")
    else:
        # --- Cálculos y Resultados ---
        # L_inicial = L_final
        L = I_ini * omega_ini
        omega_final = L / I_final
        K_ini = 0.5 * I_ini * omega_ini**2
        K_final = 0.5 * I_final * omega_final**2

        st.markdown("---")
        st.subheader("Análisis de la Conservación")
        
        st.latex(f"L_{{inicial}} = I_{{ini}} \\omega_{{ini}} = ({I_ini:.2f})({omega_ini:.2f}) = {L:.2f} \\, \\text{{kg}} \\cdot \\text{{m}}^2/\\text{{s}}")
        st.latex(f"L_{{final}} = L_{{inicial}} \\Rightarrow \\omega_{{final}} = \\frac{{L_{{inicial}}}}{{I_{{final}}}} = \\frac{{{L:.2f}}}{{{I_final:.2f}}} = {omega_final:.2f} \\, \\text{{rad/s}}")

        st.info(f"""
        * **Momento Angular Conservado ($L$):** **{L:.2f} kg·m²/s**.
        * **Velocidad Angular Final ($\omega_{{final}}$):** **{omega_final:.2f} rad/s**. (¡El patinador gira más rápido al reducir $I$!)
        """)

        st.subheader("Análisis de la Energía Cinética de Rotación ($K_{rot}$)")
        st.latex(f"K_{{rot, ini}} = \\frac{{1}}{{2}} I_{{ini}} \\omega_{{ini}}^2 = {K_ini:.2f} \\, \\text{{J}}")
        st.latex(f"K_{{rot, final}} = \\frac{{1}}{{2}} I_{{final}} \\omega_{{final}}^2 = {K_final:.2f} \\, \\text{{J}}")

        st.error(f"La Energía Cinética FINAL es **{K_final:.2f} J** y es **MAYOR** que la INICIAL ({K_ini:.2f} J).")
        st.subheader("Explicación Física")
        st.markdown("""
        * **Momento Angular:** Como el patinador (o el sistema) no tiene un torque externo neto, su momento angular $L$ se mantiene **constante**.
        * **Relación $I - \\omega$:** Al **reducir** el momento de inercia ($I$) al acercar los brazos, la **velocidad angular ($\omega$) debe aumentar** para mantener $L$ constante ($L = I\omega$).
        * **Energía Cinética:** La energía cinética de rotación **NO se conserva**. El trabajo para **reducir** el momento de inercia (al tirar de los brazos hacia adentro) es un **trabajo interno** realizado por el patinador, que se convierte en la energía cinética de rotación extra.
        """)

# ------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "4️⃣ Rodadura en Plano Inclinado (Extendido)":
    st.header("4. Rodadura de Varias Formas por un Plano Inclinado ⛰️")
    st.markdown("Compara el movimiento de rodadura pura (sin deslizamiento) de diferentes formas geométricas que descienden por un plano inclinado. ¡El resultado es contraintuitivo: solo importa la distribución de la masa!")
    
    # Controles de entrada DIGITALIZADOS
    col1, col2 = st.columns(2)
    with col1:
        angulo = st.number_input("Ángulo de Inclinación ($\\theta$, grados):", 5, 85, 30)
        L_plano = st.number_input("Longitud del Plano Inclinado ($L$, m):", 0.5, 20.0, 5.0)
    with col2:
        g = st.number_input("Aceleración de la Gravedad ($g$, m/s²):", 0.1, 20.0, 9.81)

    # Validaciones para evitar ángulos no físicos
    if angulo <= 0 or angulo >= 90:
        st.error("El ángulo de inclinación debe estar entre 1 y 89 grados.")
    elif L_plano <= 0:
        st.error("La longitud del plano inclinado debe ser mayor que cero.")
    else:
        # Convertir ángulo a radianes
        theta_rad = np.deg2rad(angulo)

        # Constantes de Momento de Inercia (C = I / (m*R^2))
        # C_Disco = 0.5, C_Esfera = 2/5 = 0.4, C_Anillo = 1.0, C_Esfera_Hueca = 2/3 ≈ 0.667
        formas_C = {
            "Esfera Sólida (C=0.4)": 0.4,
            "Disco/Cilindro Sólido (C=0.5)": 0.5,
            "Esfera Hueca (C≈0.667)": 2/3,
            "Cilindro Hueco (Anillo, C=1.0)": 1.0
        }
        
        resultados = []
        
        for forma, C in formas_C.items():
            # Aceleración lineal para rodadura pura
            # a = g*sin(theta) / (1 + C)
            a = (g * np.sin(theta_rad)) / (1 + C)
            
            # Tiempo para recorrer la distancia L: L = 0.5 * a * t^2
            # Manejar el caso de 'a' muy pequeña para evitar divisiones por cero en el cálculo de t
            if a > 1e-6: # Un umbral pequeño para considerar que hay aceleración
                t = np.sqrt((2 * L_plano) / a)
                v_final = a * t
            else: # Si no hay aceleración significativa (por ejemplo, ángulo muy bajo)
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
        st.subheader("Resultados de la Carrera")
        # Mostrar el DataFrame, formateando el tiempo a 2 decimales y manejar Infinito
        st.dataframe(df_rodadura.style.format({
            'Tiempo de Descenso (t, s)': lambda x: f"{x:.2f}" if x != np.inf else "∞",
            'Aceleración (a, m/s²)': "{:.2f}",
            'Velocidad Final (v, m/s)': "{:.2f}"
            }), hide_index=True, use_container_width=True)

        st.subheader("Explicación Física: ¿Quién Gana la Carrera? 🏆")
        st.markdown(f"""
        * **Aceleración (a):** La aceleración lineal de la rodadura pura es $a = \\frac{{g \\sin(\\theta)}}{{1 + C}}$, donde $C = \\frac{{I}}{{MR^2}}$ es una constante que depende de la geometría del objeto.
        * **El Factor $C$:** Cuanto **menor** es la constante $C$ (es decir, menos masa está distribuida lejos del eje de rotación), **mayor** es la aceleración $a$ y, por lo tanto, **menor** es el tiempo $t$ para descender el plano.
        * **Ganador:** La **Esfera Sólida** (C=0.4) gana la carrera porque tiene la menor constante $C$. El **Cilindro Hueco (Anillo)** (C=1.0) es el más lento, ya que toda su masa está lejos del eje.
        * **Independiente de la Masa/Radio:** Sorprendentemente, la aceleración y el tiempo de descenso **no dependen de la masa ($M$) ni del radio ($R$)** del objeto; solo dependen de la *forma* (el valor de $C$) y el ángulo del plano.
        """)
