import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd

# ----------------- Funciones de Cálculo para las Simulaciones -----------------

def calcular_momento_inercia(forma, masa, radio, longitud=None):
    """Calcula el momento de inercia I para diferentes geometrías."""
    if forma == "Disco/Cilindro Sólido":
        return 0.5 * masa * radio**2
    elif forma == "Cilindro Hueco (Anillo)":
        return masa * radio**2
    elif forma == "Varilla (Eje Central)":
        return (1/12) * masa * longitud**2 if longitud else 0 # Se necesita longitud
    elif forma == "Esfera Sólida":
        return (2/5) * masa * radio**2
    return 0

def simular_torque(I, tau, t_max):
    """Calcula variables cinemáticas para un torque constante."""
    alfa = tau / I  # Aceleración angular constante
    
    tiempo = np.linspace(0, t_max, 100)
    omega = alfa * tiempo  # Velocidad angular: omega = alfa * t
    theta = 0.5 * alfa * tiempo**2  # Ángulo girado: theta = 0.5 * alfa * t^2
    vueltas = theta / (2 * np.pi)  # Número de vueltas
    
    df = pd.DataFrame({
        'Tiempo (s)': tiempo,
        'Velocidad Angular (rad/s)': omega,
        'Ángulo Girado (rad)': theta,
        'Número de Vueltas': vueltas,
        'Aceleración Angular (rad/s^2)': [alfa] * 100
    })
    return df, alfa

def simular_masa_colgante(m_masa, R_cil, M_cil, t_max):
    """Simula masa colgante que desenrolla un cable de un cilindro."""
    I_cil = 0.5 * M_cil * R_cil**2  # Momento de inercia del cilindro
    g = 9.81  # Aceleración de la gravedad
    
    # Cálculo de la aceleración lineal de la masa 'a'
    # a = g / (1 + I / (m*R^2))
    a = g / (1 + I_cil / (m_masa * R_cil**2))
    
    # Tensión del cable (T = m*(g-a))
    T = m_masa * (g - a)
    
    # Aceleración angular del cilindro: alfa = a / R
    alfa = a / R_cil
    
    # Energía: solo si la masa se mueve una distancia 'h'
    tiempo = np.linspace(0, t_max, 100)
    h = 0.5 * a * tiempo**2  # Distancia que cae la masa
    
    # Energías en función del tiempo
    K_rot = 0.5 * I_cil * (alfa * tiempo)**2  # Energía Cinética Rotacional (E_rot = 0.5*I*omega^2)
    K_tras = 0.5 * m_masa * (a * tiempo)**2  # Energía Cinética Traslacional (E_tras = 0.5*m*v^2)
    U_grav = m_masa * g * h # Potencial gravitatoria
    
    df = pd.DataFrame({
        'Tiempo (s)': tiempo,
        'Energía Rotacional (J)': K_rot,
        'Energía Traslacional (J)': K_tras,
        'Energía Potencial (J)': U_grav,
        'Energía Total (J)': K_rot + K_tras + (m_masa * g * h.max() - U_grav) # La energía potencial disminuye, tomamos la referencia inicial.
    })
    return df, a, alfa, T

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

    * **Momento de Inercia ($I$):** Es la resistencia del objeto a cambiar su estado de movimiento rotacional (análogo a la masa en la traslación).
    * **Torque ($\\tau$):** Es la fuerza que provoca el cambio en el movimiento rotacional (análogo a la fuerza $F$).
    * **Segunda Ley de Newton para Rotación:** $\\tau = I \\alpha$ (análogo a $F = ma$).
    * **Momento Angular ($L$):** Medida de la rotación de un objeto. Se conserva si el torque externo neto es cero ($\\tau_{neto} = 0$).
    """)
    st.info("¡Usa el menú lateral para seleccionar una simulación!")

# ------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "1️⃣ Torque y Cinemática Rotacional":
    st.header("1. Simulación de Torque Constante y Cinemática 📈")
    st.markdown("Aplica un torque constante a una forma geométrica para observar cómo varían sus parámetros de rotación con el tiempo.")

    # Controles de entrada
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forma = st.selectbox(
            "Selecciona la Forma:",
            ("Disco/Cilindro Sólido", "Cilindro Hueco (Anillo)", "Esfera Sólida", "Varilla (Eje Central)")
        )
    with col2:
        masa = st.slider("Masa ($M$, kg):", 0.1, 10.0, 2.0, 0.1)
        radio = st.slider("Radio ($R$, m):", 0.1, 1.0, 0.5, 0.05)
    with col3:
        torque = st.slider("Torque Aplicado ($\\tau$, N·m):", 0.1, 5.0, 1.0, 0.1)
        t_max = st.slider("Tiempo de Simulación ($t_{max}$, s):", 1.0, 10.0, 5.0, 0.5)
        
    # La varilla requiere una longitud
    longitud = None
    if forma == "Varilla (Eje Central)":
        longitud = st.slider("Longitud de la Varilla ($L$, m):", 0.1, 2.0, 1.0, 0.1)

    # --- Cálculos y Resultados ---
    I = calcular_momento_inercia(forma, masa, radio, longitud)
    df_sim, alfa = simular_torque(I, torque, t_max)
    
    st.markdown("---")
    st.subheader("Resultados Teóricos Clave")
    st.latex(f"I = {I:.4f} \\, \\text{{kg}} \\cdot \\text{{m}}^2")
    st.latex(f"\\tau = I \\alpha \\Rightarrow \\alpha = \\frac{{\\tau}}{{I}} = \\frac{{{torque:.2f}}}{{{I:.4f}}} = {alfa:.4f} \\, \\text{{rad/s}}^2")
    st.latex(f"\\omega(t) = \\alpha t \\quad | \\quad \\theta(t) = \\frac{{1}}{{2}} \\alpha t^2")
    
    st.info(f"El objeto es una **{forma}** con un **Momento de Inercia ($I$)** de **{I:.4f} kg·m²**.")

    # --- Visualización Avanzada (Plotly) ---
    st.subheader("Gráfica de Velocidad Angular vs. Tiempo (Plotly Interactivo)")
    
    fig = px.line(
        df_sim, 
        x='Tiempo (s)', 
        y='Velocidad Angular (rad/s)', 
        title=f'Velocidad Angular ($\omega$) vs. Tiempo ($\\alpha = {alfa:.4f}$ rad/s²)',
        labels={'Velocidad Angular (rad/s)': 'Velocidad Angular $\omega$ (rad/s)'}
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Gráfica adicional
    st.subheader("Ángulo Girado y Vueltas vs. Tiempo")
    
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
    
    # Controles de entrada
    col1, col2 = st.columns(2)
    with col1:
        M_cil = st.slider("Masa del Cilindro ($M_{cil}$, kg):", 0.5, 5.0, 2.0, 0.1)
        R_cil = st.slider("Radio del Cilindro ($R_{cil}$, m):", 0.1, 0.5, 0.2, 0.05)
    with col2:
        m_masa = st.slider("Masa Colgante ($m_{masa}$, kg):", 0.1, 2.0, 1.0, 0.1)
        t_max = st.slider("Tiempo de Simulación ($t_{max}$, s):", 1.0, 5.0, 3.0, 0.5)

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
        I_ini = st.slider("Momento de Inercia Inicial ($I_{ini}$, kg·m²):", 1.0, 10.0, 5.0, 0.5, help="Patinador con brazos extendidos.")
        omega_ini = st.slider("Velocidad Angular Inicial ($\\omega_{ini}$, rad/s):", 0.1, 5.0, 1.0, 0.1, help="Velocidad de giro inicial.")
    with col2:
        I_final = st.slider("Momento de Inercia Final ($I_{final}$, kg·m²):", 0.1, 10.0, 1.0, 0.1, help="Patinador con brazos recogidos.")
    
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
    * **Velocidad Angular Final ($\omega_{{final}}$):** **{omega_final:.2f} rad/s**. (¡El patinador gira más rápido!)
    """)

    st.subheader("Análisis de la Energía Cinética de Rotación ($K_{rot}$)")
    st.latex(f"K_{{rot, ini}} = \\frac{{1}}{{2}} I_{{ini}} \\omega_{{ini}}^2 = {K_ini:.2f} \\, \\text{{J}}")
    st.latex(f"K_{{rot, final}} = \\frac{{1}}{{2}} I_{{final}} \\omega_{{final}}^2 = {K_final:.2f} \\, \\text{{J}}")

    st.error(f"La Energía Cinética FINAL es **{K_final:.2f} J** y es MAYOR que la INICIAL ({K_ini:.2f} J).")
    st.subheader("Explicación Física")
    st.markdown("""
    * **Momento Angular:** Como el patinador (o el sistema) no tiene un torque externo neto, su momento angular $L$ se mantiene **constante**.
    * **Relación $I - \\omega$:** Al **reducir** el momento de inercia ($I$) al acercar los brazos, la **velocidad angular ($\omega$) debe aumentar** para mantener $L$ constante ($L = I\omega$).
    * **Energía Cinética:** La energía cinética de rotación **NO se conserva**. El trabajo para **reducir** el momento de inercia (al tirar de los brazos hacia adentro) es un **trabajo interno** que se convierte en la energía cinética de rotación extra.
    """)

# ------------------------------------------------------------
# ------------------------------------------------------------

elif opcion == "4️⃣ Rodadura en Plano Inclinado (Extendido)":
    st.header("4. Rodadura de Varias Formas por un Plano Inclinado ⛰️")
    st.markdown("Compara el movimiento de rodadura pura (sin deslizamiento) de diferentes formas geométricas que descienden por un plano inclinado. El resultado es contraintuitivo, ¡solo importa la distribución de la masa!")
    
    # Controles de entrada
    col1, col2 = st.columns(2)
    with col1:
        angulo = st.slider("Ángulo de Inclinación ($\\theta$, grados):", 5, 60, 30)
        L_plano = st.slider("Longitud del Plano Inclinado ($L$, m):", 1.0, 10.0, 5.0)
    with col2:
        g = st.number_input("Aceleración de la Gravedad ($g$, m/s²):", 9.0, 10.0, 9.81)

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
        t = np.sqrt((2 * L_plano) / a)
        
        # Velocidad final: v = a * t
        v_final = a * t
        
        resultados.append({
            'Forma': forma,
            'Aceleración (a, m/s²)': a,
            'Tiempo de Descenso (t, s)': t,
            'Velocidad Final (v, m/s)': v_final
        })
        
    df_rodadura = pd.DataFrame(resultados).sort_values(by='Tiempo de Descenso (t, s)')

    st.markdown("---")
    st.subheader("Resultados de la Carrera")
    st.dataframe(df_rodadura, hide_index=True, use_container_width=True)

    st.subheader("Explicación Física: ¿Quién Gana la Carrera? 🏆")
    st.markdown(f"""
    * **Aceleración (a):** La aceleración lineal de la rodadura es $a = \\frac{{g \\sin(\\theta)}}{{1 + C}}$, donde $C = \\frac{{I}}{{MR^2}}$.
    * **El Factor $C$:** Cuanto **menor** es la constante $C$ (es decir, menos masa está en el exterior), **mayor** es la aceleración $a$ y **menor** es el tiempo $t$.
    * **Ganador:** La **Esfera Sólida** (C=0.4) gana la carrera porque tiene la menor distribución de masa lejos del eje. El **Cilindro Hueco (Anillo)** (C=1.0) es el más lento.
    * **Independiente de la Masa/Radio:** Sorprendentemente, la aceleración y el tiempo **no dependen de la masa ($M$) ni del radio ($R$!)**; solo dependen de la *forma* ($C$) y el ángulo.
    """)
