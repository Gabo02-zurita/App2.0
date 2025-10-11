import streamlit as st
import numpy as np # pyright: ignore[reportMissingImports]
import plotly.graph_objects as go # pyright: ignore[reportMissingImports]
from scipy.constants import g # pyright: ignore[reportMissingImports] # Gravedad terrestre
from scipy.integrate import odeint # pyright: ignore[reportMissingImports] # Para resolver ecuaciones diferenciales, útil en animaciones complejas

# -------------------- Configuración de la Página (¡DEBE SER LA PRIMERA!) --------------------
st.set_page_config(layout="wide", page_title="Simulaciones de Física: Impulso y Cantidad de Movimiento")

# --- CSS Personalizado para la Interfaz Creativa (Va DESPUÉS de set_page_config) ---
background_image_url = "https://i.postimg.cc/CMstreBSnW4f/eee.jpg" # ¡CAMBIA ESTA URL por una tuya si la prueba funciona!

# Importante: Asegúrate de que no haya espacios o caracteres invisibles antes de '<style>'
st.markdown(
    f"""
<style>
/* Estilos para el fondo de la aplicación */
.stApp {{
    background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("{background_image_url}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* Estilos para el contenedor principal del contenido */
.stSidebar {{
    background-color: rgba(0, 0, 0, 0.8);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
}}

/* Estilos para la barra lateral */
.stSidebar {{
    background-color: rgba(240, 240, 240, 0.9);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
}}

/* Estilos para los encabezados (H1 a H6) - ROJO */
h1[data-testid="stAppViewTitle"],
h2[data-testid^="stHeader"],
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6
{{
    color: red !important;
    text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5);
    font-size: 2.2em;
    font-weight: bold;
}}

/* Estilos para el texto de párrafo, listas, span y divs generales - NARANJA */
.stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown div {{
    color: white !important;
    font-size: 1.1em;
    font-weight: 300;
}}

/* Para el valor numérico en st.metric */
.st-bd {{
    color: white !important;
}}

/* Para el texto dentro de st.info, st.warning, st.error boxes */
.st-dg {{
    color: #555555 !important;
    font-weight: 500;
}}

/* Para los labels de los widgets (sliders, inputs, selectbox, radio) - AHORA TAMBIÉN EN ROJO */
.stSlider label, .stNumberInput label, .stSelectbox label, .stRadio label {{
    font-size: 1.15em;
    font-weight: 600;
    color: white !important; /* ¡CAMBIADO A ROJO! */
}}

/* Para el texto de las opciones de radio buttons y selectboxes */
div[data-testid="stRadio"] label span,
div[data-testid="stSelectbox"] div[role="button"] span {{
    color: orange !important; /* Las opciones mismas, siguen en naranja */
}}

/* Estilos para el texto dentro de los botones */
.stButton > button {{
    font-size: 1.1em;
    font-weight: 600;
    color: #333333 !important;
}}

/* Asegurar que el texto dentro de los "streamlit.latex" también se vea afectado */
.st-be.st-bb, .st-bh {{
    font-size: 1.1em !important;
    font-weight: 500 !important;
    color: orange !important;
}}

</style>
    """,
    unsafe_allow_html=True
)


def calcular_impulso_fuerza(parametro_entrada, valor_entrada, tiempo=None):
    """Calcula impulso o fuerza promedio."""
    if parametro_entrada == "impulso":
        # Se tiene fuerza y tiempo, calcular impulso
        impulso = valor_entrada * tiempo
        return impulso, f"Impulso: {impulso:.2f} Ns"
    elif parametro_entrada == "fuerza_promedio":    
        # Se tiene impulso y tiempo, calcular fuerza promedio
        fuerza = valor_entrada / tiempo
        return fuerza, f"Fuerza promedio: {fuerza:.2f} N"

def simular_colision_1d(m1, v1_inicial, m2, v2_inicial, e):
    """
    Simula una colisión unidimensional (elástica, inelástica o parcialmente elástica).
    e: coeficiente de restitución (0 para inelástica, 1 para elástica)
    """
    # Conservación de la cantidad de movimiento: m1*v1i + m2*v2i = m1*v1f + m2*v2f
    # Coeficiente de restitución: e = -(v1f - v2f) / (v1i - v2i) => v1f - v2f = -e * (v1i - v2i)

    v1_final = ((m1 - e * m2) * v1_inicial + (1 + e) * m2 * v2_inicial) / (m1 + m2)
    v2_final = ((1 + e) * m1 * v1_inicial + (m2 - e * m1) * v2_inicial) / (m1 + m2)
    
    return v1_final, v2_final

def simular_colision_2d(m1, v1_inicial_x, v1_inicial_y, m2, v2_inicial_x, v2_inicial_y, e, angulo_impacto_deg):
    """
    Simula una colisión 2D entre dos partículas.
    Simplificado: asume que el impacto ocurre a lo largo de un eje definido por angulo_impacto_deg.
    Para una colisión más real, necesitarías la posición de los centros y el radio de las partículas.
    """
    angulo_impacto_rad = np.deg2rad(angulo_impacto_deg)

    # Transformar velocidades a un sistema de coordenadas donde el eje x' está a lo largo de la línea de impacto
    v1i_normal = v1_inicial_x * np.cos(angulo_impacto_rad) + v1_inicial_y * np.sin(angulo_impacto_rad)
    v1i_tangencial = -v1_inicial_x * np.sin(angulo_impacto_rad) + v1_inicial_y * np.cos(angulo_impacto_rad)
    v2i_normal = v2_inicial_x * np.cos(angulo_impacto_rad) + v2_inicial_y * np.sin(angulo_impacto_rad)
    v2i_tangencial = -v2_inicial_x * np.sin(angulo_impacto_rad) + v2_inicial_y * np.cos(angulo_impacto_rad)

    # Aplicar colisión 1D en el eje normal
    v1f_normal, v2f_normal = simular_colision_1d(m1, v1i_normal, m2, v2i_normal, e)

    # Las velocidades tangenciales se conservan
    v1f_tangencial = v1i_tangencial
    v2f_tangencial = v2i_tangencial

    # Transformar velocidades finales de vuelta al sistema de coordenadas original (x, y)
    v1_final_x = v1f_normal * np.cos(angulo_impacto_rad) - v1f_tangencial * np.sin(angulo_impacto_rad)
    v1_final_y = v1f_normal * np.sin(angulo_impacto_rad) + v1f_tangencial * np.cos(angulo_impacto_rad)
    v2_final_x = v2f_normal * np.cos(angulo_impacto_rad) - v2f_tangencial * np.sin(angulo_impacto_rad)
    v2_final_y = v2f_normal * np.sin(angulo_impacto_rad) + v2f_tangencial * np.cos(angulo_impacto_rad)

    return (v1_final_x, v1_final_y), (v2_final_x, v2_final_y)

def calcular_v_sistema_pendulo(masa_bloque, masa_bala, velocidad_bala_inicial):
    """
    Calcula la velocidad del sistema bala+bloque justo después del impacto.
    Asume una colisión perfectamente inelástica.
    """
    # Conservación de la Cantidad de Movimiento (colisión inelástica)
    return (masa_bala * velocidad_bala_inicial) / (masa_bala + masa_bloque)

def calcular_h_max_pendulo(masa_bloque, masa_bala, velocidad_bala_inicial):
    """
    Calcula la altura máxima alcanzada por el sistema bala+bloque.
    """
    v_sistema = calcular_v_sistema_pendulo(masa_bloque, masa_bala, velocidad_bala_inicial)
    # Conservación de la Energía Mecánica (sistema bala+bloque asciende)
    h_max = (v_sistema**2) / (2 * g)
    return h_max

def simular_flecha_saco(m_flecha, v_flecha_inicial, m_saco, mu_k):
    """
    Simula una flecha que se incrusta en un saco y lo desplaza hasta detenerse.
    """
    # 1. Colisión perfectamente inelástica (flecha se incrusta en saco)
    v_sistema_inicial = (m_flecha * v_flecha_inicial) / (m_flecha + m_saco)

    # 2. Movimiento del sistema con fricción
    m_total = m_flecha + m_saco
    F_friccion = mu_k * m_total * g # Fuerza de fricción cinética
    a_friccion = -F_friccion / m_total # Aceleración debido a la fricción (negativa)

    # 3. Distancia recorrida hasta detenerse (v_final^2 = v_inicial^2 + 2*a*d)
    if a_friccion == 0: # Evitar división por cero si no hay fricción
        distancia_detencion = float('inf') # Se movería indefinidamente
    else:
        distancia_detencion = - (v_sistema_inicial**2) / (2 * a_friccion)

    return v_sistema_inicial, F_friccion, distancia_detencion

def simular_caida_plano_impacto(m_obj, altura_inicial, angulo_plano_deg, mu_k_plano, e_impacto):
    """
    Simula un objeto deslizándose por un plano inclinado y luego impactando el suelo.
    """
    angulo_plano_rad = np.deg2rad(angulo_plano_deg)

    # 1. Movimiento en el plano inclinado
    g_paralelo = g * np.sin(angulo_plano_rad)
    g_perpendicular = g * np.cos(angulo_plano_rad)
    F_normal = m_obj * g_perpendicular
    F_friccion_plano = mu_k_plano * F_normal
    a_plano = g_paralelo - (F_friccion_plano / m_obj)

    if a_plano < 0: # Si la fricción es muy alta y no se mueve
        st.warning("El objeto no se moverá por el plano inclinado debido a la alta fricción.")
        return 0, 0, 0, 0, 0, 0, 0

    longitud_plano = altura_inicial / np.sin(angulo_plano_rad)
    v_final_plano = np.sqrt(2 * a_plano * longitud_plano)

    # 2. Impacto con el suelo (horizontal)
    vx_impacto = v_final_plano * np.cos(angulo_plano_rad)
    vy_impacto = -v_final_plano * np.sin(angulo_plano_rad)

    # Velocidad vertical de rebote (solo afecta la componente Y)
    vy_rebote = -e_impacto * vy_impacto

    # 3. Trayectoria después del rebote (tiro parabólico)
    altura_max_rebote = (vy_rebote**2) / (2 * g)
    tiempo_vuelo_rebote = (2 * vy_rebote) / g

    distancia_horizontal_rebote = vx_impacto * tiempo_vuelo_rebote

    return (a_plano, v_final_plano, vx_impacto, vy_impacto,
            vy_rebote, altura_max_rebote, distancia_horizontal_rebote)

# -------------------- Funciones de Visualización (Plotly) --------------------

def plot_colision_1d_animacion(m1, v1_inicial, m2, v2_inicial, e):
    v1_f, v2_f = simular_colision_1d(m1, v1_inicial, m2, v2_inicial, e)

    pos_inicial_1 = -5
    pos_inicial_2 = 5
    radio_1 = m1**0.3 * 0.5 # Tamaño visual basado en masa
    radio_2 = m2**0.3 * 0.5

    num_frames = 100
    t = np.linspace(0, 2, num_frames) # Tiempo total de la animación

    frames = []
    for k in range(num_frames):
        # Antes de la colisión (asumiendo que colisionan alrededor de t=1)
        if t[k] < 1:
            x1 = pos_inicial_1 + v1_inicial * t[k]
            x2 = pos_inicial_2 + v2_inicial * t[k]
        # Después de la colisión (simplificado, asume que la colisión es instantánea en t=1)
        else:
            x1 = pos_inicial_1 + v1_inicial * 1 + v1_f * (t[k] - 1)
            x2 = pos_inicial_2 + v2_inicial * 1 + v2_f * (t[k] - 1)

        # Simplificación para evitar superposición visual en el momento de impacto
        if abs(x1 - x2) < (radio_1 + radio_2) * 0.8 and t[k] < 1.05:
            pass
        else:
            if t[k] < 1:
                x1 = pos_inicial_1 + v1_inicial * t[k]
                x2 = pos_inicial_2 + v2_inicial * t[k]
            else:
                x1 = pos_inicial_1 + v1_inicial * 1 + v1_f * (t[k] - 1)
                x2 = pos_inicial_2 + v2_inicial * 1 + v2_f * (t[k] - 1)

        frame_data = [
            go.Scatter(x=[x1], y=[0], mode='markers', marker=dict(size=radio_1*20, color='blue'), name=f'Objeto 1 (Masa: {m1} kg)'),
            go.Scatter(x=[x2], y=[0], mode='markers', marker=dict(size=radio_2*20, color='red'), name=f'Objeto 2 (Masa: {m2} kg)')
        ]
        frames.append(go.Frame(data=frame_data, name=str(k)))

    fig = go.Figure(
        data=[
            go.Scatter(x=[pos_inicial_1], y=[0], mode='markers', marker=dict(size=radio_1*20, color='blue')),
            go.Scatter(x=[pos_inicial_2], y=[0], mode='markers', marker=dict(size=radio_2*20, color='red'))
        ],
        layout=go.Layout(
            xaxis=dict(range=[-10, 10], autorange=False, zeroline=False),
            yaxis=dict(range=[-1, 1], autorange=False, showgrid=False, zeroline=False, showticklabels=False),
            title='Simulación de Colisión 1D',
            updatemenus=[dict(type='buttons', buttons=[dict(label='Play',
                                                             method='animate',
                                                             args=[None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate'}])])]
        ),
        frames=frames
    )
    return fig

def plot_colision_2d_trayectorias(m1, v1_ix, v1_iy, m2, v2_ix, v2_iy, e, angulo_impacto_deg):
    """
    Genera una visualización 2D de trayectorias antes y después de la colisión.
    """
    (v1fx, v1fy), (v2fx, v2fy) = simular_colision_2d(m1, v1_ix, v1_iy, m2, v2_ix, v2_iy, e, angulo_impacto_deg)

    # Puntos de partida para las trayectorias (arbitrarios para visualización)
    p1_start = [-10, 0]
    p2_start = [10, 0]

    # Punto de colisión (arbitrario, por ejemplo, el origen)
    colision_point = [0, 0]

    # Calcular puntos de la trayectoria antes de la colisión
    t_pre_colision = np.linspace(-1, 0, 50)
    x1_pre = [p1_start[0] + v1_ix * t for t in t_pre_colision]
    y1_pre = [p1_start[1] + v1_iy * t for t in t_pre_colision]
