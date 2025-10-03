import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.constants import g # Gravedad terrestre
from scipy.integrate import odeint # Para resolver ecuaciones diferenciales, útil en animaciones complejas

# -------------------- Configuración de la Página (¡DEBE SER LA PRIMERA!) --------------------
st.set_page_config(layout="wide", page_title="Simulaciones de Física: Impulso y Cantidad de Movimiento")

# --- CSS Personalizado para la Interfaz Creativa (Va DESPUÉS de set_page_config) ---
background_image_url = "https://i.postimg.cc/bdXCLRNG/unnamed.png" # ¡CAMBIA ESTA URL por una tuya si la prueba funciona!

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
.css-1d391kg {{
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
    x2_pre = [p2_start[0] + v2_ix * t for t in t_pre_colision]
    y2_pre = [p2_start[1] + v2_iy * t for t in t_pre_colision]

    # Calcular puntos de la trayectoria después de la colisión
    t_post_colision = np.linspace(0, 1, 50)
    x1_post = [colision_point[0] + v1fx * t for t in t_post_colision]
    y1_post = [colision_point[1] + v1fy * t for t in t_post_colision]
    x2_post = [colision_point[0] + v2fx * t for t in t_post_colision]
    y2_post = [colision_point[1] + v2fy * t for t in t_post_colision]

    fig = go.Figure()

    # Trayectorias antes
    fig.add_trace(go.Scatter(x=x1_pre, y=y1_pre, mode='lines', name='Objeto 1 (Antes)', line=dict(color='blue', dash='dot')))
    fig.add_trace(go.Scatter(x=x2_pre, y=y2_pre, mode='lines', name='Objeto 2 (Antes)', line=dict(color='red', dash='dot')))

    # Objetos en el momento de la colisión
    fig.add_trace(go.Scatter(x=[colision_point[0]], y=[colision_point[1]], mode='markers',
                             marker=dict(size=m1*10, color='blue', symbol='circle'), name='Objeto 1 (Colisión)'))
    fig.add_trace(go.Scatter(x=[colision_point[0]], y=[colision_point[1]], mode='markers',
                             marker=dict(size=m2*10, color='red', symbol='circle'), name='Objeto 2 (Colisión)'))

    # Trayectorias después
    fig.add_trace(go.Scatter(x=x1_post, y=y1_post, mode='lines', name='Objeto 1 (Después)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x2_post, y=y2_post, mode='lines', name='Objeto 2 (Después)', line=dict(color='red')))

    fig.update_layout(title='Simulación de Colisión 2D con Trayectorias',
                      xaxis_title='Posición X',
                      yaxis_title='Posición Y',
                      xaxis_range=[-12, 12], yaxis_range=[-10, 10],
                      showlegend=True,
                      # Corrección para el aspect ratio
                      yaxis=dict(
                          scaleanchor="x",
                          scaleratio=1
                      ),
                      hovermode="closest")
    return fig

# Definición de la ecuación diferencial para el péndulo
def pendulo_eq(y, t, L):
    theta, omega = y
    dydt = [omega, -(g / L) * np.sin(theta)]
    return dydt

def plot_pendulo_balistico_animacion(masa_bala, masa_caja, velocidad_bala_inicial, largo_pendulo_vis):
    """
    Genera una animación del péndulo balístico con la bala impactando la caja.
    Muestra la bala, la caja y la cuerda.
    """
    # Cálculos preliminares
    v_sistema = calcular_v_sistema_pendulo(masa_caja, masa_bala, velocidad_bala_inicial)

    if largo_pendulo_vis <= 0:
        st.error("El largo del péndulo debe ser mayor que cero para la animación.")
        return go.Figure()

    # --- Configuración de Tiempos y Frames ---
    # Tiempo para que la bala alcance la caja
    distancia_inicial_bala = largo_pendulo_vis * 1.5 # La bala empieza a 1.5 veces el largo del péndulo a la izquierda
    tiempo_pre_impacto = distancia_inicial_bala / velocidad_bala_inicial if velocidad_bala_inicial > 0 else 0.01

    # Tiempo de oscilación del péndulo (se usa la aproximación del péndulo simple para el rango de tiempo)
    periodo_pendulo_simple = 2 * np.pi * np.sqrt(largo_pendulo_vis / g)
    tiempo_oscilacion = periodo_pendulo_simple * 1.5 # Para ver un poco más de una oscilación

    num_frames = 200 # Número de fotogramas para la animación
    tiempo_total_simulacion = tiempo_pre_impacto + tiempo_oscilacion
    t_values = np.linspace(0, tiempo_total_simulacion, num_frames)

    # --- Arrays para almacenar posiciones ---
    pos_bala_x = np.zeros(num_frames)
    pos_bala_y = np.zeros(num_frames)
    pos_caja_x = np.zeros(num_frames)
    pos_caja_y = np.zeros(num_frames)

    # --- FASE 1: Movimiento de la bala antes del impacto ---
    pos_caja_equilibrio_y = -largo_pendulo_vis # El péndulo cuelga hacia abajo desde (0,0)

    for i, t in enumerate(t_values):
        if t <= tiempo_pre_impacto:
            # Bala moviéndose hacia la caja
            pos_bala_x[i] = -distancia_inicial_bala + velocidad_bala_inicial * t
            pos_bala_y[i] = pos_caja_equilibrio_y + 0.1 # Pequeño offset para que la bala no esté exactamente en el centro de la caja

            # Caja estática
            pos_caja_x[i] = 0
            pos_caja_y[i] = pos_caja_equilibrio_y
        else:
            # --- FASE 2: Movimiento del péndulo (bala + caja) ---
            t_post_impacto = t - tiempo_pre_impacto

            y0_pendulo = [0, v_sistema / largo_pendulo_vis]

            # Resolver la ecuación diferencial del péndulo desde el tiempo de impacto
            t_solve = np.linspace(0, t_post_impacto, max(2, int((i - (num_frames * tiempo_pre_impacto // tiempo_total_simulacion)))))
            sol = odeint(pendulo_eq, y0_pendulo, t_solve, args=(largo_pendulo_vis,))

            current_theta = sol[-1, 0]

            # Posiciones del péndulo (bala + caja) en función del ángulo y largo
            pos_x_current = largo_pendulo_vis * np.sin(current_theta)
            pos_y_current = -largo_pendulo_vis * np.cos(current_theta)

            pos_bala_x[i] = pos_x_current # Bala y caja se mueven juntos
            pos_bala_y[i] = pos_y_current

            pos_caja_x[i] = pos_x_current
            pos_caja_y[i] = pos_y_current

    # --- Creación del Gráfico Plotly ---
    fig = go.Figure(
        data=[
            # Bala (solo visible antes del impacto y luego 'fusionada' con la caja)
            go.Scatter(x=[pos_bala_x[0]], y=[pos_bala_y[0]], mode='markers',
                       marker=dict(size=masa_bala*200 + 5, color='orange', symbol='circle'),
                       name='Bala',
                       showlegend=True),
            # Caja / Sistema (bala+caja después del impacto)
            go.Scatter(x=[pos_caja_x[0]], y=[pos_caja_y[0]], mode='markers',
                       marker=dict(size=masa_caja*30 + 30, color='brown', symbol='square'), # Simbolo de cuadrado para caja
                       name='Caja / Sistema',
                       showlegend=True),
            # Cuerda del péndulo (desde el pivote 0,0 al centro de la caja)
            go.Scatter(x=[0, pos_caja_x[0]], y=[0, pos_caja_y[0]], mode='lines',
                       line=dict(color='black', width=2),
                       name='Cuerda',
                       showlegend=False)
        ],
        layout=go.Layout(
            xaxis=dict(range=[-largo_pendulo_vis * 2, largo_pendulo_vis * 2], zeroline=True, title="Posición X (m)"),
            yaxis=dict(range=[-largo_pendulo_vis * 1.5, 0.5], zeroline=True, title="Posición Y (m)",
                       scaleanchor="x", scaleratio=1), # Corrección del aspect ratio
            title='Animación de Péndulo Balístico',
            updatemenus=[dict(type='buttons',
                              buttons=[dict(label='Play',
                                            method='animate',
                                            args=[None, {'frame': {'duration': 50, 'redraw': True},
                                                         'fromcurrent': True,
                                                         'mode': 'immediate'}])],
                              x=0.05, y=1.05, xanchor='left', yanchor='bottom' # Posición del botón Play
                            )
                        ],
            showlegend=True,
            plot_bgcolor='rgba(240,240,240,1)',
            paper_bgcolor='rgba(255,255,255,1)',
        ),
        frames=[go.Frame(
            data=[
                # Datos de la bala (se 'esconde' después del impacto)
                go.Scatter(x=[pos_bala_x[k]] if t_values[k] <= tiempo_pre_impacto + 0.01 else [pos_caja_x[k]], # Fusiona la bala con la caja visualmente
                           y=[pos_bala_y[k]] if t_values[k] <= tiempo_pre_impacto + 0.01 else [pos_caja_y[k]],
                           mode='markers',
                           marker=dict(size=masa_bala*200 + 5, color='orange', symbol='circle')
                          ),
                # Datos de la caja/sistema
                go.Scatter(x=[pos_caja_x[k]], y=[pos_caja_y[k]], mode='markers',
                           marker=dict(size=masa_caja*30 + 30, color='brown', symbol='square')
                          ),
                # Datos de la cuerda
                go.Scatter(x=[0, pos_caja_x[k]], y=[0, pos_caja_y[k]], mode='lines',
                           line=dict(color='black', width=2))
            ],
            name=str(k)
        ) for k in range(num_frames)]
    )

    return fig

def plot_flecha_saco_animacion(m_flecha, v_flecha_inicial, m_saco, mu_k):
    """Genera una animación de la flecha incrustándose en el saco y moviéndose."""
    v_sistema_inicial = (m_flecha * v_flecha_inicial) / (m_flecha + m_saco)
    m_total = m_flecha + m_saco
    F_friccion = mu_k * m_total * g
    a_friccion = -F_friccion / m_total if m_total > 0 else 0
    
    if a_friccion == 0:
        tiempo_detencion = 5 # Tiempo arbitrario si no hay fricción (para que se mueva)
        distancia_detencion = v_sistema_inicial * tiempo_detencion
    else:
        tiempo_detencion = abs(v_sistema_inicial / a_friccion)
        # Asegurar que la distancia sea calculada correctamente, incluso si el tiempo_detencion es 0
        distancia_detencion = v_sistema_inicial * tiempo_detencion + 0.5 * a_friccion * tiempo_detencion**2


    num_frames = 150
    tiempo_animacion = np.linspace(0, tiempo_detencion * 1.2, num_frames) # Un poco más para ver la detención

    saco_width = 1.5
    saco_height = 1.0
    flecha_length = 0.8
    flecha_height = 0.1

    frames = []
    for t in tiempo_animacion:
        # Fases del movimiento: bala se mueve, luego impacta y se mueven juntos
        # Usamos un tiempo_impacto_visual para la transición de la animación
        tiempo_impacto_visual = tiempo_animacion[-1] * 0.2 # Impacto visual a 20% del tiempo total de animación

        if t < tiempo_impacto_visual and v_flecha_inicial > 0:
            # Bala moviéndose hacia el saco
            x_flecha = -flecha_length * 2 + v_flecha_inicial * t * (flecha_length * 2) / (tiempo_impacto_visual * v_flecha_inicial) # Ajuste para que llegue al saco
            x_saco = 0 # Saco estático
        else:
            # Sistema flecha-saco moviéndose juntos después del impacto
            tiempo_post_impacto = max(0, t - tiempo_impacto_visual)
            x_sistema = v_sistema_inicial * tiempo_post_impacto + 0.5 * a_friccion * tiempo_post_impacto**2
            x_flecha = x_sistema - flecha_length / 2 # La flecha está incrustada
            x_saco = x_sistema

        frame_data = [
            go.Scatter(x=[x_flecha + flecha_length / 2], y=[saco_height/2 + flecha_height/2], # Posición de la flecha
                       mode='markers', marker=dict(size=m_flecha*300 + 10, color='gray', symbol='arrow-right')),
            go.Scatter(x=[x_saco + saco_width / 2], y=[saco_height / 2], # Posición del saco
                       mode='markers', marker=dict(size=m_saco*50 + 50, color='tan', symbol='square'))
        ]
        frames.append(go.Frame(data=frame_data, name=f'{t:.2f}'))

    fig = go.Figure(
        data=[
            go.Scatter(x=[-flecha_length/2], y=[saco_height/2 + flecha_height/2], # Posición inicial de la flecha
                       mode='markers', marker=dict(size=m_flecha*300 + 10, color='gray', symbol='arrow-right')),
            go.Scatter(x=[saco_width/2], y=[saco_height/2], # Posición inicial del saco
                       mode='markers', marker=dict(size=m_saco*50 + 50, color='tan', symbol='square'))
        ],
        layout=go.Layout(
            xaxis=dict(range=[-flecha_length * 2 -1, distancia_detencion + saco_width + 1], autorange=False, title="Posición X (m)"),
            yaxis=dict(range=[-0.5, saco_height + 0.5], autorange=False, showgrid=False, zeroline=True, showticklabels=False,
                       scaleanchor="x", scaleratio=1), # Corrección del aspect ratio
            title='Animación: Flecha se Incrusta en Saco',
            updatemenus=[dict(type='buttons',
                              buttons=[dict(label='Play',
                                            method='animate',
                                            args=[None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate'}])])],
            shapes=[
                # Saco estático de fondo
                dict(type="rect", x0=0, y0=0, x1=saco_width, y1=saco_height,
                     fillcolor="tan", opacity=0.8, line_width=0, layer="below")
            ],
            plot_bgcolor='rgba(240,240,240,1)',
            paper_bgcolor='rgba(255,255,255,1)',
        ),
        frames=frames
    )
    return fig

def plot_caida_plano_impacto_animacion(m_obj, altura_inicial, angulo_plano_deg, mu_k_plano, e_impacto):
    """Animación de la caída por un plano inclinado y el rebote, mostrando la trayectoria completa."""
    angulo_plano_rad = np.deg2rad(angulo_plano_deg)
    longitud_plano = altura_inicial / np.sin(angulo_plano_rad)

    g_paralelo = g * np.sin(angulo_plano_rad)
    g_perpendicular = g * np.cos(angulo_plano_rad)
    F_normal = m_obj * g_perpendicular
    F_friccion_plano = mu_k_plano * F_normal
    a_plano = g_paralelo - (F_friccion_plano / m_obj)

    if a_plano <= 0:
        st.warning("El objeto no se moverá por el plano inclinado debido a la alta fricción.")
        return go.Figure() # No animation if no movement

    tiempo_bajada = np.sqrt(2 * longitud_plano / a_plano) if a_plano > 0 else 0
    v_final_plano = a_plano * tiempo_bajada

    vx_impacto = v_final_plano * np.cos(angulo_plano_rad)
    vy_impacto = -v_final_plano * np.sin(angulo_plano_rad)
    vy_rebote = -e_impacto * vy_impacto
    tiempo_vuelo_rebote = (2 * vy_rebote) / g if g > 0 else 0
    altura_max_rebote = (vy_rebote**2) / (2 * g)
    distancia_horizontal_rebote = vx_impacto * tiempo_vuelo_rebote

    num_frames = 200
    tiempo_total = tiempo_bajada + tiempo_vuelo_rebote
    t_values = np.linspace(0, tiempo_total * 1.1, num_frames) # Un poco más para ver el final

    x_trajectory = []
    y_trajectory = []
    frames = []

    # Puntos de la trayectoria estática para referencia del plano y el suelo
    x_plano_end_static = longitud_plano * np.cos(angulo_plano_rad)

    for i, t in enumerate(t_values):
        current_x = 0
        current_y = 0
        
        if t <= tiempo_bajada and tiempo_bajada > 0:
            # Bajada por el plano
            s_plano = 0.5 * a_plano * t**2
            current_x = s_plano * np.cos(angulo_plano_rad)
            current_y = altura_inicial - s_plano * np.sin(angulo_plano_rad)
        else:
            # Rebote parabólico
            t_rebote = t - tiempo_bajada
            if t_rebote >= 0 and tiempo_vuelo_rebote > 0:
                current_x = x_plano_end_static + vx_impacto * t_rebote
                current_y = 0 + vy_rebote * t_rebote - 0.5 * g * t_rebote**2
                # Asegurarse de que el objeto no vaya por debajo del suelo si es una colisión inelástica (e=0)
                if e_impacto == 0:
                    current_y = max(0, current_y)
            else:
                # Si no hay rebote o tiempo fuera de rango, permanece en el punto de impacto en el suelo
                current_x = x_plano_end_static + distancia_horizontal_rebote
                current_y = 0
        
        x_trajectory.append(current_x)
        y_trajectory.append(current_y)

        frame_data = [
            go.Scatter(x=[current_x], y=[current_y],
                       mode='markers', marker=dict(size=m_obj*10 + 10, color='blue', symbol='circle'))
        ]
        frames.append(go.Frame(data=frame_data, name=f'{t:.2f}'))

    fig = go.Figure(
        data=[
            # Traza para la trayectoria completa (estática, de fondo)
            go.Scatter(x=x_trajectory, y=y_trajectory,
                       mode='lines', line=dict(color='orange', width=2, dash='dot'),
                       name='Trayectoria Completa', showlegend=True), # Muestra en leyenda
            # Objeto inicial (posición del primer frame)
            go.Scatter(x=[x_trajectory[0]], y=[y_trajectory[0]],
                       mode='markers', marker=dict(size=m_obj*10 + 10, color='blue', symbol='circle'),
                       name='Objeto Móvil') # Objeto animado (se actualiza en frames)
        ],
        layout=go.Layout(
            xaxis=dict(range=[-0.5, max(x_plano_end_static, x_plano_end_static + distancia_horizontal_rebote) + 1], autorange=False, title="Posición X (m)"),
            yaxis=dict(range=[-0.5, altura_inicial * 1.2], autorange=False, title="Posición Y (m)", # Ajuste para asegurar que el suelo esté visible
                       scaleanchor="x", scaleratio=1),
            title='Animación: Caída y Rebote en Plano Inclinado',
            updatemenus=[dict(type='buttons',
                              buttons=[dict(label='Play',
                                            method='animate',
                                            args=[None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate'}])])],
            shapes=[
                # Plano inclinado estático de fondo
                dict(type="line", x0=0, y0=altura_inicial, x1=x_plano_end_static, y1=0,
                     line=dict(color="light green", width=3)),
                # Suelo estático de fondo (se extiende hasta el final de la trayectoria rebotada)
                dict(type="line", x0=0, y0=0, x1=x_plano_end_static + distancia_horizontal_rebote + 1, y1=0,
                     line=dict(color="red", width=3, dash='solid')) # Asegura que sea una línea sólida para el piso
            ],
            plot_bgcolor='rgba(240,240,240,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            showlegend=True # Asegura que la leyenda de la trayectoria se muestre
        ),
        frames=frames
    )
    return fig

# -------------------- Interacción entre Simulaciones (Ejemplo Conceptual) --------------------
def interaccion_flecha_saco_impacto():
    st.sidebar.subheader("Interacción: Flecha en Saco + Caída")
    st.sidebar.write("Simula la flecha incrustándose en un saco, y luego el saco cae por un plano inclinado.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Parámetros de la Flecha y Saco")
    m_flecha_int = st.sidebar.slider("Masa de la flecha (kg)", 0.01, 0.5, 0.1, key='m_flecha_int')
    v_flecha_inicial_int = st.sidebar.slider("Velocidad inicial de la flecha (m/s)", 20.0, 100.0, 50.0, key='v_flecha_int')
    m_saco_int = st.sidebar.slider("Masa del saco (kg)", 1.0, 20.0, 5.0, key='m_saco_int')
    mu_k_int = st.sidebar.slider("Coeficiente de fricción en el suelo (saco)", 0.0, 1.0, 0.3, key='mu_k_int')

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Parámetros del Plano Inclinado")
    angulo_plano_deg_int = st.sidebar.slider("Ángulo del plano inclinado (grados)", 5, 80, 30, key='angulo_int')
    altura_inicial_int = st.sidebar.slider("Altura inicial del plano (m)", 0.1, 5.0, 2.0, key='altura_int')
    e_impacto_int = st.sidebar.slider("Coeficiente de restitución (impacto con suelo)", 0.0, 1.0, 0.7, key='e_int')
    mu_k_plano_int = st.sidebar.slider("Coeficiente de fricción en el plano", 0.0, 1.0, 0.2, key='mu_k_plano_int')

    if st.sidebar.button("Simular Interacción"):
        st.subheader("Simulación Integrada: Flecha en Saco y Caída por Plano Inclinado")

        # Fase 1: Flecha en Saco
        st.markdown("#### Fase 1: Flecha se incrusta en el Saco")
        v_sistema_inicial, F_friccion, distancia_detencion = simular_flecha_saco(
            m_flecha_int, v_flecha_inicial_int, m_saco_int, mu_k_int
        )
        st.write(f"Velocidad inicial del sistema flecha+saco: **{v_sistema_inicial:.2f} m/s**")
        st.write(f"Distancia que el saco se desplaza hasta detenerse: **{distancia_detencion:.2f} m**")

        st.plotly_chart(plot_flecha_saco_animacion(m_flecha_int, v_flecha_inicial_int, m_saco_int, mu_k_int), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Fase 2: El Saco (ahora con la flecha) cae por un Plano Inclinado y Rebota")
        st.info("Para esta simulación integrada, la velocidad del saco se *reseteará* al inicio del plano.")
        st.info("El saco ahora se considera como una única masa: "
                f"**{(m_flecha_int + m_saco_int):.2f} kg**.")

        masa_saco_con_flecha = m_flecha_int + m_saco_int

        a_plano, v_final_plano, vx_impacto, vy_impacto, \
        vy_rebote, altura_max_rebote, distancia_horizontal_rebote = simular_caida_plano_impacto(
            masa_saco_con_flecha, altura_inicial_int, angulo_plano_deg_int, mu_k_plano_int, e_impacto_int
        )

        st.write(f"Aceleración en el plano: **{a_plano:.2f} m/s²**")
        st.write(f"Velocidad al final del plano: **{v_final_plano:.2f} m/s**")
        st.write(f"Altura máxima después del rebote: **{altura_max_rebote:.2f} m**")
        st.write(f"Distancia horizontal del rebote: **{distancia_horizontal_rebote:.2f} m**")

        st.plotly_chart(plot_caida_plano_impacto_animacion(m_obj=masa_saco_con_flecha, altura_inicial=altura_inicial_int, angulo_plano_deg=angulo_plano_deg_int, mu_k_plano=mu_k_plano_int, e_impacto=e_impacto_int), use_container_width=True)
        st.markdown("---")

# -------------------- Aplicación Principal Streamlit --------------------

st.sidebar.title("Menú de Simulaciones")
simulation_type = st.sidebar.radio(
    "Selecciona una opción:",
    
    ("Fundamentos Teóricos",
     "Simulación de Colisión 1D",
     "Simulación de Colisión 2D",
     "Cálculo de Impulso y Fuerza Promedio",
     "Péndulo Balístico",
     "Flecha que se Incrusta en un Saco",
     "Caída por Plano Inclinado + Impacto",
     "Interacción de Simulaciones: Flecha-Saco y Caída")
)

st.sidebar.markdown("---")
st.sidebar.info("¡Experimenta con los parámetros para comprender mejor los conceptos físicos!")

# -------------------- Contenido Principal de la Aplicación --------------------

if simulation_type == "Fundamentos Teóricos":
    st.header("📚 Fundamentos Teóricos de Impulso y Cantidad de Movimiento Lineal")
    st.markdown("""
        La **cantidad de movimiento lineal** (o momento lineal) es una propiedad fundamental de los objetos en movimiento.
        Se define como el producto de la **masa** de un objeto y su **velocidad**. Es una **cantidad vectorial**, lo que
        significa que tiene magnitud y dirección.

        ---

        ### Definiciones Clave:

        * Cantidad de Movimiento Lineal ($\\vec{P}$):
           $\\vec{P}$ = m $\\vec{v}$ 
            * $m$ = masa del objeto (kg)
            * $\\vec{v}$ = velocidad del objeto (m/s)
            * Unidades: kg·m/s

        * **Impulso ($\\vec{J}$):**
            Representa el cambio en la cantidad de movimiento de un objeto. También puede verse como
            la fuerza neta aplicada sobre un objeto durante un intervalo de tiempo.
            $$ \\vec{J} = \\Delta \\vec{p} = \\vec{p}_{final} - \\vec{p}_{inicial} $$
            $$ \\vec{J} = \\vec{F}_{promedio} \\Delta t $$
            Donde:
            * $\\Delta \\vec{p}$ = cambio en la cantidad de movimiento
            * $\\vec{F}_{promedio}$ = fuerza promedio neta aplicada (N)
            * $\\Delta t$ = intervalo de tiempo (s)
            * Unidades: N·s (que es equivalente a kg·m/s)

        * **Teorema del Impulso y la Cantidad de Movimiento:**
            Establece que el impulso aplicado a un objeto es igual al cambio en su cantidad de movimiento.
            Este teorema es crucial para analizar colisiones e impactos donde las fuerzas son grandes y actúan por poco tiempo.

        ---

        ### Colisiones y Conservación del Momento:

        En un **sistema aislado** (donde no actúan fuerzas externas netas), la cantidad de movimiento lineal total del sistema
        permanece constante. Esto es conocido como la **Ley de Conservación de la Cantidad de Movimiento Lineal**.

        $$ \\vec{p}_{total, inicial} = \\vec{p}_{total, final} $$
        Esto es particularmente útil para analizar **colisiones**, ya que la cantidad de movimiento total antes de la colisión es igual
        a la cantidad de movimiento total después de la colisión.

        * **Colisiones Elásticas ($e=1$):**
            Tanto la cantidad de movimiento lineal como la **energía cinética total** del sistema se conservan. Los objetos "rebotan" perfectamente.

        * **Colisiones Inelásticas ($e=0$):**
            La cantidad de movimiento lineal se conserva, pero la energía cinética total **no se conserva** (parte de la energía se transforma en calor, sonido, deformación, etc.).
            En una **colisión perfectamente inelástica**, los objetos se pegan y se mueven como uno solo después del impacto.

        * **Coeficiente de Restitución ($e$):**
            Es una medida de la "elasticidad" de una colisión entre dos objetos. Se define como la razón de la velocidad relativa de separación
            a la velocidad relativa de aproximación.
            $$ e = - \\frac{(\\vec{v}_{2,final} - \\vec{v}_{1,final})}{(\\vec{v}_{2,inicial} - \\vec{v}_{1,inicial})} $$
            * $e = 1$ para colisiones perfectamente elásticas.
            * $e = 0$ para colisiones perfectamente inelásticas.
            * $0 < e < 1$ para colisiones inelásticas.

        ---

        ### **Ejemplos de Aplicación:**
        Veremos cómo estos principios se aplican en simulaciones de colisiones, péndulos balísticos, y más.
    """)

elif simulation_type == "Simulación de Colisión 1D":
    st.header("💥 Simulación de Colisión 1D")
    st.markdown("""
        Esta simulación te permite observar cómo interactúan dos objetos moviéndose en una línea recta.
        Puedes ajustar sus **masas**, **velocidades iniciales** y el **tipo de colisión** (elástica, inelástica o parcial)
        para ver cómo cambian sus velocidades después del impacto.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Objeto 1")
        m1 = st.slider("Masa del Objeto 1 (kg)", 0.1, 10.0, 1.0, 0.1, key='m1_1d')
        v1_inicial = st.slider("Velocidad Inicial del Objeto 1 (m/s)", -10.0, 10.0, 5.0, 0.1, key='v1_1d')

    with col2:
        st.subheader("Objeto 2")
        m2 = st.slider("Masa del Objeto 2 (kg)", 0.1, 10.0, 2.0, 0.1, key='m2_1d')
        v2_inicial = st.slider("Velocidad Inicial del Objeto 2 (m/s)", -10.0, 10.0, 0.0, 0.1, key='v2_1d')

    tipo_colision = st.selectbox(
        "Tipo de Colisión",
        ["Elástica ($e=1$)", "Inelástica Perfecta ($e=0$)", "Parcialmente Inelástica ($0 < e < 1$)"],
        key='tipo_1d'
    )

    e = 0.0
    if tipo_colision == "Elástica ($e=1$)":
        e = 1.0
    elif tipo_colision == "Inelástica Perfecta ($e=0$)":
        e = 0.0
    else:
        e = st.slider("Coeficiente de Restitución ($e$)", 0.0, 1.0, 0.7, 0.01, key='e_1d')

    if st.button("Simular Colisión 1D", key='btn_sim_1d'):
        v1_final, v2_final = simular_colision_1d(m1, v1_inicial, m2, v2_inicial, e)

        st.subheader("Resultados de la Colisión")
        st.write(f"**Velocidad final del Objeto 1:** `{v1_final:.2f} m/s`")
        st.write(f"**Velocidad final del Objeto 2:** `{v2_final:.2f} m/s`")

        # Cálculo y comparación de momento y energía
        momento_inicial_total = m1 * v1_inicial + m2 * v2_inicial
        momento_final_total = m1 * v1_final + m2 * v2_final
        st.markdown(f"**Momento Total Inicial:** `{momento_inicial_total:.2f} kg·m/s`")
        st.markdown(f"**Momento Total Final:** `{momento_final_total:.2f} kg·m/s`")
        st.info("La **cantidad de movimiento total** se **conserva** en todas las colisiones, independientemente de la elasticidad.")

        energia_cinetica_inicial = 0.5 * m1 * v1_inicial**2 + 0.5 * m2 * v2_inicial**2
        energia_cinetica_final = 0.5 * m1 * v1_final**2 + 0.5 * m2 * v2_final**2
        st.markdown(f"**Energía Cinética Total Inicial:** `{energia_cinetica_inicial:.2f} J`")
        st.markdown(f"**Energía Cinética Total Final:** `{energia_cinetica_final:.2f} J`")

        if tipo_colision == "Elástica ($e=1$)":
            st.success("En colisiones elásticas, la **energía cinética total también se conserva**.")
        else:
            perdida_energia = energia_cinetica_inicial - energia_cinetica_final
            st.warning(f"En colisiones inelásticas, se **pierde energía cinética**. Pérdida: `{perdida_energia:.2f} J`")

        st.subheader("Visualización de la Colisión 1D")
        st.plotly_chart(plot_colision_1d_animacion(m1, v1_inicial, m2, v2_inicial, e), use_container_width=True)
        st.caption("Los tamaños de los círculos son proporcionales a la raíz cúbica de sus masas para una mejor visualización.")

elif simulation_type == "Simulación de Colisión 2D":
    st.header("💥 Simulación de Colisión 2D con Trayectorias")
    st.markdown("""
        Explora colisiones donde los objetos se mueven en un plano (dos dimensiones).
        Puedes ajustar las **masas**, las **componentes de velocidad inicial (x, y)** y el **tipo de colisión**.
        La visualización mostrará las trayectorias antes y después del impacto.
        **Nota:** Para simplificar, asumimos un "ángulo de impacto" que define la línea a lo largo de la cual ocurre la interacción principal.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Objeto 1")
        m1_2d = st.slider("Masa del Objeto 1 (kg)", 0.1, 10.0, 1.0, key='m1_2d')
        v1_ix = st.slider("Velocidad inicial X del Objeto 1 (m/s)", -10.0, 10.0, 3.0, key='v1x_2d')
        v1_iy = st.slider("Velocidad inicial Y del Objeto 1 (m/s)", -10.0, 10.0, 2.0, key='v1y_2d')

    with col2:
        st.subheader("Objeto 2")
        m2_2d = st.slider("Masa del Objeto 2 (kg)", 0.1, 10.0, 2.0, key='m2_2d')
        v2_ix = st.slider("Velocidad inicial X del Objeto 2 (m/s)", -10.0, 10.0, -2.0, key='v2x_2d')
        v2_iy = st.slider("Velocidad inicial Y del Objeto 2 (m/s)", -10.0, 10.0, 0.0, key='v2y_2d')

    tipo_colision_2d = st.selectbox(
        "Tipo de Colisión 2D",
        ["Elástica ($e=1$)", "Inelástica Perfecta ($e=0$)", "Parcialmente Inelástica ($0 < e < 1$)"],
        key='tipo_2d'
    )
    e_2d = 0.0
    if tipo_colision_2d == "Elástica ($e=1$)":
        e_2d = 1.0
    elif tipo_colision_2d == "Inelástica Perfecta ($e=0$)":
        e_2d = 0.0
    else:
        e_2d = st.slider("Coeficiente de Restitución ($e$)", 0.0, 1.0, 0.8, 0.01, key='e_2d')

    angulo_impacto_deg = st.slider("Ángulo de Impacto (grados)", 0, 180, 45, key='angulo_impacto_2d')
    st.info("El 'ángulo de impacto' simula la línea a lo largo de la cual las fuerzas de colisión actúan. Para colisiones entre esferas, sería la línea que une sus centros en el momento del contacto.")

    if st.button("Simular Colisión 2D", key='btn_sim_2d'):
        (v1fx, v1fy), (v2fx, v2fy) = simular_colision_2d(m1_2d, v1_ix, v1_iy, m2_2d, v2_ix, v2_iy, e_2d, angulo_impacto_deg)

        st.subheader("Resultados de la Colisión 2D")
        st.write(f"**Velocidad final del Objeto 1:** `({v1fx:.2f} m/s, {v1fy:.2f} m/s)`")
        st.write(f"  Magnitud: `{np.sqrt(v1fx**2 + v1fy**2):.2f} m/s`")
        st.write(f"**Velocidad final del Objeto 2:** `({v2fx:.2f} m/s, {v2fy:.2f} m/s)`")
        st.write(f"  Magnitud: `{np.sqrt(v2fx**2 + v2fy**2):.2f} m/s`")

        # Comparación de Cantidad de Movimiento y Energía
        p1_inicial = np.array([m1_2d * v1_ix, m1_2d * v1_iy])
        p2_inicial = np.array([m2_2d * v2_ix, m2_2d * v2_iy])
        p_total_inicial = p1_inicial + p2_inicial

        p1_final = np.array([m1_2d * v1fx, m1_2d * v1fy])
        p2_final = np.array([m2_2d * v2fx, m2_2d * v2fy])
        p_total_final = p1_final + p2_final

        st.markdown(f"**Momento Total Inicial:** `{p_total_inicial[0]:.2f}i + {p_total_inicial[1]:.2f}j kg·m/s` (Magnitud: `{np.linalg.norm(p_total_inicial):.2f}`)")
        st.markdown(f"**Momento Total Final:** `{p_total_final[0]:.2f}i + {p_total_final[1]:.2f}j kg·m/s` (Magnitud: `{np.linalg.norm(p_total_final):.2f}`)")
        st.info("La **cantidad de movimiento total se conserva** en las colisiones 2D (vectorialmente).")

        ec_inicial = 0.5 * m1_2d * (v1_ix**2 + v1_iy**2) + 0.5 * m2_2d * (v2_ix**2 + v2_iy**2)
        ec_final = 0.5 * m1_2d * (v1fx**2 + v1fy**2) + 0.5 * m2_2d * (v2fx**2 + v2fy**2)
        st.markdown(f"**Energía Cinética Total Inicial:** `{ec_inicial:.2f} J`")
        st.markdown(f"**Energía Cinética Total Final:** `{ec_final:.2f} J`")
        if e_2d == 1.0:
            st.success("En colisiones elásticas 2D, la **energía cinética total también se conserva**.")
        else:
            perdida_energia = energia_cinetica_inicial - energia_cinetica_final # type: ignore
            st.warning(f"En colisiones inelásticas 2D, hay **pérdida de energía cinética**. Pérdida: `{perdida_energia:.2f} J`")

        st.subheader("Visualización de Trayectorias")
        st.plotly_chart(plot_colision_2d_trayectorias(m1_2d, v1_ix, v1_iy, m2_2d, v2_ix, v2_iy, e_2d, angulo_impacto_deg), use_container_width=True)
        st.caption("Las líneas punteadas muestran las trayectorias antes de la colisión; las líneas sólidas, después. Los círculos indican la posición de impacto.")

elif simulation_type == "Cálculo de Impulso y Fuerza Promedio":
    st.header("🧮 Cálculo de Impulso y Fuerza Promedio")
    st.markdown("""
        Esta sección te permite calcular el **impulso** a partir de una fuerza promedio y un tiempo,
        o la **fuerza promedio** a partir de un impulso y un tiempo.
        Esto es útil para entender cómo las fuerzas actúan para cambiar la cantidad de movimiento de un objeto.
    """)

    st.subheader("Selecciona qué deseas calcular:")
    opcion_calculo = st.radio(
        "Opción de Cálculo",
        ("Calcular Impulso", "Calcular Fuerza Promedio"),
        key='calc_option'
    )

    if opcion_calculo == "Calcular Impulso":
        fuerza = st.number_input("Fuerza promedio aplicada (N)", min_value=0.0, value=10.0, step=0.1, key='fuerza_imp')
        tiempo = st.number_input("Tiempo de aplicación (s)", min_value=0.01, value=1.0, step=0.01, key='tiempo_imp')
        if st.button("Calcular Impulso", key='btn_calc_imp'):
            impulso, mensaje = calcular_impulso_fuerza("impulso", fuerza, tiempo)
            st.success(mensaje)
            st.info(f"Esto significa que el objeto experimentó un cambio en su cantidad de movimiento de `{impulso:.2f} kg·m/s`.")

            st.markdown("---")
            st.subheader("Visualización del Impulso")
            fig = go.Figure(data=go.Bar(x=['Fuerza (N)', 'Tiempo (s)', 'Impulso (Ns)'],
                                       y=[fuerza, tiempo, impulso],
                                       marker_color=['lightblue', 'lightgreen', 'gold']))
            fig.update_layout(title='Relación entre Fuerza, Tiempo e Impulso',
                              yaxis_title='Valor',
                              height=400)
            st.plotly_chart(fig, use_container_width=True)

    elif opcion_calculo == "Calcular Fuerza Promedio":
        impulso = st.number_input("Impulso (Ns)", min_value=0.0, value=10.0, step=0.1, key='impulso_fuerza')
        tiempo = st.number_input("Tiempo de aplicación (s)", min_value=0.01, value=1.0, step=0.01, key='tiempo_fuerza')
        if st.button("Calcular Fuerza Promedio", key='btn_calc_fuerza'):
            fuerza, mensaje = calcular_impulso_fuerza("fuerza_promedio", impulso, tiempo)
            st.success(mensaje)
            st.info(f"Para lograr un impulso de `{impulso:.2f} Ns` en `{tiempo:.2f} s`, se necesita una fuerza promedio de `{fuerza:.2f} N`.")

            st.markdown("---")
            st.subheader("Visualización de la Fuerza Promedio")
            fig = go.Figure(data=go.Bar(x=['Impulso (Ns)', 'Tiempo (s)', 'Fuerza Promedio (N)'],
                                       y=[impulso, tiempo, fuerza],
                                       marker_color=['gold', 'lightgreen', 'lightblue']))
            fig.update_layout(title='Relación entre Impulso, Tiempo y Fuerza Promedio',
                              yaxis_title='Valor',
                              height=400)
            st.plotly_chart(fig, use_container_width=True)

elif simulation_type == "Péndulo Balístico":
    st.header("🎯 Simulación del Péndulo Balístico")
    st.markdown("""
        Observa la **animación** completa del péndulo balístico: la bala se aproxima, impacta la caja,
        y el sistema combinado oscila hacia arriba. Esta demostración ilustra la **conservación
        del momento lineal** durante el impacto y la **conservación de la energía mecánica** en el ascenso del péndulo.
    """)

    col1, col2 = st.columns(2)
    with col1:
        masa_bala = st.slider("Masa de la Bala (kg)", 0.001, 0.1, 0.01, 0.001, key='m_bala')
        velocidad_bala_inicial = st.slider("Velocidad Inicial de la Bala (m/s)", 10.0, 1000.0, 300.0, 1.0, key='v_bala')
    with col2:
        masa_caja = st.slider("Masa de la Caja (kg)", 0.1, 10.0, 1.0, 0.1, key='m_caja') # CAMBIO: masa_saco -> masa_caja
        largo_pendulo_vis = st.slider("Largo del Péndulo (m)", 0.5, 5.0, 2.0, 0.1, key='largo_pendulo_vis')

    st.markdown(f"**Gravedad ($g$):** `{g:.2f} m/s²`")

    if st.button("Simular Péndulo Balístico", key='btn_pendulo'):
        v_sistema = calcular_v_sistema_pendulo(masa_caja, masa_bala, velocidad_bala_inicial) # CAMBIO: masa_saco -> masa_caja
        h_max = calcular_h_max_pendulo(masa_caja, masa_bala, velocidad_bala_inicial) # CAMBIO: masa_saco -> masa_caja

        st.subheader("Resultados Teóricos:")
        st.write(f"**Velocidad del sistema (bala+caja) justo después del impacto:** `{v_sistema:.2f} m/s`") # CAMBIO: saco -> caja
        st.write(f"**Altura máxima alcanzada por el centro de masa del sistema:** `{h_max:.2f} m`")
        if h_max > largo_pendulo_vis:
            st.warning(f"La altura calculada ({h_max:.2f} m) excede la longitud visual del péndulo ({largo_pendulo_vis:.2f} m). Esto es un resultado teórico; en la práctica, el péndulo podría dar una vuelta completa o el modelo sería limitado.")

        st.subheader("Animación de la Simulación:")
        st.plotly_chart(plot_pendulo_balistico_animacion(masa_bala, masa_caja, velocidad_bala_inicial, largo_pendulo_vis), use_container_width=True) # CAMBIO: masa_saco -> masa_caja
        st.caption("Pulsa 'Play' en el gráfico para iniciar la animación. El tamaño de los marcadores es relativo a la masa.")

        st.subheader("Explicación Física")
        st.markdown("""
            1.  **Colisión Inelástica (Conservación del Momento Lineal):**
                Antes del impacto, la bala tiene una cantidad de movimiento y la caja está en reposo. Cuando la bala se incrusta, forman un solo sistema (bala+caja). La cantidad de movimiento total del sistema se conserva justo antes y después del impacto:
                $$ m_{bala} \\cdot v_{bala, inicial} = (m_{bala} + m_{caja}) \\cdot v_{sistema} $$
                De esta ecuación se obtiene la velocidad inicial ($v_{sistema}$) del sistema combinado.

            2.  **Ascenso del Péndulo (Conservación de la Energía Mecánica):**
                Después del impacto, el sistema (bala+caja) tiene energía cinética en su punto más bajo. A medida que oscila hacia arriba, esta energía cinética se transforma en energía potencial gravitacional. La energía mecánica total del sistema se conserva durante este ascenso:
                $$ \\frac{1}{2} (m_{bala} + m_{caja}) v_{sistema}^2 = (m_{bala} + m_{caja}) g h_{max} $$
                De esta ecuación se calcula la altura máxima ($h_{max}$) que alcanza el sistema.
        """)

elif simulation_type == "Flecha que se Incrusta en un Saco":
    st.header("🏹 Simulación de Flecha que se Incrusta en un Saco")
    st.markdown("""
        Esta simulación modela una **colisión perfectamente inelástica** donde una flecha se incrusta
        en un saco de arena en reposo. Después del impacto, el sistema combinado (flecha+saco)
        se mueve una distancia hasta detenerse debido a la **fuerza de fricción**.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Flecha")
        m_flecha = st.slider("Masa de la Flecha (kg)", 0.005, 0.2, 0.05, 0.005, key='m_flecha_saco')
        v_flecha_inicial = st.slider("Velocidad Inicial de la Flecha (m/s)", 10.0, 200.0, 70.0, 1.0, key='v_flecha_saco')
    with col2:
        st.subheader("Saco")
        m_saco = st.slider("Masa del Saco (kg)", 0.5, 20.0, 10.0, 0.5, key='m_saco_saco')
        mu_k = st.slider("Coeficiente de Fricción Cinética (μk)", 0.0, 1.0, 0.4, 0.01, key='mu_k_saco')

    st.markdown(f"**Gravedad ($g$):** `{g:.2f} m/s²`")

    if st.button("Simular Flecha en Saco", key='btn_flecha_saco'):
        v_sistema_inicial, F_friccion, distancia_detencion = simular_flecha_saco(
            m_flecha, v_flecha_inicial, m_saco, mu_k
        )

        st.subheader("Resultados de la Simulación")
        st.write(f"**Velocidad del sistema (flecha+saco) justo después del impacto:** `{v_sistema_inicial:.2f} m/s`")
        st.write(f"**Fuerza de fricción actuando sobre el saco:** `{F_friccion:.2f} N`")
        st.write(f"**Distancia que se desplaza el saco hasta detenerse:** `{distancia_detencion:.2f} m`")

        st.subheader("Animación del Fenómeno")
        st.plotly_chart(plot_flecha_saco_animacion(m_flecha, v_flecha_inicial, m_saco, mu_k), use_container_width=True)
        st.caption("Pulsa 'Play' para ver la flecha impactando el saco y el movimiento posterior.")

        st.subheader("Explicación Física")
        st.markdown("""
            1.  **Colisión (Momento Lineal):** La colisión entre la flecha y el saco es perfectamente inelástica (se pegan). La cantidad de movimiento se conserva:
                $$ m_{flecha} v_{flecha, inicial} = (m_{flecha} + m_{saco}) v_{sistema, inicial} $$
                Esto nos da la velocidad del sistema combinado justo después del impacto.

            2.  **Movimiento con Fricción (Leyes de Newton y Cinemática):** Una vez que la flecha se incrusta, el sistema flecha-saco se mueve con una velocidad inicial ($v_{sistema, inicial}$). La única fuerza horizontal que actúa para detenerlo es la fuerza de fricción cinética:
                $$ F_{friccion} = \\mu_k N = \\mu_k (m_{flecha} + m_{saco}) g $$
                Esta fuerza causa una aceleración negativa ($a = -F_{friccion} / m_{total}$). Usando las ecuaciones de cinemática ($v_f^2 = v_i^2 + 2ad$), podemos encontrar la distancia ($d$) que se desplaza hasta que $v_f = 0$.
        """)

elif simulation_type == "Caída por Plano Inclinado + Impacto":
    st.header("⛰️ Simulación de Caída por Plano Inclinado + Impacto")
    st.markdown("""
        Esta simulación analiza un objeto que se desliza por un plano inclinado y luego
        **impacta el suelo**, rebotando. Puedes ajustar las propiedades del plano
        y el **coeficiente de restitución** del impacto para ver cómo afectan la trayectoria.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Objeto")
        m_obj = st.slider("Masa del Objeto (kg)", 0.1, 5.0, 1.0, 0.1, key='m_obj_plano')
        altura_inicial = st.slider("Altura inicial del objeto (m)", 0.5, 10.0, 5.0, 0.1, key='h_plano')
        angulo_plano_deg = st.slider("Ángulo del Plano Inclinado (grados)", 10, 80, 45, key='angulo_plano')
    with col2:
        st.subheader("Condiciones")
        mu_k_plano = st.slider("Coeficiente de Fricción Cinética en el Plano", 0.0, 0.5, 0.1, 0.01, key='mu_k_plano')
        e_impacto = st.slider("Coeficiente de Restitución (Impacto con el Suelo)", 0.0, 1.0, 0.7, 0.01, key='e_impacto_plano')

    st.markdown(f"**Gravedad ($g$):** `{g:.2f} m/s²`")

    if st.button("Simular Caída e Impacto", key='btn_plano'):
        a_plano, v_final_plano, vx_impacto, vy_impacto, \
        vy_rebote, altura_max_rebote, distancia_horizontal_rebote = simular_caida_plano_impacto(
            m_obj, altura_inicial, angulo_plano_deg, mu_k_plano, e_impacto
        )

        st.subheader("Resultados de la Simulación")
        if a_plano <= 0:
            st.error("El objeto no se deslizará por el plano debido a la alta fricción. Reduzca el coeficiente de fricción o aumente el ángulo.")
        else:
            st.write(f"**Aceleración a lo largo del plano:** `{a_plano:.2f} m/s²`")
            st.write(f"**Velocidad del objeto al final del plano (antes del impacto):** `{v_final_plano:.2f} m/s`")
            st.write(f"**Componentes de velocidad justo antes del impacto:** `vx={vx_impacto:.2f} m/s, vy={vy_impacto:.2f} m/s`")
            st.write(f"**Velocidad vertical de rebote:** `{vy_rebote:.2f} m/s`")
            st.write(f"**Altura máxima alcanzada después del rebote:** `{altura_max_rebote:.2f} m`")
            st.write(f"**Distancia horizontal recorrida durante el rebote:** `{distancia_horizontal_rebote:.2f} m`")

            st.subheader("Animación del Fenómeno")
            st.plotly_chart(plot_caida_plano_impacto_animacion(m_obj, altura_inicial, angulo_plano_deg, mu_k_plano, e_impacto), use_container_width=True)
            st.caption("Pulsa 'Play' para ver el objeto caer y rebotar. La línea punteada muestra la trayectoria completa.")

            st.subheader("Explicación Física")
            st.markdown("""
                1.  **Movimiento en el Plano Inclinado:** El objeto acelera hacia abajo por el plano debido a la componente de la gravedad paralela al plano, oponiéndose a la fuerza de fricción.
                    * Fuerza neta a lo largo del plano: $F_{neta} = m g \\sin(\\theta) - \\mu_k m g \\cos(\\theta)$
                    * Aceleración: $a = g (\\sin(\\theta) - \\mu_k \\cos(\\theta))$
                    Luego, usamos $v^2 = u^2 + 2as$ para encontrar la velocidad al final del plano.

                2.  **Impacto con el Suelo:** La colisión con el suelo afecta principalmente la componente vertical de la velocidad. El coeficiente de restitución ($e$) determina qué tan "elástico" es el rebote:
                    $$ v_{y,rebote} = -e \\cdot v_{y,impacto} $$
                    La componente horizontal de la velocidad generalmente se conserva en un impacto con una superficie horizontal (asumiendo fricción insignificante durante el impacto).

                3.  **Movimiento Parabólico Post-Impacto:** Después del rebote, el objeto sigue una trayectoria parabólica, alcanzando una altura máxima antes de volver a caer. Se analiza como un problema de tiro parabólico.
            """)

elif simulation_type == "Interacción de Simulaciones: Flecha-Saco y Caída":
    interaccion_flecha_saco_impacto()

st.markdown("---")
st.markdown("Desarrollado porEl grupo E  para el proyecto de Física.")
