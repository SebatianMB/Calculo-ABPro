

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sympy import symbols, sin, cos, pi, simplify, integrate
import pandas as pd
import pyvista as pv
import plotly.graph_objects as go
from scipy.fftpack import fft

# Variables simbólicas para cálculos matemáticos
theta_sym, phi_sym = symbols('theta phi')

# Función simbólica para el patrón de radiación (dipolo)
radiation_eq = sin(theta_sym)**2
simplified_eq = simplify(radiation_eq)

print(f"Ecuación del patrón de radiación simplificada: {simplified_eq}")

# -----------------------------------------------------------
# Cálculo Integral
# -----------------------------------------------------------

def calculate_integral(pattern_function, theta_range=(0, pi)):
    """
    Calcula el área bajo el patrón de radiación en un rango dado de theta.
    :param pattern_function: Función del patrón de radiación
    :param theta_range: Rango de integración para theta
    :return: Área total bajo la curva
    """
    area = integrate(pattern_function, (theta_sym, theta_range[0], theta_range[1]))
    return area.evalf()

# Calcular el área bajo la curva del patrón de radiación (dipolo)
area_total = calculate_integral(radiation_eq)
print(f"Área total bajo el patrón de radiación (dipolo): {area_total}")

# -----------------------------------------------------------
# Funciones para Tipos de Antenas
# -----------------------------------------------------------

def radiation_pattern_dipole(theta):
    """Patrón de radiación para un dipolo."""
    return np.sin(theta)**2

def radiation_pattern_parabolic(theta):
    """Patrón de radiación para una antena parabólica (aproximado)."""
    return np.exp(-theta**2)

def radiation_pattern_yagi(theta):
    """Patrón de radiación para una antena Yagi (aproximado)."""
    return np.cos(theta)**2

# -----------------------------------------------------------
# Función Seleccionable para Patrones de Radiación
# -----------------------------------------------------------

def select_radiation_pattern(pattern_type):
    """
    Devuelve la función del patrón de radiación según el tipo seleccionado.
    :param pattern_type: Tipo de antena ('dipole', 'parabolic', 'yagi')
    :return: Función del patrón de radiación
    """
    if pattern_type == 'dipole':
        return radiation_pattern_dipole
    elif pattern_type == 'parabolic':
        return radiation_pattern_parabolic
    elif pattern_type == 'yagi':
        return radiation_pattern_yagi
    else:
        raise ValueError("Tipo de antena no soportado. Use 'dipole', 'parabolic' o 'yagi'.")

# -----------------------------------------------------------
# Visualización del Patrón de Radiación en 2D
# -----------------------------------------------------------

def plot_polar_radiation(pattern_type='dipole'):
    """
    Genera una gráfica polar del patrón de radiación.
    :param pattern_type: Tipo de antena ('dipole', 'parabolic', 'yagi')
    """
    theta = np.linspace(0, 2 * np.pi, 360)
    pattern_function = select_radiation_pattern(pattern_type)
    r = pattern_function(theta)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, r, label=f"Patrón de Radiación ({pattern_type})")
    ax.set_title(f"Patrón de Radiación en Coordenadas Polares ({pattern_type})")
    ax.legend()
    plt.show()

# -----------------------------------------------------------
# Simulaciones en 3D (PyVista)
# -----------------------------------------------------------

def plot_3d_radiation(pattern_type='dipole'):
    """
    Genera una visualización en 3D del patrón de radiación.
    :param pattern_type: Tipo de antena ('dipole', 'parabolic', 'yagi')
    """
    theta = np.linspace(0, np.pi, 180)
    phi = np.linspace(0, 2 * np.pi, 360)
    theta, phi = np.meshgrid(theta, phi)

    pattern_function = select_radiation_pattern(pattern_type)
    r = pattern_function(theta)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    grid = pv.StructuredGrid(x, y, z)
    grid.point_data["radiation"] = r.flatten()

    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars="radiation", cmap="viridis", show_edges=True)
    plotter.add_axes()
    plotter.add_title(f"Patrón de Radiación 3D ({pattern_type})")
    plotter.show()

# -----------------------------------------------------------
# Interactividad con Plotly
# -----------------------------------------------------------

def interactive_radiation(pattern_type='dipole'):
    """
    Genera una visualización interactiva del patrón de radiación.
    :param pattern_type: Tipo de antena ('dipole', 'parabolic', 'yagi')
    """
    theta = np.linspace(0, np.pi, 180)
    phi = np.linspace(0, 2 * np.pi, 360)
    theta, phi = np.meshgrid(theta, phi)

    pattern_function = select_radiation_pattern(pattern_type)
    r = pattern_function(theta)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
    fig.update_layout(
        title=f"Patrón de Radiación 3D Interactivo ({pattern_type})",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Intensidad'
        )
    )
    fig.show()

# -----------------------------------------------------------
# Ejecución de Simulaciones
# -----------------------------------------------------------

if __name__ == "__main__":
    try:
        print("Iniciando simulaciones...")

        # Tipo de antena seleccionada
        pattern_type = 'dipole'  # Cambiar a 'parabolic' o 'yagi' según sea necesario

        # Cálculo Integral
        print(f"Calculando el área total bajo el patrón de radiación ({pattern_type})...")
        pattern_function = select_radiation_pattern(pattern_type)
        area_total = quad(pattern_function, 0, np.pi)[0]
        print(f"Área total calculada: {area_total}")

        # Visualización en 2D
        print("Generando visualización en 2D...")
        plot_polar_radiation(pattern_type)

        # Visualización en 3D
        print("Generando visualización en 3D...")
        plot_3d_radiation(pattern_type)

        # Visualización interactiva
        print("Generando visualización interactiva...")
        interactive_radiation(pattern_type)

        print("Simulaciones completadas exitosamente.")

    except Exception as e:
        print(f"Se produjo un error durante la ejecución: {str(e)}")