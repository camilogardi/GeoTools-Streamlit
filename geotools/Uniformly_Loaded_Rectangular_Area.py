
def Iz_rectangular(x, y, z, Lx, Ly, x0=0.0, y0=0.0, nx=120, ny=120):
    """
    Factor de influencia Iz = sigma_z / q en el punto (x,y,z) por una placa
    rectangular de dimensiones L (en x) y B (en y), centrada en (x0,y0).

    Parámetros:
    - x, y : coordenadas del punto donde evaluar (m)
    - z    : profundidad (z>0) (m)
    - Lx    : dimensión en x de la placa (m)
    - Ly    : dimensión en y de la placa (m)
    - x0,y0: coordenadas del centro de la placa (por defecto 0,0) (m)
    - nx,ny: número de subdivisiones para la integración numérica (>= 80 recomendado)

    Retorna:
    - Iz : factor de influencia (adimensional) tal que sigma_z = q * Iz
    """
    if z <= 0:
        raise ValueError("z debe ser mayor que 0 (profundidad positiva).")

    # Coordenadas de integración sobre la placa (centro en x0,y0)
    xs = np.linspace(x0 - Lx/2.0, x0 + Lx/2.0, nx)
    ys = np.linspace(y0 - Ly/2.0, y0 + Ly/2.0, ny)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    # kernel de Boussinesq por unidad de área (normalizado por q):
    # kernel = 3 z^3 / (2 pi (r^2 + z^2)^(5/2))
    Iz_sum = 0.0
    for xi in xs:
        for yj in ys:
            rx = xi - x
            ry = yj - y
            r2 = rx*rx + ry*ry
            denom = (r2 + z*z)**2.5
            Iz_sum += (3.0 * z**3) / (2.0 * np.pi * denom)

    Iz = Iz_sum * dx * dy
    return Iz



"""
geotech_df_plots.py

Cálculo de Iz y sigma en una malla y funciones de graficado que aceptan
como entrada un pandas.DataFrame (con columnas x,y,z,Iz,sigma_kPa).

Funciones principales:
- generate_grid(...)                       -> X, Y, Z
- compute_Iz_grid(...)                     -> Iz (nx,ny,nz)
- compute_influence_dataframe(...)         -> df, X, Y, Z, Iz
- df_to_grid(df)                           -> X, Y, Z, Iz reconstruidos desde df
- plot_xz_from_df(df, y_coord, q_kpa, ...) -> grafica sigma (kPa) en corte x-z
- plot_yz_from_df(df, x_coord, q_kpa, ...) -> grafica sigma (kPa) en corte y-z
- plot_sigma_profile_from_df(df, x_coord, y_coord, q_kpa, ...) ->
                                             grafica sigma(z) en un punto

Notas:
- Iz es independiente de q. Para graficar sigma (kPa) multiplicamos Iz * q_kpa.
- compute_influence_dataframe crea la columna 'sigma_kPa' usando el q_kpa
  que pases allí. Sin embargo las funciones de graficado aceptan un q_kpa
  explícito para permitir mostrar sigma con otra carga si se desea.
- Requiere: numpy, pandas, matplotlib
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def generate_grid(x_limits, y_limits, z_limits, nx=61, ny=61, nz=30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera malla X (nx,ny), Y (nx,ny) y vector Z (nz,).
    x_limits, y_limits, z_limits: tuple(min,max) o arrays 1D.
    """
    def make_axis(arg, n):
        arr = np.asarray(arg)
        if arr.size == 2:
            return np.linspace(arr[0], arr[1], n)
        elif arr.ndim == 1:
            return arr
        else:
            raise ValueError("Los límites deben ser (min,max) o un array 1D.")

    x = make_axis(x_limits, nx)
    y = make_axis(y_limits, ny)
    z = make_axis(z_limits, nz)

    if np.any(z <= 0):
        raise ValueError("Todos los valores de z deben ser mayores que 0 (profundidad positiva).")

    X2, Y2 = np.meshgrid(x, y, indexing='xy')
    X = X2.T  # shape (nx, ny)
    Y = Y2.T
    return X, Y, np.asarray(z)


def compute_Iz_grid(Lx, Ly, x0, y0, X, Y, Z, integ_nx=120, integ_ny=120, chunk_size=200000) -> np.ndarray:
    """
    Calcula Iz (nx,ny,nz) integrando la solución de Boussinesq sobre la placa
    rectangular Lx (eje x) x Ly (eje y) centrada en (x0,y0).
    """
    x_vec = X[:, 0]
    y_vec = Y[0, :]
    nx_eval = x_vec.size
    ny_eval = y_vec.size
    nz_eval = Z.size

    xi = np.linspace(x0 - Lx/2.0, x0 + Lx/2.0, integ_nx)
    yj = np.linspace(y0 - Ly/2.0, y0 + Ly/2.0, integ_ny)
    dx = xi[1] - xi[0]
    dy = yj[1] - yj[0]
    area_elem = dx * dy

    XI, YJ = np.meshgrid(xi, yj, indexing='xy')
    XIf = XI.ravel()
    YJf = YJ.ravel()

    Xf = X.ravel()
    Yf = Y.ravel()
    P = Xf.size

    Iz_all = np.zeros((P, nz_eval), dtype=float)
    const = 3.0 / (2.0 * np.pi)

    for k, z_k in enumerate(Z):
        z3 = z_k**3
        start = 0
        while start < P:
            end = min(start + chunk_size, P)
            X_chunk = Xf[start:end]
            Y_chunk = Yf[start:end]
            rx = XIf[:, None] - X_chunk[None, :]
            ry = YJf[:, None] - Y_chunk[None, :]
            r2 = rx*rx + ry*ry
            denom = (r2 + z_k*z_k)**2.5
            integrand = (const * z3) / denom
            Iz_chunk = np.sum(integrand, axis=0) * area_elem
            Iz_all[start:end, k] = Iz_chunk
            start = end

    Iz = Iz_all.reshape((nx_eval, ny_eval, nz_eval))
    return Iz


def compute_influence_dataframe(Lx, Ly, x0, y0,
                                x_limits, y_limits, z_limits,
                                nx=61, ny=61, nz=30,
                                integ_nx=120, integ_ny=120, chunk_size=200000,
                                q_kpa=100.0) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera malla, calcula Iz y arma un DataFrame con columnas:
    ['x','y','z','Iz','sigma_Pa','sigma_kPa'].

    Retorna: df, X, Y, Z, Iz
    """
    X, Y, Z = generate_grid(x_limits, y_limits, z_limits, nx, ny, nz)
    Iz = compute_Iz_grid(Lx, Ly, x0, y0, X, Y, Z, integ_nx, integ_ny, chunk_size)

    q_pa = float(q_kpa) * 1e3
    nx_eval, ny_eval, nz_eval = Iz.shape

    # Construir vectores para DataFrame (orden consistente: recorrer puntos de planta y luego z)
    # Usamos el mismo orden que Iz.ravel(): Iz.ravel() itera sobre x (fast), y, z (slow)
    Xf = np.repeat(X.ravel()[:, None], nz_eval, axis=1).ravel()
    Yf = np.repeat(Y.ravel()[:, None], nz_eval, axis=1).ravel()
    Zf = np.tile(Z, X.ravel().size)
    Izf = Iz.ravel()
    sigma_pa = Izf * q_pa
    sigma_kpa = sigma_pa / 1e3

    df = pd.DataFrame({
        'x': Xf,
        'y': Yf,
        'z': Zf,
        'Iz': Izf,
        'sigma_Pa': sigma_pa,
        'sigma_kPa': sigma_kpa
    })
    return df, X, Y, Z, Iz


def df_to_grid(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruye X, Y, Z e Iz (nx,ny,nz) desde el DataFrame df que contiene
    las columnas 'x','y','z','Iz'. Asume que la malla es cartesian product
    (valores completos para cada combinación).
    """
    # Obtener coordenadas únicas ordenadas
    x_unique = np.sort(df['x'].unique())
    y_unique = np.sort(df['y'].unique())
    z_unique = np.sort(df['z'].unique())

    nx = x_unique.size
    ny = y_unique.size
    nz = z_unique.size

    # Construir X,Y mallas
    X2, Y2 = np.meshgrid(x_unique, y_unique, indexing='xy')
    X = X2.T
    Y = Y2.T
    Z = z_unique

    # Reconstruir Iz por niveles de z (pivot para cada z)
    Iz = np.empty((nx, ny, nz), dtype=float)
    for k, zval in enumerate(z_unique):
        dfk = df[df['z'] == zval]
        # pivot index=x, columns=y -> values Iz
        pivot = dfk.pivot(index='x', columns='y', values='Iz')
        # Asegurar orden de filas/columnas
        pivot = pivot.reindex(index=x_unique, columns=y_unique)
        Iz[:, :, k] = pivot.values

    return X, Y, Z, Iz



# -------------------- Graficado usando DataFrame como entrada --------------------


def get_ryb_cmap(name: str = 'ryb', ncolors: int = 256, reverse: bool = False):
    """
    Crea un colormap azul -> amarillo -> rojo (low -> mid -> high).
    Si reverse=True devuelve rojo->amarillo->azul (alto->bajo).

    Uso:
      cmap = get_ryb_cmap()            # low=blue ... high=red
      cmap_inv = get_ryb_cmap(reverse=True)  # high=blue ... low=red (invertido)
    """
    colors = ['blue', 'yellow', 'red']  # low -> high
    cmap = LinearSegmentedColormap.from_list(name, colors, N=ncolors)
    if reverse:
        cmap = cmap.reversed()
    return cmap




def _interp_in_plane(Iz: np.ndarray, axis_coords: np.ndarray, coord: float, axis: int):
    """
    Interpolación en plano:
    axis==0 -> interp en y para cada x,z -> devuelve (nx,nz)
    axis==1 -> interp en x para cada y,z -> devuelve (ny,nz)
    """
    nx, ny, nz = Iz.shape
    if axis == 0:
        y_vec = axis_coords
        if not (y_vec.min() <= coord <= y_vec.max()):
            raise ValueError("y_coord fuera de rango")
        out = np.empty((nx, nz), dtype=float)
        for k in range(nz):
            out[:, k] = np.array([np.interp(coord, y_vec, Iz[ix, :, k]) for ix in range(nx)])
        return out
    elif axis == 1:
        x_vec = axis_coords
        if not (x_vec.min() <= coord <= x_vec.max()):
            raise ValueError("x_coord fuera de rango")
        out = np.empty((ny, nz), dtype=float)
        for k in range(nz):
            out[:, k] = np.array([np.interp(coord, x_vec, Iz[:, jy, k]) for jy in range(ny)])
        return out
    else:
        raise ValueError("axis debe ser 0 (interp en y) o 1 (interp en x)")


def plot_xz_from_df(df: pd.DataFrame, y_coord: float, q_kpa: float,
                    method: str = 'linear', cmap: Optional[object] = None,
                    vmin: Optional[float] = None, vmax: Optional[float] = None,
                    levels=20, figsize=(10, 5), invert_z=True, title_prefix=''):
    """
    Grafica sigma (kPa) en corte x-z tomado en y = y_coord usando el DataFrame df.
    - df: DataFrame con columnas 'x','y','z','Iz' (puede contener sigma también).
    - y_coord: coordenada y en planta donde tomar el corte.
    - q_kpa: carga superficial en kPa (sigma_kPa = Iz * q_kpa).
    - method: 'linear' (interpolación) o 'nearest'.
    - cmap: colormap (por defecto get_ryb_cmap(), low=blue..high=red).
    - vmin/vmax: valores para normalizar escala de color (útil para comparar varias figuras).
    """
    if cmap is None:
        cmap = get_ryb_cmap()  # low=blue ... high=red

    X, Y, Z, Iz = df_to_grid(df)
    x_vec = X[:, 0]
    y_vec = Y[0, :]

    if method == 'nearest':
        j = int(np.argmin(np.abs(y_vec - y_coord)))
        Iz_xz = Iz[:, j, :]  # (nx, nz)
    else:
        Iz_xz = _interp_in_plane(Iz, y_vec, y_coord, axis=0)  # (nx, nz)

    sigma_xz_kpa = Iz_xz * float(q_kpa)

    Xg, Zg = np.meshgrid(x_vec, Z, indexing='xy')
    V = sigma_xz_kpa.T  # (nz, nx)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    cf = ax.contourf(Xg, Zg, V, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(cf, ax=ax).set_label('sigma_z (kPa)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('Profundidad z (m)')
    ax.set_title(f'{title_prefix}sigma (kPa) corte x-z en y={y_coord:.3f} m')
    if invert_z:
        ax.invert_yaxis()
    return fig, ax


def plot_yz_from_df(df: pd.DataFrame, x_coord: float, q_kpa: float,
                    method: str = 'linear', cmap: Optional[object] = None,
                    vmin: Optional[float] = None, vmax: Optional[float] = None,
                    levels=20, figsize=(10, 5), invert_z=True, title_prefix=''):
    """
    Grafica sigma (kPa) en corte y-z tomado en x = x_coord usando el DataFrame df.
    Parámetros similares a plot_xz_from_df.
    """
    if cmap is None:
        cmap = get_ryb_cmap()

    X, Y, Z, Iz = df_to_grid(df)
    x_vec = X[:, 0]
    y_vec = Y[0, :]

    if method == 'nearest':
        i = int(np.argmin(np.abs(x_vec - x_coord)))
        Iz_yz = Iz[i, :, :]  # (ny, nz)
    else:
        Iz_yz = _interp_in_plane(Iz, x_vec, x_coord, axis=1)  # (ny, nz)

    sigma_yz_kpa = Iz_yz * float(q_kpa)

    Yg, Zg = np.meshgrid(y_vec, Z, indexing='xy')
    V = sigma_yz_kpa.T  # (nz, ny)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    cf = ax.contourf(Yg, Zg, V, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(cf, ax=ax).set_label('sigma_z (kPa)')
    ax.set_xlabel('y (m)'); ax.set_ylabel('Profundidad z (m)')
    ax.set_title(f'{title_prefix}sigma (kPa) corte y-z en x={x_coord:.3f} m')
    if invert_z:
        ax.invert_yaxis()
    return fig, ax



def plot_sigma_profile_from_df(df: pd.DataFrame, x_coord: float, y_coord: float, q_kpa: float,
                               method: str = 'linear', figsize=(6, 5), marker='o', title_prefix=''):
    """
    Grafica sigma(z) (kPa) en la coordenada puntual (x_coord, y_coord) usando df.
    - method: 'nearest' o 'linear' (bilineal aproximado por 2 interpolaciones lineales).
    - Retorna fig, ax, Z, sigma_kpa_vec
    """
    X, Y, Z, Iz = df_to_grid(df)
    x_vec = X[:, 0]
    y_vec = Y[0, :]
    nx, ny, nz = Iz.shape

    if not (x_vec.min() <= x_coord <= x_vec.max()):
        raise ValueError("x_coord fuera de rango")
    if not (y_vec.min() <= y_coord <= y_vec.max()):
        raise ValueError("y_coord fuera de rango")

    q = float(q_kpa)
    sigma_kpa = np.empty(nz, dtype=float)

    if method == 'nearest':
        d2 = (X - x_coord)**2 + (Y - y_coord)**2
        i_flat = np.argmin(d2.ravel())
        i_idx, j_idx = np.unravel_index(i_flat, X.shape)
        Iz_point = Iz[i_idx, j_idx, :]
        sigma_kpa = Iz_point * q
    else:
        # interp en y para cada x, luego interp en x para x_coord, por cada z
        for k in range(nz):
            Iz_x_at_y = np.array([np.interp(y_coord, y_vec, Iz[ix, :, k]) for ix in range(nx)])  # (nx,)
            Iz_xy = np.interp(x_coord, x_vec, Iz_x_at_y)
            sigma_kpa[k] = Iz_xy * q

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    ax.plot(sigma_kpa, Z, marker=marker)
    ax.set_xlabel('sigma_z (kPa)')
    ax.set_ylabel('Profundidad z (m)')
    ax.set_title(f'{title_prefix}Perfil sigma(z) en (x={x_coord:.3f}, y={y_coord:.3f})')
    ax.invert_yaxis()
    ax.grid(True)
    return fig, ax, Z, sigma_kpa

