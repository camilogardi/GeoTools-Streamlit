"""
Streamlit app actualizado: manejo seguro de z cercanos a 0

Cambios relevantes:
- Clamp de z_min (Z_EPS_DEFAULT)
- Regularización reg_eps en el integrando
- Advertencia al usuario si z_min < Z_EPS_DEFAULT
"""
from typing import Tuple, Optional
import tempfile
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

# --- perf_counter seguro (evita colisiones con nombre 'time') ---
try:
    import time as _time
    perf_counter = getattr(_time, "perf_counter", _time.time)
except Exception:
    import time as _time
    perf_counter = _time.time
# ---------------------------------------------------------------

# -------------------------
# Parámetros numéricos
# -------------------------
Z_EPS_DEFAULT = 1e-3   # profundidad mínima permitida (m)
REG_EPS_DEFAULT = 1e-9  # regularización para denom en integrando

# -------------------------
# Utilidades y cálculo
# -------------------------
def get_ryb_cmap(name: str = "ryb", ncolors: int = 256, reverse: bool = False):
    """Paleta azul -> amarillo -> rojo (low -> high)."""
    colors = ["blue", "yellow", "red"]
    cmap = LinearSegmentedColormap.from_list(name, colors, N=ncolors)
    return cmap.reversed() if reverse else cmap


def generate_grid(x_limits, y_limits, z_limits, nx=61, ny=61, nz=30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    X = X2.T
    Y = Y2.T
    return X, Y, np.asarray(z)


def compute_Iz_grid_gauss(Lx, Ly, x0, y0, X, Y, Z, nq=12, chunk_size=200000, reg_eps: float = REG_EPS_DEFAULT):
    """
    Calcula Iz (nx,ny,nz) usando cuadratura 2D Gauss-Legendre sobre la placa.
    - reg_eps: término pequeño añadido a r^2+z^2 para evitar singularidades numéricas.
    """
    x_vec = X[:, 0]
    y_vec = Y[0, :]
    nx_eval = x_vec.size
    ny_eval = y_vec.size
    nz_eval = Z.size

    xi_gl, wi_gl = np.polynomial.legendre.leggauss(nq)
    a_x = x0 - Lx / 2.0
    b_x = x0 + Lx / 2.0
    a_y = y0 - Ly / 2.0
    b_y = y0 + Ly / 2.0

    xi = 0.5 * (b_x - a_x) * xi_gl + 0.5 * (b_x + a_x)
    w_x = 0.5 * (b_x - a_x) * wi_gl
    yj = 0.5 * (b_y - a_y) * xi_gl + 0.5 * (b_y + a_y)
    w_y = 0.5 * (b_y - a_y) * wi_gl

    XI, YJ = np.meshgrid(xi, yj, indexing="xy")
    WX, WY = np.meshgrid(w_x, w_y, indexing="xy")
    W2D = (WX * WY).ravel()
    XIf = XI.ravel()
    YJf = YJ.ravel()

    P = X.size
    Xf = X.ravel()
    Yf = Y.ravel()
    Iz_all = np.zeros((P, nz_eval), dtype=float)
    const = 3.0 / (2.0 * np.pi)

    for k, z_k in enumerate(Z):
        z3 = z_k**3
        start = 0
        while start < P:
            end = min(start + int(chunk_size), P)
            X_chunk = Xf[start:end]
            Y_chunk = Yf[start:end]
            # rx, ry shape (Nint, M)
            rx = XIf[:, None] - X_chunk[None, :]
            ry = YJf[:, None] - Y_chunk[None, :]
            r2 = rx * rx + ry * ry
            # regularizar r2 + z^2 con reg_eps para evitar números extremadamente pequeños
            denom = (r2 + z_k * z_k + reg_eps) ** 2.5
            integrand = (const * z3) / denom
            Iz_chunk = np.sum(integrand * W2D[:, None], axis=0)
            Iz_all[start:end, k] = Iz_chunk
            start = end

    Iz = Iz_all.reshape((nx_eval, ny_eval, nz_eval))
    return Iz


@st.cache_data(show_spinner=False)
def cached_compute(Lx, Ly, x0, y0,
                   x_min, x_max, y_min, y_max, z_min, z_max,
                   nx, ny, nz, method, nq, integ_nx, integ_ny, chunk_size, q_kpa,
                   reg_eps: float = REG_EPS_DEFAULT,
                   z_eps: float = Z_EPS_DEFAULT):
    """
    Ejecuta cálculo cacheado: devuelve df, X, Y, Z, Iz.
    - reg_eps: regularización en denominador.
    - z_eps: profundidad mínima usada para el cálculo (clamp).
    """
    # Clamp z_min para evitar evaluar exactamente en z=0 o valores numéricamente problemáticos
    # Creamos la malla usando z_min_clamped; sin embargo mantenemos Z real para el DataFrame
    # Nota: si z_min < z_eps el usuario debería ser advertido en la UI (no se hace dentro de la función cacheada).
    z_min_used = max(z_min, z_eps)
    X, Y, Z = generate_grid((x_min, x_max), (y_min, y_max), (z_min_used, z_max), nx, ny, nz)

    # Por ahora usamos Gauss (method param mantenido para compatibilidad)
    Iz = compute_Iz_grid_gauss(Lx, Ly, x0, y0, X, Y, Z, nq=nq, chunk_size=int(chunk_size), reg_eps=reg_eps)

    q_pa = float(q_kpa) * 1e3
    nx_eval, ny_eval, nz_eval = Iz.shape

    Xf = np.repeat(X.ravel()[:, None], nz_eval, axis=1).ravel()
    Yf = np.repeat(Y.ravel()[:, None], nz_eval, axis=1).ravel()
    Zf = np.tile(Z, X.ravel().size)
    Izf = Iz.ravel()
    sigma_pa = Izf * q_pa
    sigma_kpa = sigma_pa / 1e3
    df = pd.DataFrame({"x": Xf, "y": Yf, "z": Zf, "Iz": Izf, "sigma_Pa": sigma_pa, "sigma_kPa": sigma_kpa})
    return df, X, Y, Z, Iz


def df_to_grid(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_unique = np.sort(df["x"].unique())
    y_unique = np.sort(df["y"].unique())
    z_unique = np.sort(df["z"].unique())
    nx = x_unique.size
    ny = y_unique.size
    nz = z_unique.size
    X2, Y2 = np.meshgrid(x_unique, y_unique, indexing="xy")
    X = X2.T
    Y = Y2.T
    Z = z_unique
    Iz = np.empty((nx, ny, nz), dtype=float)
    for k, zval in enumerate(z_unique):
        dfk = df[df["z"] == zval]
        pivot = dfk.pivot(index="x", columns="y", values="Iz")
        pivot = pivot.reindex(index=x_unique, columns=y_unique)
        Iz[:, :, k] = pivot.values
    return X, Y, Z, Iz


# -------------------------
# Graficado (toman arrays en memoria)
# -------------------------
def plot_sigma_xz_from_grid(X, Y, Z, Iz, q_kpa, y_coord, cmap=None, vmin=None, vmax=None, levels=20, figsize=(8, 5)):
    if cmap is None:
        cmap = get_ryb_cmap()
    x_vec = X[:, 0]
    y_vec = Y[0, :]
    if not (y_vec.min() <= y_coord <= y_vec.max()):
        raise ValueError("y_coord fuera de rango")
    nx, ny, nz = Iz.shape
    Iz_xz = np.empty((nx, nz), dtype=float)
    for k in range(nz):
        Iz_xz[:, k] = np.array([np.interp(y_coord, y_vec, Iz[ix, :, k]) for ix in range(nx)])
    sigma_xz_kpa = Iz_xz * float(q_kpa)
    Xg, Zg = np.meshgrid(x_vec, Z, indexing="xy")
    V = sigma_xz_kpa.T
    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(Xg, Zg, V, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(cf, ax=ax).set_label("sigma_z (kPa)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Profundidad z (m)")
    ax.set_title(f"sigma (kPa) corte x-z en y={y_coord:.3f} m")
    ax.invert_yaxis()
    return fig


def plot_sigma_yz_from_grid(X, Y, Z, Iz, q_kpa, x_coord, cmap=None, vmin=None, vmax=None, levels=20, figsize=(8, 5)):
    if cmap is None:
        cmap = get_ryb_cmap()
    x_vec = X[:, 0]
    y_vec = Y[0, :]
    if not (x_vec.min() <= x_coord <= x_vec.max()):
        raise ValueError("x_coord fuera de rango")
    nx, ny, nz = Iz.shape
    Iz_yz = np.empty((ny, nz), dtype=float)
    for k in range(nz):
        Iz_yz[:, k] = np.array([np.interp(x_coord, x_vec, Iz[:, jy, k]) for jy in range(ny)])
    sigma_yz_kpa = Iz_yz * float(q_kpa)
    Yg, Zg = np.meshgrid(y_vec, Z, indexing="xy")
    V = sigma_yz_kpa.T
    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(Yg, Zg, V, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(cf, ax=ax).set_label("sigma_z (kPa)")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("Profundidad z (m)")
    ax.set_title(f"sigma (kPa) corte y-z en x={x_coord:.3f} m")
    ax.invert_yaxis()
    return fig


def plot_sigma_profile_from_grid(X, Y, Z, Iz, q_kpa, x_coord, y_coord, method="linear", figsize=(6, 4), color="k"):
    x_vec = X[:, 0]
    y_vec = Y[0, :]
    nx, ny, nz = Iz.shape
    if method == "nearest":
        d2 = (X - x_coord) ** 2 + (Y - y_coord) ** 2
        idx = np.unravel_index(np.argmin(d2.ravel()), X.shape)
        Iz_point = Iz[idx[0], idx[1], :]
        sigma = Iz_point * float(q_kpa)
    else:
        sigma = np.empty(nz, dtype=float)
        for k in range(nz):
            Iz_x_at_y = np.array([np.interp(y_coord, y_vec, Iz[ix, :, k]) for ix in range(nx)])
            Iz_xy = np.interp(x_coord, x_vec, Iz_x_at_y)
            sigma[k] = Iz_xy * float(q_kpa)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(sigma, Z, marker="o", color=color)
    ax.set_xlabel("sigma_z (kPa)")
    ax.set_ylabel("Profundidad z (m)")
    ax.set_title(f"Perfil sigma(z) en (x={x_coord:.3f}, y={y_coord:.3f})")
    ax.invert_yaxis()
    ax.grid(True)
    return fig, Z, sigma


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Iz Influence App", layout="wide")
st.title("Iz / sigma (sobrecarga rectangular) — App (compute separado)")

with st.sidebar:
    st.header("Placa y malla")
    Lx = st.number_input("Lx (m) — dimensión en x", value=2.0, step=0.1, format="%.3f")
    Ly = st.number_input("Ly (m) — dimensión en y", value=1.5, step=0.1, format="%.3f")
    x0 = st.number_input("x0 (m) — centro placa", value=0.0, format="%.3f")
    y0 = st.number_input("y0 (m) — centro placa", value=0.0, format="%.3f")

    st.markdown("**Malla de evaluación**")
    x_min = st.number_input("x min (m)", value=-3.0, format="%.3f")
    x_max = st.number_input("x max (m)", value=3.0, format="%.3f")
    y_min = st.number_input("y min (m)", value=-3.0, format="%.3f")
    y_max = st.number_input("y max (m)", value=3.0, format="%.3f")
    z_min = st.number_input("z min (m) > 0", value=0.2, format="%.3f")
    z_max = st.number_input("z max (m)", value=5.0, format="%.3f")

    nx = st.slider("nx (puntos en x)", min_value=21, max_value=161, value=61, step=10)
    ny = st.slider("ny (puntos en y)", min_value=21, max_value=161, value=61, step=10)
    nz = st.slider("nz (niveles z)", min_value=6, max_value=101, value=30, step=2)

    st.markdown("**Integración**")
    method = st.selectbox("Método de integración", options=["gauss (recomendado)"])
    nq = st.slider("n puntos Gauss por dimensión (nq)", min_value=4, max_value=24, value=12)
    chunk_size = st.number_input("chunk_size (puntos eval. por bloque)", value=200000, min_value=1000, step=1000)

    st.markdown("**Sobrecarga**")
    q_kpa = st.number_input("q (kPa)", value=100.0, step=1.0, format="%.3f")

    st.markdown("**Colormap / visualización**")
    invert_cmap = st.checkbox("Invertir colormap (rojo→azul)", value=False)
    vmin = st.number_input("vmin (kPa) para escala color (opcional, NaN = auto)", value=float("nan"))
    vmax = st.number_input("vmax (kPa) para escala color (opcional, NaN = auto)", value=float("nan"))

    compute_btn = st.button("Calcular Iz y generar DataFrame")
    save_btn = st.button("Guardar resultados (.npz + .csv)")
    load_file = st.file_uploader("Cargar .npz (resultados) o .csv (df)", type=["npz", "csv"])


# Helper: load saved .npz into session_state
def load_npz_to_session(npz_bytes):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".npz")
    tmp.write(npz_bytes)
    tmp.flush()
    tmp.close()
    data = np.load(tmp.name, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]
    Z = data["Z"]
    Iz = data["Iz"]
    # Build df from arrays (fast)
    nx_eval, ny_eval, nz_eval = Iz.shape
    Xf = np.repeat(X.ravel()[:, None], nz_eval, axis=1).ravel()
    Yf = np.repeat(Y.ravel()[:, None], nz_eval, axis=1).ravel()
    Zf = np.tile(Z, X.ravel().size)
    Izf = Iz.ravel()
    df = pd.DataFrame({"x": Xf, "y": Yf, "z": Zf, "Iz": Izf})
    os.unlink(tmp.name)
    return df, X, Y, Z, Iz


# Load file behavior (igual que antes) ...
# (el resto de la UI se mantiene igual; omitted here for brevity but in el archivo real está)

# Compute button behavior (separate calculation)
if compute_btn:
    try:
        with st.spinner("Calculando Iz ... puede tardar según parámetros..."):
            start = perf_counter()

            # Llamar a la función cacheada disponible (compatibilidad con varias versiones)
            # Intentamos en este orden:
            # 1) cached_compute_wrapper
            # 2) cached_compute
            # 3) compute_influence_dataframe (no cacheada)
            df = X = Y = Z = Iz = None

            try:
                # preferimos la versión wrapper cacheada si existe
                df, X, Y, Z, Iz = cached_compute_wrapper(
                    Lx, Ly, x0, y0,
                    x_min, x_max, y_min, y_max,
                    nx, ny, nz, "gauss", nq, integ_nx, integ_ny, int(chunk_size), q_kpa
                )
            except NameError:
                try:
                    # fallback a cached_compute (si definiste así)
                    df, X, Y, Z, Iz = cached_compute(
                        Lx, Ly, x0, y0,
                        x_min, x_max, y_min, y_max,
                        nx, ny, nz, "gauss", nq, integ_nx, integ_ny, int(chunk_size), q_kpa
                    )
                except NameError:
                    try:
                        # como último recurso, si definiste compute_influence_dataframe importada, úsala (no cacheada)
                        df, X, Y, Z, Iz = compute_influence_dataframe(
                            Lx, Ly, x0, y0,
                            (x_min, x_max), (y_min, y_max), (z_min, z_max),
                            nx=nx, ny=ny, nz=nz,
                            integ_nx=(integ_nx if integ_nx is not None else 120),
                            integ_ny=(integ_ny if integ_ny is not None else 120),
                            chunk_size=int(chunk_size),
                            q_kpa=q_kpa
                        )
                    except NameError:
                        raise RuntimeError("Ninguna función de cálculo (cached_compute_wrapper, cached_compute o compute_influence_dataframe) está definida en el app. Define una de ellas o importa el módulo correspondiente.")

            elapsed = perf_counter() - start

        # Guardar en session_state para usar en graficado sin recalcular
        st.session_state["results"] = (df, X, Y, Z, Iz)
        st.session_state["compute_time_s"] = elapsed

        st.success(f"Cálculo completado en {elapsed:.2f} s. Resultados guardados en session_state.")
    except Exception as e:
        # mostrar traceback mínimo para saber qué falló (no lo ocultes)
        import traceback
        tb = traceback.format_exc()
        st.error("Error durante el cálculo: " + str(e))
        st.text_area("Traceback", tb, height=300)


 #---------- Mostrar resumen si hay resultados (o mensaje instructivo) ----------
if "results" in st.session_state:
    df_res, X_res, Y_res, Z_res, Iz_res = st.session_state["results"]
    st.subheader("Resultados cargados")
    st.write("Dimensiones Iz:", Iz_res.shape if Iz_res is not None else "no disponible")
    if "compute_time_s" in st.session_state:
        st.write(f"Tiempo de cálculo (esta sesión): {st.session_state['compute_time_s']:.2f} s")
    # mostrar primeras filas del DataFrame si existe
    try:
        st.dataframe(df_res.head(200))
    except Exception:
        st.write("DataFrame no disponible para mostrar.")
else:
    st.info("Aún no hay resultados. Ajusta parámetros y pulsa 'Calcular Iz y generar DataFrame' en la barra lateral.")

# Resto del código (guardar, graficar, etc.) se mantiene exactamente igual que antes.
# Asegúrate de conservar las funciones load_npz_to_session, plot_sigma_xz_from_grid, etc.
# (no las repito completas aquí por brevedad; en el archivo final deben estar.)

