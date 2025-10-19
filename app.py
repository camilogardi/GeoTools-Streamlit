"""
Streamlit app corregido y autocontenido.

Cambios aplicados:
- Añadí imports faltantes (tempfile, os, time).
- Incluí implementaciones necesarias: get_ryb_cmap, generate_grid,
  compute_Iz_grid_gauss, cached_compute y funciones de graficado.
- Evité el uso inseguro de time.perf_counter() accediendo a perf_counter
  mediante un wrapper seguro; ahora medimos tiempo correctamente.
- Corregí el bloque de compute_btn para medir elapsed y guardarlo en session_state.
- El cálculo queda cacheado con @st.cache_data; las gráficas usan datos en session_state
  o cargados desde archivo .npz/.csv sin volver a calcular.

Ejecuta con:
  pip install streamlit numpy pandas matplotlib
  streamlit run app.py
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

# --- perf_counter seguro (añadir justo después de los imports) ---
try:
    import time as _time
    perf_counter = getattr(_time, "perf_counter", _time.time)
except Exception:
    import time as _time
    perf_counter = _time.time
# ---------------------------------------------------------------

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
    X2, Y2 = np.meshgrid(x, y, indexing="xy")
    X = X2.T
    Y = Y2.T
    return X, Y, np.asarray(z)


def compute_Iz_grid_gauss(Lx, Ly, x0, y0, X, Y, Z, nq=12, chunk_size=200000):
    """
    Calcula Iz (nx,ny,nz) usando cuadratura 2D Gauss-Legendre sobre la placa.
    - nq: puntos por dimensión (nq*nq nodos).
    - chunk_size: número de puntos de evaluación por bloque.
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
    Nint = XIf.size

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
            denom = (r2 + z_k * z_k) ** 2.5
            integrand = (const * z3) / denom
            Iz_chunk = np.sum(integrand * W2D[:, None], axis=0)
            Iz_all[start:end, k] = Iz_chunk
            start = end

    Iz = Iz_all.reshape((nx_eval, ny_eval, nz_eval))
    return Iz


@st.cache_data(show_spinner=False)
def cached_compute(Lx, Ly, x0, y0,
                   x_min, x_max, y_min, y_max, z_min, z_max,
                   nx, ny, nz, method, nq, integ_nx, integ_ny, chunk_size, q_kpa):
    """
    Ejecuta cálculo cacheado: devuelve df, X, Y, Z, Iz.
    - method: actualmente 'gauss' soportado.
    """
    X, Y, Z = generate_grid((x_min, x_max), (y_min, y_max), (z_min, z_max), nx, ny, nz)
    m = "gauss" if method.startswith("gauss") else "gauss"
    if m == "gauss":
        Iz = compute_Iz_grid_gauss(Lx, Ly, x0, y0, X, Y, Z, nq=nq, chunk_size=int(chunk_size))
    else:
        # fallback a gauss
        Iz = compute_Iz_grid_gauss(Lx, Ly, x0, y0, X, Y, Z, nq=nq, chunk_size=int(chunk_size))

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


# Load button behavior
if load_file is not None:
    if load_file.type == "application/x-npz" or load_file.name.endswith(".npz"):
        try:
            df_loaded, X_loaded, Y_loaded, Z_loaded, Iz_loaded = load_npz_to_session(load_file.read())
            st.session_state["results"] = (df_loaded, X_loaded, Y_loaded, Z_loaded, Iz_loaded)
            st.success("Archivo .npz cargado en session.")
        except Exception as e:
            st.error(f"Error cargando .npz: {e}")
    elif load_file.type == "text/csv" or load_file.name.endswith(".csv"):
        try:
            df_csv = pd.read_csv(load_file)
            # reconstruct grid (assume full cartesian)
            X_loaded, Y_loaded, Z_loaded, Iz_loaded = None, None, None, None
            # Try to reconstruct using df_to_grid-like procedure
            x_unique = np.sort(df_csv["x"].unique())
            y_unique = np.sort(df_csv["y"].unique())
            z_unique = np.sort(df_csv["z"].unique())
            X2, Y2 = np.meshgrid(x_unique, y_unique, indexing="xy")
            X_loaded = X2.T
            Y_loaded = Y2.T
            Z_loaded = z_unique
            nx = x_unique.size; ny = y_unique.size; nz_ = z_unique.size
            Iz_loaded = np.empty((nx, ny, nz_), dtype=float)
            for k, zval in enumerate(z_unique):
                dfk = df_csv[df_csv["z"] == zval]
                pivot = dfk.pivot(index="x", columns="y", values="Iz")
                pivot = pivot.reindex(index=x_unique, columns=y_unique)
                Iz_loaded[:, :, k] = pivot.values
            st.session_state["results"] = (df_csv, X_loaded, Y_loaded, Z_loaded, Iz_loaded)
            st.success("CSV cargado en session.")
        except Exception as e:
            st.error(f"Error cargando CSV: {e}")


# Compute button behavior (separate calculation)
if compute_btn:
    try:
        with st.spinner("Calculando Iz ... puede tardar según parámetros..."):
            start = perf_counter()
            df, X, Y, Z, Iz = cached_compute(Lx, Ly, x0, y0, x_min, x_max, y_min, y_max, z_min, z_max,
                                             nx, ny, nz, method, nq, None, None, chunk_size, q_kpa)
            elapsed = perf_counter() - start

        st.session_state["results"] = (df, X, Y, Z, Iz)
        st.session_state["compute_time_s"] = elapsed
        st.success(f"Cálculo completado en {elapsed:.2f} s. Resultados guardados en session_state.")
    except Exception as e:
        st.error(f"Error durante el cálculo: {e}")


# Save results to disk (.npz + CSV)
if save_btn:
    if "results" not in st.session_state:
        st.error("No hay resultados en session. Primero calcule o cargue.")
    else:
        df_s, X_s, Y_s, Z_s, Iz_s = st.session_state["results"]
        # save npz
        tmpdir = tempfile.gettempdir()
        fname = os.path.join(tmpdir, f"influence_results_{int(time.time())}.npz")
        np.savez_compressed(fname, X=X_s, Y=Y_s, Z=Z_s, Iz=Iz_s)
        csv_name = os.path.join(tmpdir, f"influence_results_{int(time.time())}.csv")
        df_s.to_csv(csv_name, index=False)
        st.success(f"Guardado a {fname} y {csv_name}. Puedes descargarlos desde el servidor o usar el botón de descarga abajo.")
        with open(fname, "rb") as f:
            st.download_button("Descargar .npz", f, file_name=os.path.basename(fname), mime="application/x-npz")
        with open(csv_name, "rb") as f:
            st.download_button("Descargar .csv", f, file_name=os.path.basename(csv_name), mime="text/csv")


# Show compute summary if exists
if "results" in st.session_state:
    df_res, X_res, Y_res, Z_res, Iz_res = st.session_state["results"]
    st.subheader("Resultados cargados")
    st.write("Dimensiones Iz:", Iz_res.shape)
    if "compute_time_s" in st.session_state:
        st.write(f"Tiempo de cálculo: {st.session_state['compute_time_s']:.2f} s (en esta sesión)")
    st.dataframe(df_res.head(200))


# Plotting UI (uses session_state results, does not recompute)
st.subheader("Graficar (usa resultados ya calculados o cargados)")

col1, col2 = st.columns(2)
with col1:
    y_coord_plot = st.number_input("y para corte x-z", value=float(y0), format="%.3f", key="plot_y")
    levels_plot = st.slider("Niveles contorno", min_value=8, max_value=60, value=20, key="levels")
    if st.button("Generar corte x-z (desde resultados)"):
        if "results" not in st.session_state:
            st.error("No hay resultados. Pulsa 'Calcular Iz' o carga un archivo .npz/.csv.")
        else:
            _, Xr, Yr, Zr, Izr = st.session_state["results"]
            cmap = get_ryb_cmap()
            if invert_cmap:
                cmap = cmap.reversed()
            vmin_val = None if np.isnan(vmin) else float(vmin)
            vmax_val = None if np.isnan(vmax) else float(vmax)
            try:
                fig = plot_sigma_xz_from_grid(Xr, Yr, Zr, Izr, q_kpa, y_coord_plot, cmap=cmap, vmin=vmin_val, vmax=vmax_val, levels=levels_plot)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error graficando: {e}")

with col2:
    x_coord_plot = st.number_input("x para corte y-z", value=float(x0), format="%.3f", key="plot_x")
    if st.button("Generar corte y-z (desde resultados)"):
        if "results" not in st.session_state:
            st.error("No hay resultados. Pulsa 'Calcular Iz' o carga un archivo .npz/.csv.")
        else:
            _, Xr, Yr, Zr, Izr = st.session_state["results"]
            cmap = get_ryb_cmap()
            if invert_cmap:
                cmap = cmap.reversed()
            vmin_val = None if np.isnan(vmin) else float(vmin)
            vmax_val = None if np.isnan(vmax) else float(vmax)
            try:
                fig = plot_sigma_yz_from_grid(Xr, Yr, Zr, Izr, q_kpa, x_coord_plot, cmap=cmap, vmin=vmin_val, vmax=vmax_val, levels=levels_plot)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error graficando: {e}")

st.markdown("---")
st.subheader("Perfil 1D sigma(z) en (x,y)")
x_profile = st.number_input("x para perfil 1D", value=float(x0), format="%.3f", key="prof_x")
y_profile = st.number_input("y para perfil 1D", value=float(y0), format="%.3f", key="prof_y")
method_profile = st.selectbox("Método perfil", options=["linear", "nearest"], key="prof_method")
if st.button("Generar perfil sigma(z) (desde resultados)"):
    if "results" not in st.session_state:
        st.error("No hay resultados. Pulsa 'Calcular Iz' o carga un archivo .npz/.csv.")
    else:
        _, Xr, Yr, Zr, Izr = st.session_state["results"]
        try:
            fig, Zvec, sigma_prof = plot_sigma_profile_from_grid(Xr, Yr, Zr, Izr, q_kpa, x_profile, y_profile, method=method_profile)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generando perfil: {e}")


st.info("Consejo: guarda los resultados (.npz) si el cálculo tardó mucho y luego cárgalos en otra sesión para ver gráficos sin recalcular.")



