# -------------------------
# Streamlit UI
# -------------------------


from typing import Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

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
    # expect arrays saved: X, Y, Z, Iz, df (optional)
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
    start_t = time.perf_counter()
    try:
        with st.spinner("Calculando Iz ... puede tardar según parámetros..."):
            df, X, Y, Z, Iz = cached_compute(Lx, Ly, x0, y0, x_min, x_max, y_min, y_max, z_min, z_max,
                                             nx, ny, nz, method, nq, None, None, chunk_size, q_kpa)
        elapsed = time.perf_counter() - start_t
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