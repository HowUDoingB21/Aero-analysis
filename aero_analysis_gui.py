"""
=============================================================================
  AeroSim Analyzer — SimScale CSV Aerodynamic Data Viewer
  Interpolación polinomial + GUI con Tkinter/Matplotlib
=============================================================================
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
try:
    from scipy.interpolate import make_interp_spline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
from itertools import product as iproduct
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Paleta de colores
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":       "#0f1117",
    "surface":  "#1a1d27",
    "card":     "#22263a",
    "accent":   "#4f8ef7",
    "accent2":  "#f7724f",
    "accent3":  "#4ff7a0",
    "text":     "#e8eaf0",
    "subtext":  "#8b90a8",
    "border":   "#2d3148",
    "success":  "#4ff7a0",
    "warning":  "#f7d44f",
}

SPEED_COLORS = ["#4f8ef7", "#f7724f", "#4ff7a0", "#f7d44f", "#c44ff7"]

FORCE_OPTIONS = {
    "Lift [N]  — Sustentacion real (descompuesta)": "lift_N",
    "Drag [N]  — Arrastre real (descompuesto)":     "drag_N",
    "L/D       — Relacion Lift / Drag":             "L_D_ratio",
    "CL  — Coef. de Sustentacion":                  "CL",
    "CD  — Coef. de Arrastre":                      "CD",
    "L/D coef  — CL / CD":                          "LD",
    "Fy  — Fuerza lateral [N]":                     "fy",
    "Fx  — Fuerza axial [N]":                       "fx",
    "Fz  — Fuerza normal [N]":                      "fz",
    "CZ  — Coef. de Fuerza Normal":                 "CZ",
}

POLY_OPTIONS = [
    "Lineal (grado 1)",
    "Cuadrático (grado 2)",
    "Cúbico (grado 3)",
    "Grado 4",
    "Grado 5",
    "Grado 6",
    "Grado 7",
]

INTERP_OPTIONS = {
    "Polinomial (np.polyfit)": "poly",
    "Spline cúbico":           "spline",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Interpolación bivariable  f(V, α) → variable aerodinámica
# ─────────────────────────────────────────────────────────────────────────────

def poly2d_features(V_arr, aoa_arr, deg):
    """Crea la matriz de características para polinomio 2D de grado deg."""
    feats = []
    for i, j in iproduct(range(deg + 1), range(deg + 1)):
        if i + j <= deg:
            feats.append(V_arr**i * aoa_arr**j)
    return np.column_stack(feats)


# Métodos de ajuste 2D disponibles
SURF_METHODS = {
    "Spline CL(alpha) x q  [MEJOR PRECISION]": "spline_cl",
    "Fisico: CL(alpha) x q  [RECOMENDADO]":    "physical",
    "Polinomio 2D normalizado":                 "poly2d",
    "RBF (scipy, interpolacion exacta)":        "rbf",
}


def fit_surface(V_arr, aoa_arr, y_arr, rho_arr, method, deg, col, sref=1.0):
    """
    Ajusta la superficie f(V, alpha) usando el método seleccionado.
    Devuelve un dict con todo lo necesario para evaluar y reportar.
    """
    result = dict(method=method, deg=deg, col=col,
                  V_mu=V_arr.mean(), V_s=V_arr.std(),
                  aoa_mu=aoa_arr.mean(), aoa_s=aoa_arr.std(),
                  rho_mean=rho_arr.mean(), sref=float(sref))

    # Presión dinámica q = ½ρV²
    q_arr = 0.5 * rho_arr * V_arr ** 2
    result["q_arr"] = q_arr

    # ── Spline cúbico sobre CL promedio — máxima precisión ─────────────────
    if method == "spline_cl":
        if not HAS_SCIPY:
            raise ImportError("Instala scipy: pip install scipy")
        from scipy.interpolate import CubicSpline
        sref   = result.get("sref", 1.0)
        cl_arr = y_arr / np.where(q_arr * sref > 0, q_arr * sref, 1e-9)
        result["sref"] = sref

        # Promediar CL por ángulo de ataque (elimina varianza inter-velocidades)
        aoa_unique = np.sort(np.unique(aoa_arr))
        cl_mean    = np.array([
            cl_arr[aoa_arr == a].mean() for a in aoa_unique
        ])
        cs         = CubicSpline(aoa_unique, cl_mean, extrapolate=True)
        result["spline_cl"] = cs
        result["aoa_knots"] = aoa_unique
        result["cl_knots"]  = cl_mean
        y_hat = cs(aoa_arr) * q_arr * sref

    # ── Método físico: ajusta CL(α) y predice Lift = CL×q×S ────────────────
    elif method == "physical":
        # CL = lift / (q × S)  →  independiente de ρ y V
        sref   = result.get("sref", 1.0)
        cl_arr = y_arr / np.where(q_arr * sref > 0, q_arr * sref, 1e-9)
        X1d    = np.column_stack([aoa_arr ** k for k in range(deg + 1)])
        coeffs, _, _, _ = np.linalg.lstsq(X1d, cl_arr, rcond=None)
        result["coeffs_1d"] = coeffs
        result["sref"]      = sref
        y_hat  = (X1d @ coeffs) * q_arr * sref

    # ── Polinomio 2D con entradas normalizadas ───────────────────────────────
    elif method == "poly2d":
        Vn  = (V_arr   - result["V_mu"])   / max(result["V_s"],   1e-9)
        an  = (aoa_arr - result["aoa_mu"]) / max(result["aoa_s"], 1e-9)
        X   = poly2d_features(Vn, an, deg)
        coeffs, _, _, _ = np.linalg.lstsq(X, y_arr, rcond=None)
        result["coeffs_2d"] = coeffs
        y_hat = X @ coeffs

    # ── RBF exacto ───────────────────────────────────────────────────────────
    elif method == "rbf":
        if not HAS_SCIPY:
            raise ImportError("Instala scipy: pip install scipy")
        from scipy.interpolate import RBFInterpolator
        Vn = (V_arr   - result["V_mu"])   / max(result["V_s"],   1e-9)
        an = (aoa_arr - result["aoa_mu"]) / max(result["aoa_s"], 1e-9)
        pts = np.column_stack([Vn, an])
        rbf = RBFInterpolator(pts, y_arr, kernel="thin_plate_spline", degree=2)
        result["rbf_model"] = rbf
        result["pts_train"] = pts
        y_hat = rbf(pts)

    # Métricas globales
    ss_res = np.sum((y_arr - y_hat) ** 2)
    ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
    result["r2"]    = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    result["rmse"]  = float(np.sqrt(np.mean((y_arr - y_hat) ** 2)))
    result["y_hat"] = y_hat

    # Métricas por velocidad
    errs_by_speed = {}
    for spd in sorted(np.unique(V_arr)):
        m     = V_arr == spd
        real  = y_arr[m]
        pred  = y_hat[m]
        rmse_s  = float(np.sqrt(np.mean((real - pred) ** 2)))
        # Error relativo: usa max(|real|) como referencia para evitar /0
        ref   = max(np.abs(real).max(), 1e-9)
        err_p = float(np.sqrt(np.mean((real - pred) ** 2)) / ref * 100)
        errs_by_speed[float(spd)] = {"rmse": rmse_s, "err_pct": err_p}
    result["errs_by_speed"] = errs_by_speed

    return result


def eval_surface(fit_result, V_val, aoa_val, rho_val=1.225):
    """Evalúa la superficie ajustada en (V, aoa)."""
    method = fit_result["method"]
    q_val  = 0.5 * rho_val * float(V_val) ** 2

    if method == "spline_cl":
        sref = fit_result.get("sref", 1.0)
        cl   = float(fit_result["spline_cl"](float(aoa_val)))
        return cl * q_val * sref

    elif method == "physical":
        X1d  = np.array([float(aoa_val) ** k
                         for k in range(len(fit_result["coeffs_1d"]))])
        cl   = float(X1d @ fit_result["coeffs_1d"])
        sref = fit_result.get("sref", 1.0)
        return cl * q_val * sref

    elif method == "poly2d":
        Vn  = (float(V_val)   - fit_result["V_mu"])   / max(fit_result["V_s"],   1e-9)
        an  = (float(aoa_val) - fit_result["aoa_mu"]) / max(fit_result["aoa_s"], 1e-9)
        x   = poly2d_features(np.array([Vn]), np.array([an]),
                               fit_result["deg"])
        return float(x @ fit_result["coeffs_2d"])

    elif method == "rbf":
        Vn  = (float(V_val)   - fit_result["V_mu"])   / max(fit_result["V_s"],   1e-9)
        an  = (float(aoa_val) - fit_result["aoa_mu"]) / max(fit_result["aoa_s"], 1e-9)
        return float(fit_result["rbf_model"](np.array([[Vn, an]])))

    return float("nan")


def format_surface_equation(fit_result, col_name="f"):
    """Formatea la ecuación del ajuste como string."""
    method = fit_result["method"]

    if method == "spline_cl":
        knots = fit_result.get("aoa_knots", [])
        cl_k  = fit_result.get("cl_knots",  [])
        lines = ["f(V, \u03b1) = Lift",
                 "  CL(\u03b1) = Spline c\u00fabico (interpolaci\u00f3n exacta)",
                 f"  N\u00ba nodos: {len(knots)}  |  \u03b1 \u2208 [{knots.min():.1f}\u00b0, {knots.max():.1f}\u00b0]",
                 f"  CL \u2208 [{cl_k.min():.5f}, {cl_k.max():.5f}]",
                 "  Lift = CL(\u03b1) \u00d7 \u00bd\u03c1V\u00b2 \u00d7 S"]
        return "\n".join(lines)

    elif method == "physical":
        coeffs = fit_result["coeffs_1d"]
        deg    = len(coeffs) - 1
        terms  = []
        for k, c in enumerate(coeffs):
            if abs(c) < 1e-14:
                continue
            sign   = "+" if c >= 0 else "\u2212"
            a_part = ("" if k == 0
                      else ("\u00b7\u03b1" if k == 1
                            else f"\u00b7\u03b1^{k}"))
            terms.append((sign, f"{abs(c):.4g}{a_part}"))
        line = "  CL(\u03b1) = "
        result = []
        for k, (sign, term) in enumerate(terms):
            chunk = (term if k == 0 and sign == "+"
                     else (f"\u2212{term}" if k == 0 else f" {sign} {term}"))
            if len(line) + len(chunk) > 68:
                result.append(line); line = "    " + chunk.lstrip()
            else:
                line += chunk
        result.append(line)
        return (f"f(V, \u03b1) = {col_name}\n"
                + "\n".join(result)
                + f"\n  Lift = CL(\u03b1) \u00d7 \u00bd\u03c1V\u00b2 \u00d7 S")

    elif method == "poly2d":
        coeffs = fit_result.get("coeffs_2d", [])
        deg    = fit_result["deg"]
        terms  = []
        for idx, (i, j) in enumerate(
            [(i, j) for i in range(deg+1) for j in range(deg+1) if i+j<=deg]
        ):
            if idx >= len(coeffs): break
            c = coeffs[idx]
            if abs(c) < 1e-14: continue
            v_p = ("" if i==0 else ("\u00b7V\u0302" if i==1 else f"\u00b7V\u0302^{i}"))
            a_p = ("" if j==0 else ("\u00b7\u03b1\u0302" if j==1 else f"\u00b7\u03b1\u0302^{j}"))
            sign = "+" if c >= 0 else "\u2212"
            terms.append((sign, f"{abs(c):.4g}{v_p}{a_p}"))
        line = "  f(V\u0302,\u03b1\u0302) = "; result2 = []
        for k,(sign,term) in enumerate(terms):
            chunk=(term if k==0 and sign=="+" else (f"\u2212{term}" if k==0 else f" {sign} {term}"))
            if len(line)+len(chunk)>68: result2.append(line); line="    "+chunk.lstrip()
            else: line+=chunk
        result2.append(line)
        note = (f"  (V\u0302=(V\u2212{fit_result['V_mu']:.1f})/{fit_result['V_s']:.1f},  "
                f"\u03b1\u0302=(\u03b1\u2212{fit_result['aoa_mu']:.2f})/{fit_result['aoa_s']:.2f})")
        return f"f(V, \u03b1) = {col_name}\n" + "\n".join(result2) + "\n" + note

    else:
        return f"f(V, \u03b1) = {col_name}\n  RBF Thin Plate Spline (interpolacion exacta en puntos de datos)"


# Legacy shims for backward compatibility
def fit_poly2d(V_arr, aoa_arr, y_arr, deg):
    r = fit_surface(V_arr, aoa_arr, y_arr,
                    np.full_like(V_arr, 1.225), "poly2d", deg, "")
    return r["coeffs_2d"], r["r2"], r["rmse"]

def eval_poly2d(coeffs, V_val, aoa_val, deg):
    return float("nan")  # replaced by eval_surface


def format_poly2d_equation(coeffs, deg, col_name="f"):
    """Formatea la ecuación polinomial 2D como string legible."""
    lines_out = [f"f(V, \u03b1) = {col_name}"]
    terms = []
    for i, j in iproduct(range(deg + 1), range(deg + 1)):
        if i + j > deg:
            continue
        idx  = len(terms)
        c    = coeffs[idx] if idx < len(coeffs) else 0.0
        if abs(c) < 1e-14:
            terms.append(None)
            continue
        v_part = ("" if i == 0
                  else ("\u00b7V"      if i == 1
                        else f"\u00b7V^{i}"))
        a_part = ("" if j == 0
                  else ("\u00b7\u03b1"      if j == 1
                        else f"\u00b7\u03b1^{j}"))
        sign   = "+" if c >= 0 else "\u2212"
        terms.append((sign, f"{abs(c):.4g}{v_part}{a_part}"))

    valid  = [(s, t) for item in terms if item for s, t in [item]]
    line   = "  = "
    result = []
    for k, (sign, term) in enumerate(valid):
        chunk = (term if k == 0 and sign == "+"
                 else (f"\u2212{term}" if k == 0
                       else f" {sign} {term}"))
        if len(line) + len(chunk) > 68:
            result.append(line)
            line = "    " + chunk.lstrip()
        else:
            line += chunk
    result.append(line)
    return "\n".join([lines_out[0]] + result)



# ─────────────────────────────────────────────────────────────────────────────
#  Funciones de cálculo
# ─────────────────────────────────────────────────────────────────────────────

def compute_coefficients(df: pd.DataFrame, sref: float) -> pd.DataFrame:
    """Agrega columnas de coeficientes aerodinámicos al dataframe.
    Usa lift_N / drag_N precalculados si existen, si no los deriva de fy/fx."""
    rho = df["density_kgm3"]
    V   = df["speed_ms"]
    q   = 0.5 * rho * V**2
    qS  = q * sref

    df = df.copy()
    df["q"] = q

    # --- Lift y Drag: usar columnas precalculadas si están disponibles ---
    if "lift_N" in df.columns and "drag_N" in df.columns:
        # Ya están correctamente descompuestos por AoA
        pass
    else:
        # Fallback: aproximación con fy/fx
        df["lift_N"]    = df["fy"]
        df["drag_N"]    = -df["fx"]
        df["L_D_ratio"] = (df["lift_N"] / df["drag_N"].replace(0, np.nan)).round(4)

    # --- Coeficientes adimensionales ---
    df["CL"] = df["lift_N"] / qS
    df["CD"] = df["drag_N"] / qS
    df["CZ"] = df["fz"]     / qS
    df["LD"] = (df["CL"] / df["CD"].replace(0, np.nan))
    return df


def do_interpolation(aoa_pts: np.ndarray, y_pts: np.ndarray,
                     method: str, degree: int):
    """Devuelve arrays (x_fine, y_fine) para la curva interpolada."""
    sort_idx = np.argsort(aoa_pts)
    x = aoa_pts[sort_idx]
    y = y_pts[sort_idx]

    x_fine = np.linspace(x.min(), x.max(), 400)

    if method == "poly":
        coeffs  = np.polyfit(x, y, degree)
        y_fine  = np.polyval(coeffs, x_fine)
        label   = f"Polinomio grado {degree}"
        return x_fine, y_fine, label, coeffs

    elif method == "spline":
        if HAS_SCIPY:
            k      = min(3, len(x) - 1)
            spl    = make_interp_spline(x, y, k=k)
            y_fine = spl(x_fine)
            label  = "Spline cúbico"
        else:
            y_fine = np.interp(x_fine, x, y)
            label  = "Spline lineal (instala scipy para cúbico)"
        return x_fine, y_fine, label, None

    return x_fine, np.zeros_like(x_fine), "—", None


# ─────────────────────────────────────────────────────────────────────────────
#  Aplicación principal
# ─────────────────────────────────────────────────────────────────────────────

class AeroApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("AeroSim Analyzer — SimScale")
        self.geometry("1400x860")
        self.minsize(1100, 700)
        self.configure(bg=PALETTE["bg"])

        self.df_raw   = None      # dataframe original
        self.df_work  = None      # con coeficientes

        # Variables de control
        self.var_file   = tk.StringVar(value="Sin archivo cargado")
        self.var_sref   = tk.DoubleVar(value=1.0)
        self.var_force  = tk.StringVar(value=list(FORCE_OPTIONS.keys())[0])
        self.var_degree = tk.IntVar(value=3)
        self.var_method = tk.StringVar(value=list(INTERP_OPTIONS.keys())[0])
        self.var_status = tk.StringVar(value="Cargue un archivo CSV para comenzar.")
        self.speed_vars    = {}        # {speed_value: BooleanVar}
        self.surf_fit      = None     # resultado del ajuste 2D
        self.var_surf_col  = tk.StringVar(value=list(FORCE_OPTIONS.keys())[0])
        self.var_surf_deg  = tk.IntVar(value=3)
        self.var_v_query   = tk.DoubleVar(value=190.0)
        self.var_aoa_query = tk.DoubleVar(value=10.0)
        self.var_rho_query = tk.DoubleVar(value=1.225)
        self.var_query_result = tk.StringVar(value="—")
        self.var_show_raw   = tk.BooleanVar(value=True)
        self.var_show_interp= tk.BooleanVar(value=True)
        self.var_show_resid = tk.BooleanVar(value=False)
        self.var_show_eq    = tk.BooleanVar(value=True)
        self.var_grid       = tk.BooleanVar(value=True)

        self._configure_style()
        self._build_ui()


    # ─── ESTILOS TTK (fix Windows dark mode) ─────────────────────────────────

    def _configure_style(self):
        """Configura ttk.Style con tema clam para que los widgets
        respeten colores del tema oscuro en Windows."""
        style = ttk.Style(self)
        style.theme_use("clam")

        BG   = PALETTE["card"]
        FG   = PALETTE["text"]
        SEL  = PALETTE["accent"]
        SUB  = PALETTE["subtext"]
        SURF = PALETTE["surface"]

        style.configure("Dark.TCombobox",
                        fieldbackground=BG, background=BG,
                        foreground=FG, selectbackground=SEL,
                        selectforeground="white", arrowcolor=FG,
                        bordercolor=PALETTE["border"],
                        lightcolor=PALETTE["border"],
                        darkcolor=PALETTE["border"],
                        font=("Helvetica", 9))
        style.map("Dark.TCombobox",
                  fieldbackground=[("readonly", BG)],
                  foreground=[("readonly", FG)],
                  background=[("readonly", BG), ("active", PALETTE["border"])],
                  arrowcolor=[("readonly", FG)])

        self.option_add("*TCombobox*Listbox.background",       BG)
        self.option_add("*TCombobox*Listbox.foreground",       FG)
        self.option_add("*TCombobox*Listbox.selectBackground", SEL)
        self.option_add("*TCombobox*Listbox.selectForeground", "white")
        self.option_add("*TCombobox*Listbox.font", ("Helvetica", 9))

        style.configure("Dark.Vertical.TScrollbar",
                        background=PALETTE["border"], troughcolor=SURF,
                        arrowcolor=FG, bordercolor=SURF)
        style.configure("TSeparator", background=PALETTE["border"])

        style.configure("Dark.TNotebook", background=PALETTE["bg"],
                        bordercolor=PALETTE["border"], tabmargins=[2, 2, 0, 0])
        style.configure("Dark.TNotebook.Tab",
                        background=PALETTE["card"], foreground=SUB,
                        padding=[12, 5], font=("Helvetica", 9))
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", SURF)],
                  foreground=[("selected", FG)])

        style.configure("Dark.Treeview",
                        background=BG, foreground=FG,
                        fieldbackground=BG, rowheight=24,
                        font=("Helvetica", 9))
        style.configure("Dark.Treeview.Heading",
                        background=SURF, foreground=SEL,
                        font=("Helvetica", 9, "bold"), relief="flat")
        style.map("Dark.Treeview",
                  background=[("selected", SEL)],
                  foreground=[("selected", "white")])

    # ─── CONSTRUCCIÓN DE LA INTERFAZ ─────────────────────────────────────────

    def _build_ui(self):
        # Barra superior
        self._build_topbar()

        # Contenedor principal
        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL,
                               bg=PALETTE["bg"], sashwidth=6,
                               sashrelief=tk.FLAT)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Panel izquierdo — controles
        left = tk.Frame(paned, bg=PALETTE["surface"], width=320)
        left.pack_propagate(False)
        paned.add(left, minsize=260)

        # Panel derecho — gráfica + tabla
        right = tk.Frame(paned, bg=PALETTE["bg"])
        paned.add(right, minsize=600)

        self._build_controls(left)
        self._build_plot_area(right)

        # Barra de estado
        self._build_statusbar()

    def _build_topbar(self):
        bar = tk.Frame(self, bg=PALETTE["card"], height=52)
        bar.pack(fill=tk.X, padx=0, pady=0)
        bar.pack_propagate(False)

        tk.Label(bar, text="AeroSim Analyzer",
                 font=("Helvetica", 16, "bold"),
                 fg=PALETTE["accent"], bg=PALETTE["card"]).pack(side=tk.LEFT, padx=18)

        tk.Label(bar, text="Análisis de datos CFD con interpolación polinomial",
                 font=("Helvetica", 10), fg=PALETTE["subtext"],
                 bg=PALETTE["card"]).pack(side=tk.LEFT, padx=4)

        tk.Button(bar, text="Cargar CSV",
                  command=self._load_file,
                  font=("Helvetica", 10, "bold"),
                  fg="white", bg=PALETTE["accent"],
                  activebackground="#3a7ae0", activeforeground="white",
                  relief=tk.RIDGE, bd=1, padx=14, pady=6,
                  cursor="hand2").pack(side=tk.RIGHT, padx=14, pady=8)

    def _build_controls(self, parent):
        canvas  = tk.Canvas(parent, bg=PALETTE["surface"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview, style="Dark.Vertical.TScrollbar")
        frame   = tk.Frame(canvas, bg=PALETTE["surface"])

        frame.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Asociar scroll con rueda del ratón
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        W = 290

        # ── Archivo ──
        self._section(frame, "  Archivo CSV", W)
        self._info_label(frame, self.var_file, W)

        # ── Parámetros de referencia ──
        self._section(frame, "  Parametros de referencia", W)
        self._labeled_entry(frame, "Área de referencia Sref [m²]:", self.var_sref, W)

        btn_apply = tk.Button(frame, text="Recalcular coeficientes",
                              command=self._apply_sref,
                              font=("Helvetica", 9, "bold"),
                              fg="white", bg=PALETTE["accent2"],
                              activebackground="#e05a30", activeforeground="white",
                              relief=tk.RIDGE, bd=1, padx=10, pady=5,
                              cursor="hand2")
        btn_apply.pack(fill=tk.X, padx=10, pady=(2, 8))

        # ── Velocidades ──
        self._section(frame, "  Velocidades [m/s]", W)
        self.speed_frame = tk.Frame(frame, bg=PALETTE["surface"])
        self.speed_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        # ── Variable a graficar ──
        self._section(frame, "  Variable a graficar", W)
        self.force_combo = ttk.Combobox(frame,
                                        textvariable=self.var_force,
                                        values=list(FORCE_OPTIONS.keys()),
                                        style="Dark.TCombobox",
                                        state="readonly", width=36)
        self.force_combo.pack(padx=10, pady=(0, 8))
        self.force_combo.current(0)
        self.force_combo.bind("<<ComboboxSelected>>", lambda e: self._plot())

        # ── Interpolación ──
        self._section(frame, "  Interpolacion", W)
        self._labeled_combo(frame, "Método:",
                            self.var_method,
                            list(INTERP_OPTIONS.keys()), W)
        self._labeled_combo(frame, "Grado polinomial:",
                            None, POLY_OPTIONS, W,
                            is_index=True, var_int=self.var_degree)

        # ── Opciones de visualización ──
        self._section(frame, "  Visualizacion", W)
        for text, var in [
            ("Mostrar puntos de simulación", self.var_show_raw),
            ("Mostrar curva interpolada",    self.var_show_interp),
            ("Mostrar residuales",           self.var_show_resid),
            ("Mostrar ecuación en gráfica",  self.var_show_eq),
            ("Grilla",                       self.var_grid),
        ]:
            self._checkbox(frame, text, var)

        # ── Superficie 2D ──
        self._section(frame, "  Superficie f(V,\u03b1)", W)

        tk.Label(frame, text="Variable:", font=("Helvetica", 9),
                 fg=PALETTE["text"], bg=PALETTE["surface"],
                 anchor="w").pack(fill=tk.X, padx=10)
        self.surf_force_combo = ttk.Combobox(frame,
                                             textvariable=self.var_surf_col,
                                             values=list(FORCE_OPTIONS.keys()),
                                             style="Dark.TCombobox",
                                             state="readonly", width=36)
        self.surf_force_combo.pack(padx=10, pady=(2, 4))
        self.surf_force_combo.current(0)

        # Método de ajuste bivariable
        self.var_surf_method = tk.StringVar(value=list(SURF_METHODS.keys())[0])  # spline_cl
        tk.Label(frame, text="Metodo de ajuste:", font=("Helvetica", 9),
                 fg=PALETTE["text"], bg=PALETTE["surface"],
                 anchor="w").pack(fill=tk.X, padx=10)
        self.surf_method_combo = ttk.Combobox(frame,
                                              textvariable=self.var_surf_method,
                                              values=list(SURF_METHODS.keys()),
                                              style="Dark.TCombobox",
                                              state="readonly", width=36)
        self.surf_method_combo.pack(padx=10, pady=(2, 4))
        self.surf_method_combo.current(0)

        tk.Label(frame, text="Grado polinomio (1-6):", font=("Helvetica", 9),
                 fg=PALETTE["text"], bg=PALETTE["surface"],
                 anchor="w").pack(fill=tk.X, padx=10)
        self.surf_deg_spin = tk.Spinbox(frame, from_=1, to=6,
                                        textvariable=self.var_surf_deg,
                                        font=("Helvetica", 10), width=5,
                                        bg=PALETTE["card"], fg=PALETTE["text"],
                                        buttonbackground=PALETTE["border"],
                                        relief=tk.FLAT)
        self.surf_deg_spin.pack(anchor="w", padx=10, pady=(2, 6))

        tk.Button(frame, text="Ajustar y graficar superficie",
                  command=self._plot_surface,
                  font=("Helvetica", 9, "bold"),
                  fg="white", bg="#7c4ff7",
                  activebackground="#6a3de0", activeforeground="white",
                  relief=tk.RIDGE, bd=1, padx=10, pady=6,
                  cursor="hand2").pack(fill=tk.X, padx=10, pady=3)

        # Evaluador puntual
        tk.Label(frame, text="Evaluar en un punto:",
                 font=("Helvetica", 9, "bold"),
                 fg=PALETTE["subtext"], bg=PALETTE["surface"],
                 anchor="w").pack(fill=tk.X, padx=10, pady=(6, 0))

        row_q = tk.Frame(frame, bg=PALETTE["surface"])
        row_q.pack(fill=tk.X, padx=10, pady=2)
        for lbl, var, w in [("V [m/s]:", self.var_v_query, 7),
                             ("\u03b1 [\u00b0]:", self.var_aoa_query, 6)]:
            tk.Label(row_q, text=lbl, font=("Helvetica", 8),
                     fg=PALETTE["text"], bg=PALETTE["surface"]).pack(side=tk.LEFT)
            tk.Entry(row_q, textvariable=var, font=("Helvetica", 9), width=w,
                     bg=PALETTE["card"], fg=PALETTE["text"],
                     insertbackground=PALETTE["text"],
                     relief=tk.FLAT, bd=3).pack(side=tk.LEFT, padx=3)

        row_q2 = tk.Frame(frame, bg=PALETTE["surface"])
        row_q2.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(row_q2, text="\u03c1 [kg/m\u00b3]:", font=("Helvetica", 8),
                 fg=PALETTE["text"], bg=PALETTE["surface"]).pack(side=tk.LEFT)
        tk.Entry(row_q2, textvariable=self.var_rho_query,
                 font=("Helvetica", 9), width=7,
                 bg=PALETTE["card"], fg=PALETTE["text"],
                 insertbackground=PALETTE["text"],
                 relief=tk.FLAT, bd=3).pack(side=tk.LEFT, padx=3)
        tk.Label(row_q2, text="S [m\u00b2]: (usa Sref arriba)",
                 font=("Helvetica", 7), fg=PALETTE["subtext"],
                 bg=PALETTE["surface"]).pack(side=tk.LEFT, padx=4)

        tk.Button(frame, text="Calcular",
                  command=self._evaluate_surface,
                  font=("Helvetica", 9, "bold"),
                  fg="white", bg=PALETTE["accent"],
                  activebackground="#3a7ae0", activeforeground="white",
                  relief=tk.RIDGE, bd=1, padx=6, pady=4,
                  cursor="hand2").pack(anchor="w", padx=10, pady=2)

        self.query_result_lbl = tk.Label(frame, textvariable=self.var_query_result,
                                          font=("Courier", 10, "bold"),
                                          fg=PALETTE["accent3"],
                                          bg=PALETTE["surface"],
                                          anchor="w", wraplength=260,
                                          justify=tk.LEFT)
        self.query_result_lbl.pack(fill=tk.X, padx=10, pady=(0, 8))

        # ── Botones de acción ──
        self._section(frame, "  Acciones", W)

        btn_data = [
            ("Generar grafica",         self._plot,                PALETTE["accent"],  "white"),
            ("Exportar grafica",        self._export_fig,          PALETTE["accent3"], PALETTE["bg"]),
            ("Graficar error vs AoA",   self._plot_error_analysis, "#7c4ff7",          "white"),
            ("Exportar spline CL(a)",   self._export_spline,       "#2ecc71",          PALETTE["bg"]),
            ("Ver tabla de datos",      self._show_table,          PALETTE["warning"], PALETTE["bg"]),
        ]
        for label, cmd, bg, fg in btn_data:
            tk.Button(frame, text=label, command=cmd,
                      font=("Helvetica", 9, "bold"),
                      fg=fg, bg=bg,
                      activebackground=bg, activeforeground=fg,
                      relief=tk.RIDGE, bd=1,
                      padx=10, pady=6, cursor="hand2").pack(
                          fill=tk.X, padx=10, pady=3)

    def _build_plot_area(self, parent):
        # Notebook: Gráfica + Residuales + Info
        self.notebook = ttk.Notebook(parent, style="Dark.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1 — Gráfica principal
        tab_plot = tk.Frame(self.notebook, bg=PALETTE["bg"])
        self.notebook.add(tab_plot, text="  Gráfica principal  ")

        self.fig = Figure(figsize=(10, 6), facecolor=PALETTE["bg"])
        self.ax  = self.fig.add_subplot(111)
        self._style_axes(self.ax)
        self.ax.text(0.5, 0.5, "Cargue un CSV y configure los parámetros",
                     ha="center", va="center",
                     color=PALETTE["subtext"], fontsize=13,
                     transform=self.ax.transAxes)

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=tab_plot)
        self.canvas_plot.draw()
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_plot, tab_plot)
        toolbar.update()
        toolbar.configure(background=PALETTE["card"])

        # Tab 2 — Residuales
        tab_resid = tk.Frame(self.notebook, bg=PALETTE["bg"])
        self.notebook.add(tab_resid, text="  Residuales  ")

        self.fig_r = Figure(figsize=(10, 6), facecolor=PALETTE["bg"])
        self.ax_r  = self.fig_r.add_subplot(111)
        self._style_axes(self.ax_r)

        self.canvas_resid = FigureCanvasTkAgg(self.fig_r, master=tab_resid)
        self.canvas_resid.draw()
        self.canvas_resid.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tab 3 — Estadísticas
        tab_stats = tk.Frame(self.notebook, bg=PALETTE["surface"])
        self.notebook.add(tab_stats, text="  Estadísticas  ")

        self.stats_text = tk.Text(tab_stats, font=("Courier", 10),
                                  bg=PALETTE["card"], fg=PALETTE["text"],
                                  insertbackground=PALETTE["text"],
                                  relief=tk.FLAT, padx=12, pady=12)
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        # Tab 4 — Superficie 3D
        tab_surf = tk.Frame(self.notebook, bg=PALETTE["bg"])
        self.notebook.add(tab_surf, text="  Superficie f(V,\u03b1)  ")

        self.fig_s = Figure(figsize=(10, 7), facecolor=PALETTE["bg"])
        self.ax_s3d = self.fig_s.add_subplot(121, projection="3d")
        self.ax_sct = self.fig_s.add_subplot(122)
        self.fig_s.subplots_adjust(wspace=0.35)
        self._style_axes(self.ax_sct)

        self.canvas_surf = FigureCanvasTkAgg(self.fig_s, master=tab_surf)
        self.canvas_surf.draw()
        self.canvas_surf.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_s = NavigationToolbar2Tk(self.canvas_surf, tab_surf)
        toolbar_s.update()
        toolbar_s.configure(background=PALETTE["card"])

        # Caja de ecuación debajo de la gráfica
        eq_frame = tk.Frame(tab_surf, bg=PALETTE["card"], height=72)
        eq_frame.pack(fill=tk.X, side=tk.BOTTOM)
        eq_frame.pack_propagate(False)
        tk.Label(eq_frame, text="ECUACIÓN  f(V, \u03b1) :",
                 font=("Courier", 9, "bold"),
                 fg=PALETTE["accent"], bg=PALETTE["card"],
                 anchor="w").pack(fill=tk.X, padx=12, pady=(6, 0))
        self.eq_display = tk.Label(eq_frame, text="\u2014  Ajusta la superficie para ver la ecuación  \u2014",
                                    font=("Courier", 9),
                                    fg=PALETTE["accent3"], bg=PALETTE["card"],
                                    anchor="w", justify=tk.LEFT,
                                    wraplength=1300)
        self.eq_display.pack(fill=tk.X, padx=20, pady=(0, 6))

        # Tab 5 — Análisis de Error
        tab_err = tk.Frame(self.notebook, bg=PALETTE["bg"])
        self.notebook.add(tab_err, text="  Error vs AoA  ")

        self.fig_e = Figure(figsize=(10, 7), facecolor=PALETTE["bg"])
        self.canvas_err = FigureCanvasTkAgg(self.fig_e, master=tab_err)
        self.canvas_err.draw()
        self.canvas_err.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_e = NavigationToolbar2Tk(self.canvas_err, tab_err)
        toolbar_e.update()
        toolbar_e.configure(background=PALETTE["card"])

        # Controles de la pestaña de error
        ctrl_err = tk.Frame(tab_err, bg=PALETTE["card"], height=44)
        ctrl_err.pack(fill=tk.X, side=tk.BOTTOM)
        ctrl_err.pack_propagate(False)

        tk.Label(ctrl_err, text="Tipo de error:",
                 font=("Helvetica", 9), fg=PALETTE["text"],
                 bg=PALETTE["card"]).pack(side=tk.LEFT, padx=(12, 4), pady=10)

        self.var_err_type = tk.StringVar(value="Absoluto [N]")
        ttk.Combobox(ctrl_err, textvariable=self.var_err_type,
                     values=["Absoluto [N]", "Porcentual [%] (excluye AoA=0)"],
                     style="Dark.TCombobox", state="readonly", width=32).pack(
                         side=tk.LEFT, pady=10)

        tk.Label(ctrl_err, text="  Variable:",
                 font=("Helvetica", 9), fg=PALETTE["text"],
                 bg=PALETTE["card"]).pack(side=tk.LEFT, padx=(16, 4))

        self.var_err_col = tk.StringVar(value=list(FORCE_OPTIONS.keys())[0])
        ttk.Combobox(ctrl_err, textvariable=self.var_err_col,
                     values=list(FORCE_OPTIONS.keys()),
                     style="Dark.TCombobox", state="readonly", width=32).pack(
                         side=tk.LEFT, pady=10)

        tk.Button(ctrl_err, text="Graficar error",
                  command=self._plot_error_analysis,
                  font=("Helvetica", 9, "bold"),
                  fg="white", bg="#7c4ff7",
                  activebackground="#6a3de0", activeforeground="white",
                  relief=tk.RIDGE, bd=1, padx=10, pady=4,
                  cursor="hand2").pack(side=tk.LEFT, padx=12, pady=8)

        tk.Label(ctrl_err,
                 text="Requiere ajuste de superficie activo",
                 font=("Helvetica", 8), fg=PALETTE["subtext"],
                 bg=PALETTE["card"]).pack(side=tk.LEFT, padx=4)

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=PALETTE["card"], height=26)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)
        tk.Label(bar, textvariable=self.var_status,
                 font=("Helvetica", 9), fg=PALETTE["subtext"],
                 bg=PALETTE["card"], anchor="w").pack(side=tk.LEFT, padx=12)

    # ─── WIDGETS HELPERS ─────────────────────────────────────────────────────

    def _section(self, parent, title, width=280):
        tk.Label(parent, text=title,
                 font=("Helvetica", 10, "bold"),
                 fg=PALETTE["accent"], bg=PALETTE["surface"],
                 anchor="w").pack(fill=tk.X, padx=10, pady=(14, 2))
        ttk.Separator(parent, orient="horizontal").pack(
            fill=tk.X, padx=10, pady=(0, 6))

    def _info_label(self, parent, var, width):
        tk.Label(parent, textvariable=var, wraplength=width-20,
                 font=("Helvetica", 8), fg=PALETTE["subtext"],
                 bg=PALETTE["surface"], anchor="w",
                 justify=tk.LEFT).pack(fill=tk.X, padx=10, pady=(0, 6))

    def _labeled_entry(self, parent, label, var, width):
        tk.Label(parent, text=label, font=("Helvetica", 9),
                 fg=PALETTE["text"], bg=PALETTE["surface"],
                 anchor="w").pack(fill=tk.X, padx=10)
        e = tk.Entry(parent, textvariable=var,
                     font=("Helvetica", 10),
                     bg=PALETTE["card"], fg=PALETTE["text"],
                     insertbackground=PALETTE["text"],
                     relief=tk.FLAT, bd=4)
        e.pack(fill=tk.X, padx=10, pady=(2, 6))

    def _labeled_combo(self, parent, label, var, values, width,
                       is_index=False, var_int=None):
        tk.Label(parent, text=label, font=("Helvetica", 9),
                 fg=PALETTE["text"], bg=PALETTE["surface"],
                 anchor="w").pack(fill=tk.X, padx=10)
        if is_index and var_int is not None:
            # Proxy string → int
            proxy = tk.StringVar(value=values[var_int.get() - 1])
            def _on_select(event):
                var_int.set(values.index(proxy.get()) + 1)
            c = ttk.Combobox(parent, textvariable=proxy,
                              style="Dark.TCombobox",
                              values=values, state="readonly", width=36)
            c.bind("<<ComboboxSelected>>", _on_select)
        else:
            c = ttk.Combobox(parent, textvariable=var,
                              style="Dark.TCombobox",
                              values=values, state="readonly", width=36)
        c.pack(padx=10, pady=(2, 6))
        c.bind("<<ComboboxSelected>>", lambda e: self._plot())

    def _checkbox(self, parent, text, var):
        tk.Checkbutton(parent, text=text, variable=var,
                       font=("Helvetica", 9),
                       fg=PALETTE["text"], bg=PALETTE["surface"],
                       activebackground=PALETTE["surface"],
                       activeforeground=PALETTE["accent"],
                       selectcolor=PALETTE["card"],
                       command=self._plot,
                       anchor="w").pack(fill=tk.X, padx=14, pady=1)

    def _style_axes(self, ax):
        ax.set_facecolor(PALETTE["card"])
        ax.tick_params(colors=PALETTE["subtext"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])

    # ─── CARGA DE ARCHIVO ────────────────────────────────────────────────────

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Seleccionar CSV de SimScale",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            df = pd.read_csv(path)
            required = {"speed_ms", "aoa_deg", "density_kgm3"}
            # Aceptar formato nuevo (lift_N/drag_N) o clásico (fx/fy/fz)
            has_forces = {"fx", "fy", "fz"}.issubset(set(df.columns))
            has_aero   = {"lift_N", "drag_N"}.issubset(set(df.columns))
            if not has_forces and not has_aero:
                required = required | {"fx", "fy", "fz"}  # forzar error descriptivo
            missing  = required - set(df.columns)
            if missing:
                messagebox.showerror("Error",
                    f"Columnas requeridas no encontradas:\n{missing}")
                return

            # Filtrar solo simulaciones terminadas
            if "status" in df.columns:
                df = df[df["status"] == "FINISHED"].copy()

            self.df_raw = df
            self.var_file.set(f"{path.split('/')[-1]}\n"
                              f"{len(df)} simulaciones  |  "
                              f"V: {sorted(df['speed_ms'].unique())} m/s")

            self._apply_sref(reload=True)
            self.var_status.set(f"✔  Archivo cargado: {len(df)} simulaciones.")

        except Exception as ex:
            messagebox.showerror("Error al cargar", str(ex))

    def _apply_sref(self, reload=False):
        if self.df_raw is None:
            return
        try:
            sref = float(self.var_sref.get())
            if sref <= 0:
                raise ValueError("Sref debe ser positivo")
        except Exception:
            messagebox.showwarning("Parámetro inválido",
                                   "Ingrese un valor numérico positivo para Sref.")
            return

        self.df_work = compute_coefficients(self.df_raw, sref)

        if reload:
            self._rebuild_speed_checkboxes()
            self.force_combo["values"] = list(FORCE_OPTIONS.keys())
            if not self.var_force.get():
                self.force_combo.current(0)

        self._plot()

    def _rebuild_speed_checkboxes(self):
        for w in self.speed_frame.winfo_children():
            w.destroy()
        self.speed_vars = {}
        speeds = sorted(self.df_work["speed_ms"].unique())
        for i, spd in enumerate(speeds):
            var = tk.BooleanVar(value=True)
            color = SPEED_COLORS[i % len(SPEED_COLORS)]
            tk.Checkbutton(self.speed_frame,
                           text=f"  {spd:.0f} m/s",
                           variable=var,
                           font=("Helvetica", 10, "bold"),
                           fg=color, bg=PALETTE["surface"],
                           activebackground=PALETTE["surface"],
                           selectcolor=PALETTE["card"],
                           command=self._plot).pack(anchor="w", pady=2)
            self.speed_vars[spd] = var

    # ─── GRÁFICA ─────────────────────────────────────────────────────────────

    def _plot(self):
        if self.df_work is None:
            return

        force_key = FORCE_OPTIONS.get(self.var_force.get())
        if force_key is None:
            return

        method_key = INTERP_OPTIONS.get(self.var_method.get(), "poly")
        degree     = self.var_degree.get()

        # ── Gráfica principal ──
        self.ax.cla()
        self._style_axes(self.ax)

        if self.var_grid.get():
            self.ax.grid(True, color=PALETTE["border"], linewidth=0.6,
                         linestyle="--", alpha=0.6)

        all_residuals = {}
        stats_lines   = []
        active_speeds = []

        for i, (spd, bvar) in enumerate(self.speed_vars.items()):
            if not bvar.get():
                continue

            sub = self.df_work[self.df_work["speed_ms"] == spd].copy()
            sub = sub.dropna(subset=[force_key])

            if len(sub) < 2:
                continue

            aoa_pts = sub["aoa_deg"].values
            y_pts   = sub[force_key].values
            color   = SPEED_COLORS[i % len(SPEED_COLORS)]
            active_speeds.append(spd)

            # Puntos reales
            if self.var_show_raw.get():
                self.ax.scatter(aoa_pts, y_pts, color=color, zorder=5,
                                s=55, edgecolors="white", linewidths=0.6,
                                label=f"V={spd:.0f} m/s (datos)")

            # Interpolación
            if self.var_show_interp.get() and len(sub) >= max(2, degree + 1):
                try:
                    x_f, y_f, lbl, coeffs = do_interpolation(
                        aoa_pts, y_pts, method_key, degree)
                    self.ax.plot(x_f, y_f, color=color, linewidth=2.2,
                                 linestyle="-",
                                 label=f"V={spd:.0f} m/s ({lbl})")

                    # Residuales
                    y_interp_pts = np.interp(aoa_pts, x_f, y_f)
                    residuals    = y_pts - y_interp_pts
                    all_residuals[spd] = (aoa_pts, residuals, color)

                    # R²
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y_pts - y_pts.mean())**2)
                    r2     = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0
                    rmse   = np.sqrt(np.mean(residuals**2))

                    # Ecuación
                    if self.var_show_eq.get() and method_key == "poly" and coeffs is not None:
                        terms = []
                        for k, c in enumerate(coeffs):
                            pw = degree - k
                            if abs(c) < 1e-10:
                                continue
                            sgn  = "+" if c >= 0 else "−"
                            term = f"{abs(c):.4g}·α^{pw}" if pw > 1 else \
                                   (f"{abs(c):.4g}·α" if pw == 1 else f"{abs(c):.4g}")
                            terms.append(f"{sgn} {term}")
                        eq_str = " ".join(terms).lstrip("+ ").replace("+ −", "− ")
                        stats_lines.append(
                            f"V = {spd:.0f} m/s  |  R² = {r2:.5f}  |  RMSE = {rmse:.5e}\n"
                            f"  f(α) = {eq_str}\n")
                    else:
                        stats_lines.append(
                            f"V = {spd:.0f} m/s  |  R² = {r2:.5f}  |  RMSE = {rmse:.5e}\n")

                except Exception as ex:
                    self.var_status.set(f"⚠ Interpolación fallida para V={spd}: {ex}")

        # Etiquetas y leyenda
        ylabel = self.var_force.get().split("—")[-1].strip()
        self.ax.set_xlabel("Ángulo de ataque α [°]", fontsize=11)
        self.ax.set_ylabel(ylabel, fontsize=11)
        title_txt = f"Datos SimScale — {ylabel}"
        if method_key == "poly":
            title_txt += f"  |  Interpolación polinomial grado {degree}"
        else:
            title_txt += "  |  Spline cúbico"
        self.ax.set_title(title_txt, fontsize=12, pad=10)
        if active_speeds:
            leg = self.ax.legend(facecolor=PALETTE["surface"],
                                 edgecolor=PALETTE["border"],
                                 labelcolor=PALETTE["text"],
                                 fontsize=9, framealpha=0.9)

        self.canvas_plot.draw()

        # ── Gráfica de residuales ──
        self.ax_r.cla()
        self._style_axes(self.ax_r)
        if self.var_grid.get():
            self.ax_r.grid(True, color=PALETTE["border"], linewidth=0.6,
                           linestyle="--", alpha=0.6)

        if all_residuals:
            for spd, (aoa_r, res, color) in all_residuals.items():
                self.ax_r.stem(aoa_r, res, linefmt=color,
                               markerfmt="o", basefmt="grey")
                self.ax_r.axhline(0, color=PALETTE["subtext"],
                                  linewidth=0.8, linestyle="--")
            self.ax_r.set_xlabel("Ángulo de ataque α [°]", fontsize=11)
            self.ax_r.set_ylabel("Residual  (dato − interpolación)", fontsize=11)
            self.ax_r.set_title("Residuales de la interpolación", fontsize=12, pad=10)

        self.canvas_resid.draw()

        # ── Panel de estadísticas ──
        self.stats_text.delete("1.0", tk.END)
        header = (
            "══════════════════════════════════════════════════\n"
            f"  Variable  : {self.var_force.get()}\n"
            f"  Método    : {self.var_method.get()}\n"
            f"  Sref      : {self.var_sref.get()} m²\n"
            "══════════════════════════════════════════════════\n\n"
        )
        self.stats_text.insert(tk.END, header)
        for line in stats_lines:
            self.stats_text.insert(tk.END, line + "\n")

        if not stats_lines:
            self.stats_text.insert(tk.END, "No hay datos suficientes para interpolar.\n")

        self.var_status.set(
            f"✔  Gráfica actualizada — "
            f"{len(active_speeds)} velocidad(es) activa(s).")

    # ─── EXPORTAR ────────────────────────────────────────────────────────────

    def _export_fig(self):
        if self.df_work is None:
            messagebox.showinfo("Sin datos", "Cargue un archivo primero.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                       ("SVG", "*.svg"), ("All", "*.*")],
            title="Guardar gráfica")
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight",
                             facecolor=PALETTE["bg"])
            self.var_status.set(f"✔  Gráfica guardada en: {path}")
            messagebox.showinfo("Éxito", f"Gráfica guardada:\n{path}")


    # ─── SUPERFICIE BIVARIABLE f(V, α) ───────────────────────────────────────

    def _plot_surface(self):
        """Ajusta polinomio 2D y muestra superficie 3D + mapa de contorno."""
        if self.df_work is None:
            messagebox.showinfo("Sin datos", "Cargue un archivo primero.")
            return

        col_label = self.var_surf_col.get()
        col       = FORCE_OPTIONS.get(col_label)
        deg       = int(self.var_surf_deg.get())

        method_key = SURF_METHODS.get(self.var_surf_method.get(), "physical")
        sub = self.df_work.dropna(subset=[col])
        V_arr   = sub["speed_ms"].values.astype(float)
        aoa_arr = sub["aoa_deg"].values.astype(float)
        y_arr   = sub[col].values.astype(float)
        rho_arr = sub["density_kgm3"].values.astype(float)

        try:
            sref = float(self.var_sref.get())
            fit_r = fit_surface(V_arr, aoa_arr, y_arr, rho_arr,
                                method_key, deg, col, sref=sref)
        except Exception as ex:
            messagebox.showerror("Error en ajuste", str(ex))
            return

        self.surf_fit = fit_r
        r2   = fit_r["r2"]
        rmse = fit_r["rmse"]

        # Malla de evaluación usando eval_surface
        V_lin   = np.linspace(V_arr.min(),   V_arr.max(),   60)
        aoa_lin = np.linspace(aoa_arr.min(), aoa_arr.max(), 60)
        VV, AA  = np.meshgrid(V_lin, aoa_lin)
        rho_mean = rho_arr.mean()
        ZZ = np.array([eval_surface(fit_r, v, a, rho_mean)
                       for v, a in zip(VV.ravel(), AA.ravel())]).reshape(VV.shape)

        # ── Limpiar figure y recrear con GridSpec (3D | contorno / CL abajo) ──
        self.fig_s.clf()
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2,
                      height_ratios=[3, 1.2],
                      hspace=0.45, wspace=0.40,
                      figure=self.fig_s)
        self.ax_s3d = self.fig_s.add_subplot(gs[0, 0], projection="3d")
        self.ax_sct = self.fig_s.add_subplot(gs[0, 1])
        self.ax_cl  = self.fig_s.add_subplot(gs[1, :])   # CL(α) abajo
        self._query_marker = []

        # ── Gráfica 3D ──────────────────────────────────────────────────────
        self.ax_s3d.set_facecolor(PALETTE["card"])
        self.ax_s3d.tick_params(colors=PALETTE["subtext"], labelsize=7)
        self.ax_s3d.xaxis.pane.fill = False
        self.ax_s3d.yaxis.pane.fill = False
        self.ax_s3d.zaxis.pane.fill = False

        surf = self.ax_s3d.plot_surface(VV, AA, ZZ, cmap="plasma",
                                         alpha=0.85, linewidth=0,
                                         antialiased=True)
        self.ax_s3d.scatter(V_arr, aoa_arr, y_arr,
                            color=PALETTE["accent3"], s=30, zorder=5)
        self.ax_s3d.set_xlabel("V [m/s]",      fontsize=8, color=PALETTE["text"], labelpad=6)
        self.ax_s3d.set_ylabel("\u03b1 [\u00b0]", fontsize=8, color=PALETTE["text"], labelpad=6)
        self.ax_s3d.set_zlabel(col_label[:14],  fontsize=7, color=PALETTE["text"], labelpad=4)
        self.ax_s3d.set_title(
            f"Superficie 3D  |  Polinomio grado {deg}\nR\u00b2={r2:.5f}   RMSE={rmse:.4f}",
            fontsize=9, color=PALETTE["text"], pad=6)
        cb3d = self.fig_s.colorbar(surf, ax=self.ax_s3d, shrink=0.45,
                                    aspect=12, pad=0.10)
        cb3d.ax.tick_params(labelsize=7, colors=PALETTE["subtext"])

        # ── Mapa de contorno ─────────────────────────────────────────────────
        self._style_axes(self.ax_sct)
        self.ax_sct.set_facecolor(PALETTE["card"])

        cf = self.ax_sct.contourf(VV, AA, ZZ, levels=20, cmap="plasma", alpha=0.9)
        ct = self.ax_sct.contour( VV, AA, ZZ, levels=20,
                                   colors="white", linewidths=0.4, alpha=0.3)
        self.ax_sct.clabel(ct, inline=True, fontsize=6, fmt="%.1f", colors="white")
        self.ax_sct.scatter(V_arr, aoa_arr, c=PALETTE["accent3"],
                            s=25, zorder=5, label="Datos CFD")
        self.ax_sct.set_xlabel("V [m/s]", fontsize=10, color=PALETTE["text"])
        self.ax_sct.set_ylabel("\u03b1 [\u00b0]", fontsize=10, color=PALETTE["text"])
        self.ax_sct.set_title("Mapa de contorno", fontsize=10, color=PALETTE["text"])
        cbct = self.fig_s.colorbar(cf, ax=self.ax_sct, shrink=0.9,
                                    aspect=20, pad=0.04)
        cbct.ax.tick_params(labelsize=7, colors=PALETTE["subtext"])
        self.ax_sct.legend(facecolor=PALETTE["surface"],
                            edgecolor=PALETTE["border"],
                            labelcolor=PALETTE["text"], fontsize=8)

        # ── Curva CL(α) en panel inferior ────────────────────────────────────
        method_k = SURF_METHODS.get(self.var_surf_method.get(), "")
        self._style_axes(self.ax_cl)
        self.ax_cl.set_facecolor(PALETTE["card"])
        self.ax_cl.grid(True, color=PALETTE["border"],
                        linewidth=0.5, linestyle="--", alpha=0.5)

        aoa_fine = np.linspace(aoa_arr.min(), aoa_arr.max(), 300)

        if method_k == "spline_cl":
            cl_fine  = fit_r["spline_cl"](aoa_fine)
            cl_data  = fit_r["cl_knots"]
            aoa_data = fit_r["aoa_knots"]
            model_lbl = "Spline c\u00fabico"
        elif method_k == "physical":
            X1d      = np.column_stack([aoa_fine**k
                        for k in range(len(fit_r["coeffs_1d"]))])
            cl_fine  = X1d @ fit_r["coeffs_1d"]
            q_ref    = fit_r.get("q_arr", np.ones_like(y_arr))
            sref_ref = fit_r.get("sref", 1.0)
            cl_data  = y_arr / np.where(q_ref * sref_ref > 0,
                                         q_ref * sref_ref, 1e-9)
            aoa_data = aoa_arr
            model_lbl = f"Polin. grado {deg}"
        else:
            # Para poly2d / rbf: calcular CL aproximado a densidad media
            q_mean = 0.5 * rho_arr.mean() * V_arr.mean()**2
            sref_v = fit_r.get("sref", 1.0)
            cl_data  = y_arr / max(q_mean * sref_v, 1e-9)
            aoa_data = aoa_arr
            cl_fine  = np.interp(aoa_fine,
                                  np.sort(aoa_data),
                                  cl_data[np.argsort(aoa_data)])
            model_lbl = "Aproximado"

        self.ax_cl.plot(aoa_fine, cl_fine,
                        color=PALETTE["accent"], linewidth=2.2,
                        label=model_lbl, zorder=3)
        # Datos por velocidad con colores distintos
        for idx_s, spd in enumerate(sorted(np.unique(V_arr))):
            m_s = V_arr == spd
            self.ax_cl.scatter(aoa_arr[m_s],
                               y_arr[m_s] / np.where(
                                   (0.5*rho_arr[m_s]*spd**2 * fit_r.get("sref",1.0)) > 0,
                                   0.5*rho_arr[m_s]*spd**2 * fit_r.get("sref",1.0), 1e-9),
                               color=SPEED_COLORS[idx_s % len(SPEED_COLORS)],
                               s=22, zorder=5, alpha=0.85,
                               label=f"V={spd:.0f} m/s")

        self.ax_cl.set_xlabel("\u03b1 [\u00b0]", fontsize=9,
                              color=PALETTE["text"])
        self.ax_cl.set_ylabel("CL", fontsize=9, color=PALETTE["text"])
        self.ax_cl.set_title(
            f"Curva CL(\u03b1)  —  {model_lbl}  "
            f"(confirma independencia de V)",
            fontsize=9, color=PALETTE["text"])
        self.ax_cl.legend(facecolor=PALETTE["surface"],
                          edgecolor=PALETTE["border"],
                          labelcolor=PALETTE["text"],
                          fontsize=7, ncol=6, loc="upper left")

        self.canvas_surf.draw()

        # Mostrar ecuación en la barra inferior de la pestaña
        try:
            self.eq_display.config(text=eq_str)
        except Exception:
            pass

        # Cambiar a la pestaña de superficie
        self.notebook.select(3)

        # Estadísticas detalladas con error por velocidad
        col_short = col_label.split("—")[0].strip()
        eq_str    = format_surface_equation(fit_r, col_short)
        method_lbl = self.var_surf_method.get()

        self.stats_text.delete("1.0", tk.END)
        sep = "=" * 54
        self.stats_text.insert(tk.END,
            f"{sep}\n"
            f"  Ajuste  : f(V, \u03b1) = {col_label[:38]}\n"
            f"  Metodo  : {method_lbl[:50]}\n"
            f"  Grado   : {deg}\n"
            f"  Puntos  : {len(y_arr)}\n"
            f"  R\u00b2      : {r2:.6f}\n"
            f"  RMSE    : {rmse:.6f}\n"
            f"{sep}\n\n"
        )
        self.stats_text.insert(tk.END, "  ERROR POR VELOCIDAD:\n")
        for spd, edata in sorted(fit_r["errs_by_speed"].items()):
            bar_len = min(int(edata["err_pct"] / 2), 30)
            bar     = "\u2588" * bar_len
            quality = ("OK" if edata["err_pct"] < 5
                       else ("~" if edata["err_pct"] < 15 else "!!"))
            self.stats_text.insert(tk.END,
                f"  V={spd:5.0f} m/s  RMSE={edata['rmse']:8.4f}  "
                f"err={edata['err_pct']:5.1f}%  {bar} {quality}\n"
            )
        # Physical insight note for physical method
        method_k = SURF_METHODS.get(self.var_surf_method.get(), "")
        if method_k in ("physical", "spline_cl"):
            sref_v = fit_r.get("sref", 1.0)
            self.stats_text.insert(tk.END,
                "\n  INSIGHT FISICO:\n"
                "  CL depende SOLO de alpha, no de V ni de rho.\n"
                "  Por lo tanto NO necesitas simular multiples densidades:\n"
                "  con una sola rho obtienes CL(alpha) valido para cualquier\n"
                "  altitud, velocidad o fluido (si Re es similar).\n"
                f"  Sref usado en el ajuste: {sref_v} m^2\n"
                "\n"
                "  ECUACION COMPLETA:\n"
                "  Lift = CL(alpha) x (1/2 x rho x V^2) x S\n\n"
            )
        # ── Ecuación / modelo de CL(α) ──────────────────────────────────────
        sep2 = "-" * 54
        self.stats_text.insert(tk.END, f"\n  {sep2}\n")
        self.stats_text.insert(tk.END,
            "  MODELO  CL(\u03b1)  \u2192  Lift = CL(\u03b1) \u00d7 \u00bdρV\u00b2 \u00d7 S\n"
            f"  {sep2}\n\n")

        if method_k == "spline_cl" and "aoa_knots" in fit_r:
            # Tabla completa de nodos de la spline
            self.stats_text.insert(tk.END,
                "  Tipo   : Spline c\u00fabico (interpolaci\u00f3n exacta en nodos)\n"
                f"  Nodos  : {len(fit_r['aoa_knots'])}\n"
                f"  \u03b1 rango: [{fit_r['aoa_knots'].min():.1f}\u00b0"
                f", {fit_r['aoa_knots'].max():.1f}\u00b0]\n"
                f"  CL rango: [{fit_r['cl_knots'].min():.6f}"
                f", {fit_r['cl_knots'].max():.6f}]\n\n"
                f"  {'':>2}\u03b1 [\u00b0]{'':>4}CL{'':>10}Lift@V=190,\u03c1=1.225,S=1\n"
                f"  {'':>2}{'-'*48}\n"
            )
            for a, cl in zip(fit_r["aoa_knots"], fit_r["cl_knots"]):
                q_ex   = 0.5 * 1.225 * 190.0**2
                sref_v = fit_r.get("sref", 1.0)
                lift_ex = cl * q_ex * sref_v
                self.stats_text.insert(tk.END,
                    f"  {a:8.2f}     {cl:.7f}    {lift_ex:9.3f} N\n")

        elif method_k == "physical" and "coeffs_1d" in fit_r:
            coeffs = fit_r["coeffs_1d"]
            deg_cl = len(coeffs) - 1
            # Build readable polynomial string
            terms = []
            for k, c in enumerate(coeffs):
                if abs(c) < 1e-16:
                    continue
                sign  = "+" if c >= 0 else "\u2212"
                a_str = ("" if k == 0
                         else ("\u00b7\u03b1" if k == 1
                               else f"\u00b7\u03b1^{k}"))
                terms.append((sign, f"{abs(c):.5e}{a_str}"))
            poly_line = "  CL(\u03b1) = "
            poly_lines = []
            for i, (sign, term) in enumerate(terms):
                chunk = (term if i == 0 and sign == "+"
                         else (f"\u2212{term}" if i == 0
                               else f" {sign} {term}"))
                if len(poly_line) + len(chunk) > 70:
                    poly_lines.append(poly_line)
                    poly_line = "           " + chunk.lstrip()
                else:
                    poly_line += chunk
            poly_lines.append(poly_line)

            self.stats_text.insert(tk.END,
                f"  Tipo   : Polinomio grado {deg_cl}\n\n")
            for pl in poly_lines:
                self.stats_text.insert(tk.END, "  " + pl + "\n")
            self.stats_text.insert(tk.END,
                f"\n  {'':>2}\u03b1 [\u00b0]{'':>4}CL{'':>10}Lift@V=190,\u03c1=1.225,S=1\n"
                f"  {'':>2}{'-'*48}\n"
            )
            aoa_sample = np.linspace(
                fit_r.get("aoa_mu", 0) - fit_r.get("aoa_s", 20),
                fit_r.get("aoa_mu", 0) + fit_r.get("aoa_s", 20), 10)
            aoa_sample = np.clip(aoa_sample, 0, 40)
            q_ex = 0.5 * 1.225 * 190.0**2
            sref_v = fit_r.get("sref", 1.0)
            for a in aoa_sample:
                cl_v = float(sum(c * a**k for k, c in enumerate(coeffs)))
                self.stats_text.insert(tk.END,
                    f"  {a:8.2f}     {cl_v:.7f}    {cl_v*q_ex*sref_v:9.3f} N\n")

        else:
            self.stats_text.insert(tk.END, "  " + eq_str.replace("\n", "\n  ") + "\n")

        self.stats_text.insert(tk.END, "\n")

        self.ax_sct.set_title(
            f"Mapa de contorno  |  R\u00b2={r2:.5f}  RMSE={rmse:.3f}",
            fontsize=9, color=PALETTE["text"])
        self.var_status.set(
            f"Superficie ajustada — Poly2D grado {deg} | R²={r2:.5f} | RMSE={rmse:.4f}")

    def _evaluate_surface(self):
        """Evalúa la superficie ajustada en el punto (V, α) ingresado."""
        if self.surf_fit is None:
            messagebox.showinfo("Sin ajuste",
                                "Primero ajusta la superficie con el botón correspondiente.")
            return
        try:
            V_val   = float(self.var_v_query.get())
            aoa_val = float(self.var_aoa_query.get())
        except ValueError:
            messagebox.showwarning("Valor inválido", "Ingrese valores numéricos válidos.")
            return

        try:
            rho = float(self.var_rho_query.get())
        except Exception:
            rho = self.surf_fit.get("rho_mean", 1.225)
        result = eval_surface(self.surf_fit, V_val, aoa_val, rho)

        col_label = self.var_surf_col.get().split("—")[0].strip()
        sref = self.surf_fit.get("sref", 1.0)
        try:
            sref_ui = float(self.var_sref.get())
        except Exception:
            sref_ui = sref
        # Show breakdown for physical method
        method = self.surf_fit.get("method","")
        if method == "physical":
            q_show = 0.5 * rho * V_val**2
            # CL at this aoa
            coeffs = self.surf_fit["coeffs_1d"]
            cl_show = float(sum(c * aoa_val**k for k, c in enumerate(coeffs)))
            detail = (f"CL({aoa_val:.1f}\u00b0) = {cl_show:.5f}\n"
                      f"q = \u00bd\u03c1V\u00b2 = {q_show:.1f} Pa\n"
                      f"Lift = CL \u00d7 q \u00d7 S = {result:.3f} N")
        else:
            detail = f"{result:.4f}"
        self.var_query_result.set(detail)
        self.var_status.set(
            f"f(V={V_val} m/s, \u03b1={aoa_val}\u00b0, \u03c1={rho}) = {result:.4f}  [{col_label}]")

        # Marcar el punto en el mapa de contorno si existe
        try:
            for artist in getattr(self, "_query_marker", []):
                artist.remove()
        except Exception:
            pass
        m = self.ax_sct.plot(V_val, aoa_val, marker="*", markersize=16,
                              color=PALETTE["warning"], zorder=10,
                              label=f"({V_val},{aoa_val}°)→{result:.2f}")
        t = self.ax_sct.annotate(f"{result:.2f}",
                                  xy=(V_val, aoa_val),
                                  xytext=(8, 8), textcoords="offset points",
                                  fontsize=9, color=PALETTE["warning"],
                                  fontweight="bold")
        self._query_marker = m + [t]
        self.canvas_surf.draw()


    # ─── ANÁLISIS DE ERROR (Calculado vs SimScale) ───────────────────────────

    def _plot_error_analysis(self):
        """Grafica el error absoluto o porcentual por velocidad, igual que
        los scripts independientes del usuario pero integrado en la GUI."""
        if self.surf_fit is None:
            messagebox.showinfo("Sin ajuste",
                "Primero ajusta la superficie en la pestaña Superficie f(V,α).")
            return
        if self.df_work is None:
            messagebox.showinfo("Sin datos", "Cargue un archivo primero.")
            return

        col_label = self.var_err_col.get()
        col       = FORCE_OPTIONS.get(col_label)
        err_type  = self.var_err_type.get()
        pct_mode  = "Porcentual" in err_type

        sub = self.df_work.dropna(subset=[col]).copy()

        # ── Calcular predicción usando el ajuste activo ──────────────────────
        try:
            rho_col = sub["density_kgm3"].values
            V_col   = sub["speed_ms"].values
            aoa_col = sub["aoa_deg"].values
            pred    = np.array([
                eval_surface(self.surf_fit, v, a, r)
                for v, a, r in zip(V_col, aoa_col, rho_col)
            ])
        except Exception as ex:
            messagebox.showerror("Error al predecir", str(ex))
            return

        real = sub[col].values
        sub  = sub.copy()
        sub["_pred"] = pred
        sub["_err"]  = pred - real

        if pct_mode:
            # Excluir AoA = 0 (división por cero)
            sub = sub[sub["aoa_deg"] != 0.0].copy()
            sub["_err_pct"] = ((sub["_pred"] - sub[col]) /
                                sub[col].abs().clip(lower=1e-9)) * 100

        # ── Layout dinámico ──────────────────────────────────────────────────
        speeds = sorted(sub["speed_ms"].unique())
        n      = len(speeds)
        cols_n = min(3, n)
        rows_n = int(np.ceil(n / cols_n))

        self.fig_e.clf()
        self.fig_e.patch.set_facecolor(PALETTE["bg"])

        title = ("Error Porcentual de Lift vs AoA  (excluye AoA=0)"
                 if pct_mode
                 else f"Error Absoluto ({col_label.split('—')[0].strip()})  =  Calculado − SimScale")
        self.fig_e.suptitle(title, fontsize=11,
                             color=PALETTE["text"], y=0.98)

        axs = []
        for idx, spd in enumerate(speeds):
            ax = self.fig_e.add_subplot(rows_n, cols_n, idx + 1)
            self._style_axes(ax)
            ax.set_facecolor(PALETTE["card"])
            ax.grid(True, color=PALETTE["border"],
                    linewidth=0.6, linestyle="--", alpha=0.6)

            df_s = sub[sub["speed_ms"] == spd].sort_values("aoa_deg")
            aoa_s = df_s["aoa_deg"].values

            if pct_mode:
                y_vals = df_s["_err_pct"].values
                color  = "#c44ff7"
                ylabel = "Error relativo [%]"
            else:
                y_vals = df_s["_err"].values
                color  = PALETTE["accent2"]
                ylabel = "Error [N]  (calc − sim)"

            ax.plot(aoa_s, y_vals, marker="o", color=color,
                    linewidth=1.6, markersize=5)
            ax.axhline(0, color=PALETTE["subtext"],
                       linestyle="--", linewidth=1)

            # Anotación: RMSE y err_max
            rmse_s   = np.sqrt(np.mean((df_s["_pred"].values - df_s[col].values)**2))
            max_err  = np.max(np.abs(y_vals))
            ax.set_title(f"V = {spd:.0f} m/s  |  RMSE={rmse_s:.3f}",
                         fontsize=8, color=PALETTE["text"])
            ax.set_xlabel("AoA [°]", fontsize=8, color=PALETTE["subtext"])
            ax.set_ylabel(ylabel,    fontsize=7, color=PALETTE["subtext"])
            ax.tick_params(labelsize=7)

            # Banda ±5% / ±max referencia
            if pct_mode:
                ax.axhspan(-5, 5, color=PALETTE["accent3"],
                           alpha=0.07, label="±5%")
            axs.append(ax)

        self.fig_e.tight_layout(rect=[0, 0, 1, 0.95])
        self.canvas_err.draw()

        # Cambiar a la pestaña de error
        self.notebook.select(4)
        self.var_status.set(
            f"Error graficado — {len(speeds)} velocidades  |  "
            f"{'Porcentual' if pct_mode else 'Absoluto'}")


    # ─── EXPORTAR SPLINE CL(α) ───────────────────────────────────────────────

    def _export_spline(self):
        """Exporta los coeficientes del spline cúbico en varios formatos."""
        if self.surf_fit is None or self.surf_fit.get("method") != "spline_cl":
            messagebox.showinfo(
                "Sin spline activo",
                "Ajusta la superficie con el método\n"
                "'Spline CL(alpha) x q' primero.")
            return

        cs    = self.surf_fit["spline_cl"]
        knots = self.surf_fit["aoa_knots"]
        sref  = self.surf_fit.get("sref", 1.0)

        # scipy CubicSpline stores coefficients as cs.c  shape (4, n-1)
        # c[k, i] = coefficient of (x - x[i])^(3-k) on segment i
        # Reorder to natural: a=c[3], b=c[2], c_=c[1], d=c[0]
        c_mat = cs.c          # shape (4, n_segments)
        n_seg = c_mat.shape[1]

        path = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[
                ("Python script",    "*.py"),
                ("CSV coeficientes", "*.csv"),
                ("MATLAB/Octave",    "*.m"),
                ("JSON",             "*.json"),
                ("Todos",            "*.*"),
            ],
            title="Exportar spline CL(α)")
        if not path:
            return

        ext = path.lower().rsplit(".", 1)[-1]

        # ── Python ───────────────────────────────────────────────────────────
        if ext == "py":
            lines = [
                '"""',
                "Spline cúbico  CL(α)  exportado desde AeroSim Analyzer",
                "",
                "Uso:",
                "    cl   = eval_cl(alpha_deg)",
                "    lift = cl * 0.5 * rho * V**2 * S",
                "",
                "Parámetros:",
                f"    Sref usado en el ajuste : {sref} m²",
                f"    Rango válido α           : {knots[0]:.2f}° – {knots[-1]:.2f}°",
                '"""',
                "",
                "import numpy as np",
                "",
                "# Nodos del spline",
                f"KNOTS = np.array({knots.tolist()})",
                "",
                "# Coeficientes por segmento — shape (4, n_segmentos)",
                "# Para segmento i en [KNOTS[i], KNOTS[i+1]]:",
                "#   CL(α) = A[i] + B[i]*(α-KNOTS[i]) + C[i]*(α-KNOTS[i])²"
                " + D[i]*(α-KNOTS[i])³",
                f"# scipy ordena como c[0]=d, c[1]=c_, c[2]=b, c[3]=a",
                f"COEFFS = np.array({c_mat.tolist()})",
                "",
                f"SREF = {sref}",
                "",
                "def eval_cl(alpha_deg: float) -> float:",
                '    """Evalúa CL en un ángulo de ataque dado [°]."""',
                f"    alpha_deg = float(np.clip(alpha_deg, {knots[0]:.4f},"
                f" {knots[-1]:.4f}))",
                "    # Encontrar segmento",
                "    i = int(np.searchsorted(KNOTS, alpha_deg, side='right') - 1)",
                f"    i = int(np.clip(i, 0, {n_seg - 1}))",
                "    dx = alpha_deg - KNOTS[i]",
                "    # Evaluar polinomio (scipy: c[0]·dx³ + c[1]·dx² + c[2]·dx + c[3])",
                "    cl = (COEFFS[0, i]*dx**3 + COEFFS[1, i]*dx**2",
                "          + COEFFS[2, i]*dx + COEFFS[3, i])",
                "    return float(cl)",
                "",
                "def eval_lift(alpha_deg: float, V_ms: float,",
                "              rho_kgm3: float = 1.225, S_m2: float = SREF) -> float:",
                '    """Lift = CL(α) × ½ρV² × S  [N]."""',
                "    return eval_cl(alpha_deg) * 0.5 * rho_kgm3 * V_ms**2 * S_m2",
                "",
                "",
                'if __name__ == "__main__":',
                "    import sys",
                "    # Ejemplo de uso",
                "    test_cases = [",
                "        (10,  30,  1.225),",
                "        (20, 190,  1.225),",
                "        (30, 350,  1.225),",
                "        (15, 250,  0.9),   # densidad a ~1000 m",
                "    ]",
                "    print(f'{'α':>6}  {'V':>6}  {'ρ':>6}  {'CL':>10}  {'Lift [N]':>12}')",
                "    print('-' * 50)",
                "    for aoa, V, rho in test_cases:",
                "        cl   = eval_cl(aoa)",
                "        lift = eval_lift(aoa, V, rho)",
                "        print(f'{aoa:6.1f}  {V:6.1f}  {rho:6.3f}  {cl:10.6f}  {lift:12.4f}')",
            ]
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        # ── CSV ──────────────────────────────────────────────────────────────
        elif ext == "csv":
            rows = ["alpha_i,alpha_i1,A,B,C,D"]
            for i in range(n_seg):
                a  = c_mat[3, i]   # constant
                b  = c_mat[2, i]   # linear
                c_ = c_mat[1, i]   # quadratic
                d  = c_mat[0, i]   # cubic
                rows.append(
                    f"{knots[i]:.6f},{knots[i+1]:.6f},"
                    f"{a:.10e},{b:.10e},{c_:.10e},{d:.10e}")
            header = [
                "# Spline cúbico CL(α) — AeroSim Analyzer",
                f"# CL(α) = A + B*(α-αi) + C*(α-αi)² + D*(α-αi)³  para α ∈ [αi, αi+1]",
                f"# Sref = {sref} m²",
                f"# Lift = CL * 0.5 * rho * V^2 * Sref",
            ]
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(header) + "\n" + "\n".join(rows))

        # ── MATLAB / Octave ───────────────────────────────────────────────────
        elif ext == "m":
            knot_str = ", ".join(f"{k:.6f}" for k in knots)
            lines = [
                "% Spline cúbico CL(α) — exportado desde AeroSim Analyzer",
                f"% Sref = {sref} m²",
                "% Uso:  cl = eval_cl(alpha_deg)",
                "%       lift = cl * 0.5 * rho * V^2 * Sref",
                "",
                f"KNOTS = [{knot_str}];",
                "",
                "% Coeficientes: COEFFS(k+1, i+1) = coef de (α-αi)^(3-k) en segmento i",
            ]
            lines.append("COEFFS = [")
            for row in c_mat:
                lines.append("  " + "  ".join(f"{v:.10e}" for v in row) + ";")
            lines.append("];")
            lines += [
                "",
                f"SREF = {sref};",
                "",
                "function cl = eval_cl(alpha_deg)",
                f"  alpha_deg = max({knots[0]:.4f}, min({knots[-1]:.4f}, alpha_deg));",
                "  i = max(1, sum(KNOTS <= alpha_deg));",
                f"  i = min(i, {n_seg});",
                "  dx = alpha_deg - KNOTS(i);",
                "  cl = COEFFS(1,i)*dx^3 + COEFFS(2,i)*dx^2 + COEFFS(3,i)*dx + COEFFS(4,i);",
                "end",
                "",
                "function lift = eval_lift(alpha_deg, V, rho, S)",
                "  if nargin < 3, rho = 1.225; end",
                "  if nargin < 4, S = SREF; end",
                "  lift = eval_cl(alpha_deg) * 0.5 * rho * V^2 * S;",
                "end",
            ]
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        # ── JSON ─────────────────────────────────────────────────────────────
        elif ext == "json":
            import json
            data = {
                "description":  "Spline cúbico CL(α) — AeroSim Analyzer",
                "sref_m2":      sref,
                "alpha_range":  [float(knots[0]), float(knots[-1])],
                "formula":      "CL(α) = A + B*(α-αi) + C*(α-αi)² + D*(α-αi)³",
                "lift_formula": "Lift = CL * 0.5 * rho * V^2 * Sref",
                "segments": [
                    {
                        "alpha_i":   float(knots[i]),
                        "alpha_i1":  float(knots[i + 1]),
                        "A": float(c_mat[3, i]),
                        "B": float(c_mat[2, i]),
                        "C": float(c_mat[1, i]),
                        "D": float(c_mat[0, i]),
                    }
                    for i in range(n_seg)
                ],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        self.var_status.set(f"Spline exportado: {path}")
        messagebox.showinfo(
            "Exportado",
            f"Spline CL(α) guardado en:\n{path}\n\n"
            f"Segmentos: {n_seg}  |  Sref: {sref} m²")

    # ─── TABLA DE DATOS ──────────────────────────────────────────────────────

    def _show_table(self):
        if self.df_work is None:
            messagebox.showinfo("Sin datos", "Cargue un archivo primero.")
            return

        win = tk.Toplevel(self)
        win.title("Tabla de datos — SimScale")
        win.geometry("1100x520")
        win.configure(bg=PALETTE["bg"])

        cols_show = ["sim_key", "speed_ms", "aoa_deg",
                     "lift_N", "drag_N", "L_D_ratio",
                     "CL", "CD", "LD", "fx", "fy", "fz"]
        cols_show = [c for c in cols_show if c in self.df_work.columns]

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Treeview",
                        background=PALETTE["card"],
                        foreground=PALETTE["text"],
                        fieldbackground=PALETTE["card"],
                        rowheight=24)
        style.configure("Dark.Treeview.Heading",
                        background=PALETTE["surface"],
                        foreground=PALETTE["accent"],
                        font=("Helvetica", 9, "bold"))
        style.map("Dark.Treeview",
                  background=[("selected", PALETTE["accent"])])

        tree = ttk.Treeview(win, columns=cols_show, show="headings",
                            style="Dark.Treeview")

        col_widths = {"sim_key": 190, "speed_ms": 70, "aoa_deg": 70,
                      "lift_N": 90, "drag_N": 90, "L_D_ratio": 75,
                      "CL": 75, "CD": 75, "LD": 75,
                      "fx": 85, "fy": 85, "fz": 85, "CZ": 75}
        for col in cols_show:
            tree.heading(col, text=col)
            tree.column(col, width=col_widths.get(col, 90), anchor="center")

        df_sorted = self.df_work.sort_values(["speed_ms", "aoa_deg"])
        for _, row in df_sorted[cols_show].iterrows():
            vals = []
            for c in cols_show:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.5f}" if not np.isnan(v) else "—")
                else:
                    vals.append(str(v))
            tree.insert("", tk.END, values=vals)

        sb_y = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        sb_x = ttk.Scrollbar(win, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)

        sb_x.pack(side=tk.BOTTOM, fill=tk.X)
        sb_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)

        # Exportar CSV de tabla
        def export_table():
            p = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")],
                title="Exportar tabla")
            if p:
                df_sorted[cols_show].to_csv(p, index=False)
                messagebox.showinfo("Éxito", f"Tabla exportada:\n{p}")

        tk.Button(win, text="💾  Exportar tabla CSV",
                  command=export_table,
                  font=("Helvetica", 10, "bold"),
                  fg="white", bg=PALETTE["accent"],
                  relief=tk.FLAT, padx=12, pady=6,
                  cursor="hand2").pack(pady=8)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = AeroApp()
    app.mainloop()
