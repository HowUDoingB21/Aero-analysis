# AeroSim Analyzer

<img width="1134" height="402" alt="ORBITAL LOGOS TRANSPARENTE-01" src="https://github.com/user-attachments/assets/140166b7-c082-4e5e-b808-374f510bcd68" />

<p align="center">
  <b>Orbital Dynamics</b> — Análisis de datos CFD con interpolación aerodinámica
</p>

Herramienta de análisis para datos de simulaciones CFD de SimScale. Permite cargar resultados de barridos paramétricos, interpolar superficies aerodinámicas y exportar modelos para uso en computadoras de vuelo.

---

## Características

- **Gráfica principal** — CL, CD, Lift, Drag, L/D vs AoA por velocidad con interpolación polinomial o spline
- **Superficie f(V, α)** — Ajuste bivariable con 3 métodos:
  - *Spline CL(α) × q* — máxima precisión (<0.35% error), monótono en rango operacional
  - *Polinomio 2D normalizado* — ecuación algebraica cerrada
  - *RBF* — interpolación exacta en puntos de datos (requiere scipy)
- **Análisis de error** — Error absoluto y porcentual vs AoA por velocidad (comparado con SimScale)
- **Estadísticas** — Ecuación de CL(α) en grados 3, 4 y 5; tabla de nodos; R² y RMSE por velocidad
- **Exportación del spline** — Python, CSV, MATLAB/Octave, JSON, **Arduino (.h + .cpp)**

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/<usuario>/aero-analysis-gui.git
cd aero-analysis-gui

# 2. (Opcional) Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar
python aero_analysis_gui.py
```

## Dependencias

| Librería | Versión mínima | Uso |
|---|---|---|
| pandas | 1.5 | Carga y procesamiento del CSV |
| numpy | 1.23 | Álgebra y ajuste polinomial |
| matplotlib | 3.6 | Gráficas (Tkinter backend) |
| scipy | 1.9 | Spline cúbico e interpolación RBF *(opcional)* |

Sin `scipy`, el método Spline cae a interpolación lineal y el RBF no está disponible.

## Formato del CSV

El archivo CSV debe tener al menos estas columnas:

| Columna | Descripción |
|---|---|
| `speed_ms` | Velocidad del flujo [m/s] |
| `aoa_deg` | Ángulo de ataque [°] |
| `density_kgm3` | Densidad del aire [kg/m³] |
| `status` | Estado de la sim (`FINISHED`) |
| `lift_N` | Fuerza de sustentación [N] *(formato nuevo)* |
| `drag_N` | Fuerza de arrastre [N] *(formato nuevo)* |
| `L_D_ratio` | Relación L/D *(formato nuevo)* |
| `fx`, `fy`, `fz` | Fuerzas en ejes X/Y/Z [N] *(formato clásico)* |

El script acepta ambos formatos automáticamente.

## Flujo de uso típico

```
1. Cargar CSV → Sref → Recalcular
2. Elegir variable (Lift, CL, L/D…) → Generar gráfica
3. Superficie f(V,α) → Spline CL(α)×q → Ajustar
4. Graficar error vs AoA → verificar <1%
5. Exportar spline → Arduino (.h + .cpp)
```

## Exportación para Arduino (Caronte V1)

El botón **"Exportar spline CL(α)"** genera `AeroModel.h` y `AeroModel.cpp` listos para integrar en el firmware:

```cpp
// ServoController.cpp
#include "AeroModel.h"

float ServoController::aeroInverse(float lift_N, float air_density, float velocity_ms) {
    return aeroAlphaFromLift(lift_N, air_density, velocity_ms);
}
```

La inversión usa bisección (20 iteraciones, ~5 µs @ 168 MHz) en el rango operacional `[0°, DELTA_MAX_DEG]` donde CL(α) es estrictamente monótona.

## Insight físico clave

> CL depende **solo de α**, no de V ni de ρ.  
> Con una sola densidad de simulación obtienes CL(α) válido para cualquier altitud y velocidad.  
> `Lift = CL(α) × ½ρV² × S`

## Estructura del proyecto

```
aero-analysis-gui/
├── aero_analysis_gui.py    # Aplicación principal
├── requirements.txt        # Dependencias Python
├── README.md
└── .gitignore
```

Los archivos generados por exportación (`AeroModel.h`, `AeroModel.cpp`, `cl_spline.py`, etc.) **no se incluyen en el repo** — se generan desde la GUI con tus propios datos CFD.

---

*Orbital Dynamics — 2026*
