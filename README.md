# AeroSim Analyzer
<img width="1134" height="402" alt="ORBITAL LOGOS TRANSPARENTE-01" src="https://github.com/user-attachments/assets/3b6a5949-8e7a-4f26-a1a2-81ed9a1c930d" />

<p align="center">
  <b>Orbital Dynamics</b> — CFD Data Analysis with Aerodynamic Interpolation
</p>

A desktop analysis tool for SimScale CFD parametric sweep results. Load simulation data, interpolate aerodynamic surfaces, and export models ready for use in flight computers.

---

## Features

- **Main plot** — CL, CD, Lift, Drag, L/D vs AoA per velocity with polynomial or spline interpolation
- **Surface f(V, α)** — Bivariate fitting with 3 methods:
  - *Spline CL(α) × q* — highest accuracy (<0.35% error), monotone within operational range
  - *Normalized 2D polynomial* — closed-form algebraic equation
  - *RBF* — exact interpolation at data points (requires scipy)
- **Error analysis** — Absolute and percentage error vs AoA per velocity (compared against SimScale)
- **Statistics panel** — CL(α) equation at degrees 3, 4 and 5; knot table; R² and RMSE per velocity
- **Spline export** — Python, CSV, MATLAB/Octave, JSON, **Arduino (.h + .cpp)**

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<username>/aero-analysis-gui.git
cd aero-analysis-gui

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python aero_analysis_gui.py
```

## Dependencies

| Library | Minimum version | Purpose |
|---|---|---|
| pandas | 1.5 | CSV loading and processing |
| numpy | 1.23 | Linear algebra and polynomial fitting |
| matplotlib | 3.6 | Plots (Tkinter backend) |
| scipy | 1.9 | Cubic spline and RBF interpolation *(optional)* |

Without `scipy`, the Spline method falls back to linear interpolation and RBF is unavailable.

## CSV Format

The input CSV must contain at least these columns:

| Column | Description |
|---|---|
| `speed_ms` | Flow velocity [m/s] |
| `aoa_deg` | Angle of attack [°] |
| `density_kgm3` | Air density [kg/m³] |
| `status` | Simulation status (`FINISHED`) |
| `lift_N` | Lift force [N] *(new format)* |
| `drag_N` | Drag force [N] *(new format)* |
| `L_D_ratio` | Lift-to-drag ratio *(new format)* |
| `fx`, `fy`, `fz` | Forces along X/Y/Z axes [N] *(classic format)* |

Both formats are detected and handled automatically.

## Typical Workflow

```
1. Load CSV → set Sref → Recalculate
2. Choose variable (Lift, CL, L/D…) → Plot
3. Surface f(V,α) → Spline CL(α)×q → Fit
4. Plot error vs AoA → verify < 1%
5. Export spline → Arduino (.h + .cpp)
```

## Arduino Export (Caronte V1)

The **"Export spline CL(α)"** button generates `AeroModel.h` and `AeroModel.cpp`, ready to drop into the firmware:

```cpp
// ServoController.cpp
#include "AeroModel.h"

float ServoController::aeroInverse(float lift_N, float air_density, float velocity_ms) {
    return aeroAlphaFromLift(lift_N, air_density, velocity_ms);
}
```

The inversion uses bisection (20 iterations, ~5 µs @ 168 MHz) within the operational range `[0°, DELTA_MAX_DEG]` where CL(α) is strictly monotone.

## Key Physical Insight

> CL depends **only on α**, not on V or ρ.  
> A single-density simulation sweep gives a CL(α) curve valid at any altitude and velocity.  
> `Lift = CL(α) × ½ρV² × S`

## Project Structure

```
aero-analysis-gui/
├── aero_analysis_gui.py    # Main application
├── requirements.txt        # Python dependencies
├── README.md
└── .gitignore
```

Exported files (`AeroModel.h`, `AeroModel.cpp`, `cl_spline.py`, etc.) are **not tracked** by the repository — they are generated from the GUI using your own CFD data.

---

*Orbital Dynamics — 2026*
