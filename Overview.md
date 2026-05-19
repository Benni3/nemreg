# nemreg Architecture Overview

This document describes the internal structure and design philosophy of **nemreg**.

The goal of nemreg is to provide:

- Structured data handling (1D / 2D / 3D)
- Flexible regression models
- Clean uncertainty access
- Optional and configurable plotting
- Scalable and maintainable architecture
- Clear separation of responsibilities

---

# Folder Structure

src/nemreg/
│
├── core/
├── fitters/
├── models/
├── plot/
├── stats/
└── init.py


---

# core/

The heart of nemreg.  
All main data structures live here.

This layer defines **what objects are**, not how they are fitted or plotted.

---

## dataset.py

Contains the `Dataset` class.

### Responsibilities

- Store:
  - x, y (and optional z)
  - measurement uncertainties (sigma_x, sigma_y, sigma_z)
- Store metadata:
  - label
  - units
  - description
- Automatically detect dimensionality:
  - 1D: y = f(x)
  - 2D: z = f(x, y)
  - 3D: w = f(x, y, z)
- Validate:
  - array shapes
  - NaN filtering
  - finite values
- Provide:
  - weighting extraction
  - normalization utilities
  - clean accessors

### Design Principle

Dataset should:
- Know nothing about fitting
- Know nothing about plotting
- Only represent structured data

---

## session.py

Contains the `Session` class.

### Responsibilities

- Manage multiple datasets
- Add / remove datasets
- List datasets
- Store fit results
- Coordinate fitting
- Coordinate plotting

Think of `Session` as:

> An experiment workspace.

Users should typically interact with this object.

---

## result.py

Contains the `FitResult` class.

### Responsibilities

- Store:
  - fitted parameters
  - covariance matrix
  - parameter uncertainties
  - residuals
  - predictions
- Compute:
  - R²
  - Chi²
  - Reduced Chi²
  - AIC / BIC
- Provide:
  - `.params`
  - `.params_unc`
  - `.cov`
  - `.residuals`
  - `.predict(x)`
  - `.summary()`
  - `.plot()`

### Design Principle

All uncertainty logic lives here.

If the user wants uncertainties, they access them through `FitResult`.

---

## exceptions.py

Custom error types:

- `InvalidDatasetError`
- `ModelDimensionMismatch`
- `FitConvergenceError`

Professional error handling improves clarity and usability.

---

# fitters/

Numerical engines.

These modules perform fitting but:

- Do not store datasets
- Do not plot
- Do not manage sessions

They are pure computation layers.

---

## scipy_curvefit.py

Wrapper around `scipy.optimize.curve_fit`.

Handles:

- Parameter initialization
- Bounds
- Sigma weighting
- Absolute sigma logic
- Covariance extraction
- Error handling

Used for general nonlinear models.

---

## linear_least_squares.py

Analytical solver for linear models.

Used for:

- Linear regression
- Polynomial regression
- Multi-parameter linear models

Avoids nonlinear optimization when unnecessary.

---

## robust.py (future expansion)

Optional advanced methods:

- Huber regression
- RANSAC
- L1 regression

---

## bootstrap.py (future expansion)

Resampling-based uncertainty estimation.

---

# models/

Defines mathematical models.

This layer defines *what is being fitted*, not *how it is fitted*.

Group models by family, not one file per function.

---

## base.py

Defines a `FitModel` base class.

Responsibilities:

- Define parameter structure
- Define dimensionality
- Validate input shape
- Provide callable interface

---

## polynomial.py

Contains:

- linear
- quadratic
- cubic
- generic polynomial

For 1D regression.

---

## sine.py

Contains:

- sine
- damped sine
- sine with offset

---

## surface.py

Contains 2D models:

- plane
- quadratic surface
- generic polynomial surface

---

## physics.py

Domain-specific models:

- pendulum series
- kinematic equations
- exponential decay
- etc.

These are convenience models.

---

# plot/

All plotting logic lives here.

Plotting should be optional and configurable.

---

## plotter.py

Contains a `PlotManager` or `PlotConfig` class.

Responsibilities:

- Plot data
- Plot fitted curve / surface
- Plot residuals
- Plot confidence bands
- Support multiple datasets

---

## styles.py

Default styling:

- Colors
- Themes
- Line styles
- Marker styles

---

## layouts.py

Handles:

- Multi-panel plots
- Residual subplots
- 3D projections

---

# stats/

Statistical helpers.

This layer computes metrics but does not store data.

---

## metrics.py

Contains:

- R²
- Adjusted R²
- Chi²
- Reduced Chi²
- AIC
- BIC

---

## residuals.py

Helpers for:

- Residual normalization
- Weighted residuals
- Outlier detection

---

## confidence.py (future expansion)

Confidence intervals and prediction bands.

---

# Public API (via __init__.py)

`__init__.py` should expose only the clean public interface:

- Dataset
- Session
- FitResult
- fit
- common models (linear, quadratic, sine, etc.)

Keep it minimal and curated.

---

# Design Philosophy

nemreg should:

- Separate data from computation
- Separate computation from plotting
- Store all uncertainties inside result objects
- Allow multiple datasets per session
- Allow 1D, 2D, and up to 3D models
- Be scalable without becoming messy

---

# Future Expansion Ideas

- Joint fitting across datasets
- Parameter constraints
- Bayesian fitting
- Automatic model selection
- Symbolic-to-numeric model builder
- Interactive plotting
- Bootstrap confidence intervals

---

# Summary

nemreg is structured around five layers:

1. core      → data and result objects  
2. fitters   → numerical engines  
3. models    → mathematical definitions  
4. plot      → visualization  
5. stats     → statistical evaluation  

This separation ensures clarity, flexibility, and long-term scalability.

### Pipeline image

Dataset (core)
      ↓
Model (models)
      ↓
Fitter (fitters)
      ↓
FitResult (core)
      ↓
Stats (stats)
      ↓
Plot (plot)