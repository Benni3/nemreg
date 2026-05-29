# nemreg

[![PyPI](https://img.shields.io/pypi/v/nemreg)](https://pypi.org/project/nemreg/)
[![License](https://img.shields.io/github/license/Benni3/NemReg)](LICENSE)

A lightweight regression framework built on top of SciPy.

**nemreg** is designed to make regression analysis structured, compact, and easy to use — without sacrificing flexibility.  
It wraps `scipy.optimize.curve_fit` in a clean scientific workflow that includes dataset handling, statistics, summaries, and plotting.

---

## Install nemreg

``` install nemreg ```


## Why nemreg?

SciPy is extremely powerful — but low-level.  
With SciPy alone, you must manually:

- Prepare data shapes
- Compute residuals
- Compute R² / RMSE / χ²
- Handle uncertainties
- Write plotting code
- Structure outputs

**nemreg** provides this structure for you.

```python
import nemreg as nr

ds = nr.Dataset(x, y, yerr=1.0)
res = nr.fit(ds, nr.models.linear())
nr.plot(mode="result_data", dataset=ds, result=res)
```

That’s it.

---

## Features

- Clean `Dataset` abstraction (1D and multivariable)
- Weighted and unweighted regression
- Automatic statistics:
  - R²
  - RMSE
  - χ²
  - Reduced χ²
- Structured `FitResult` object
- Built-in plotting dispatcher
- Built-in models (linear, polynomial, etc.)
- Support for replicated measurements
- Clean summary output
- Multivariable regression support

---

## Installation

Clone the repository and install dependencies:

```
pip install numpy scipy matplotlib
```

Optional:
```
pip install pytest jupyter
```

Then install nemreg (editable mode recommended):

```
pip install -e .
```

---

## Quick Start

```python
import numpy as np
import nemreg as nr

x = np.linspace(-5, 5, 40)
y = 2.5*x - 1 + np.random.normal(0, 1, size=x.size)

ds = nr.Dataset(x=x, y=y)
res = nr.fit(ds, nr.models.linear())
nr.plot(mode="result_data", dataset=ds, result=res)
```

---

## Project Philosophy

nemreg is not meant to replace SciPy.

It is meant to:

- Reduce boilerplate
- Encourage structured analysis
- Improve readability
- Make regression workflows easier to teach and share

SciPy remains the optimization engine under the hood.

---

## Note from the Author

This library is intended for free use and free sharing, as it was created in the spirit of making regression easier for everyone.

Please refer to the `LICENCE` file for details on usage, modification, and redistribution.

The author strongly recommends browsing through the `nemreg_demo.ipynb` file.  
It allows you to:

- Verify dependencies
- Confirm your environment works correctly
- See example workflows
- Explore available commands

---

## Planned Features

- LaTeX export for automatic report generation
- Native Levenberg–Marquardt (LM) core implementation, trying to improve the regression capabilities on more complex functions
- Additional diagnostics (influence, leverage, etc.)
- Robust fitting / outlier handling
- Extended model library
- Improved multivariable visualization

---

## License

See the `LICENCE` file for full details.

---

## Contributing

Contributions, suggestions, and improvements are welcome.  
Please open an issue or submit a pull request.

---

## Acknowledgements

Built on top of:
- NumPy
- SciPy
- Matplotlib

These libraries power the numerical engine behind nemreg.