import numpy as np
from nemreg.core.model import Model


def _multi_indices(d: int, degree: int):
    """
    Generate all exponent tuples (a1,...,ad) with sum(ai) <= degree.
    Includes (0,0,...,0) (the intercept term).
    """
    if d == 1:
        return [(k,) for k in range(degree + 1)]

    out = []

    def rec(prefix, remaining_degree, dim_left):
        if dim_left == 1:
            out.append(tuple(prefix + [remaining_degree]))
            return
        for k in range(remaining_degree + 1):
            rec(prefix + [k], remaining_degree - k, dim_left - 1)

    # For total degree <= degree: generate for each total t = 0..degree
    for t in range(degree + 1):
        rec([], t, d)

    return out


def _monomial_design(x_dn: np.ndarray, exps):
    """
    Build design matrix Phi of shape (n, n_terms),
    where each column j is prod_i x_i**exps[j][i].
    x_dn: shape (d, n)
    """
    d, n = x_dn.shape
    Phi = np.ones((n, len(exps)), dtype=float)
    for j, e in enumerate(exps):
        # multiply across dims
        col = np.ones(n, dtype=float)
        for i in range(d):
            ei = e[i]
            if ei != 0:
                col *= x_dn[i] ** ei
        Phi[:, j] = col
    return Phi


def multivariable_polynomial(d: int, degree: int):
    """
    Multivariable polynomial with interaction terms (total degree <= degree).

    d: number of features (1..3)
    degree: max total degree (0..10)

    Works with x passed as shape (d, n) inside the model (your fit() should pass dataset.x.T)
    """
    if d not in (1, 2, 3):
        raise ValueError("d must be 1, 2, or 3")
    if not isinstance(degree, int):
        raise TypeError("degree must be an int")
    if degree < 0 or degree > 10:
        raise ValueError("degree must be between 0 and 10")

    exps = _multi_indices(d, degree)  # list of tuples length d
    n_terms = len(exps)

    # Optional safety: avoid insane overparameterization
    # (still allows degree=10, but stops truly huge term counts if you later expand d)
    if n_terms > 500:
        raise ValueError(f"Too many polynomial terms ({n_terms}). Reduce degree.")

    # Nice parameter names like:
    # c0 (intercept), c_x1, c_x2, c_x1^2, c_x1*x2, ...
    def _name_for_exp(e):
        if sum(e) == 0:
            return "c0"
        parts = []
        for i, p in enumerate(e, start=1):
            if p == 0:
                continue
            if p == 1:
                parts.append(f"x{i}")
            else:
                parts.append(f"x{i}^{p}")
        return "c_" + "*".join(parts)

    param_names = tuple(_name_for_exp(e) for e in exps)

    def f(x, *coeffs):
        x = np.asarray(x, dtype=float)

        # Expect x shape (d, n)
        if x.ndim != 2 or x.shape[0] != d:
            raise ValueError(f"multivariable_polynomial expects x shape ({d}, n)")

        Phi = _monomial_design(x, exps)         # (n, n_terms)
        coeffs = np.asarray(coeffs, float)      # (n_terms,)
        return Phi @ coeffs                     # (n,)

    def guess(x, y):
        # Simple safe guess: intercept = mean(y), others = 0
        y = np.asarray(y, dtype=float).ravel()
        g = np.zeros(n_terms, dtype=float)
        g[0] = float(np.mean(y))  # c0
        return g

    def _mono_expr(e):
        parts = []
        for i, p in enumerate(e, start=1):
            if p == 0:
                continue
            if p == 1:
                parts.append(f"x{i}")
            else:
                parts.append(f"x{i}**{p}")
        return "*".join(parts) if parts else "1"

    def _mono_latex(e):
        parts = []
        for i, p in enumerate(e, start=1):
            if p == 0:
                continue
            if p == 1:
                parts.append(rf"x_{{{i}}}")
            else:
                parts.append(rf"x_{{{i}}}^{{{p}}}")
        return r" \cdot ".join(parts) if parts else "1"

    expr_terms = []
    latex_terms = []
    for cname, e in zip(param_names, exps):
        m_expr = _mono_expr(e)
        m_ltx = _mono_latex(e)

        if m_expr == "1":
            expr_terms.append(f"{cname}")
            latex_terms.append(rf"{cname}")
        else:
            expr_terms.append(f"{cname}*{m_expr}")
            latex_terms.append(rf"{cname}\,{m_ltx}")

    expr = " + ".join(expr_terms)
    latex = " + ".join(latex_terms)

    return Model(
        name=f"multivariable_polynomial_d{d}_deg{degree}",
        func=f,
        param_names=param_names,
        guess=guess,
        expr=expr,
        latex=latex
    )