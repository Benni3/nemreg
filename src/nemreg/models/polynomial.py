import numpy as np
from nemreg.core.model import Model

def polynomial(degree: int) -> Model:
    """
    Polynomial model (1D):
        y = c0 + c1*x + c2*x^2 + ... + cN*x^N
    degree: N (0..10)
    """
    if not isinstance(degree, int):
        raise TypeError("degree must be an int")
    if degree < 0 or degree > 10:
        raise ValueError("degree must be between 0 and 10")

    param_names = tuple(f"c{k}" for k in range(degree + 1))

    def f(x, *coeffs):
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            # if passed as (d, n), for 1D polynomial we accept d==1
            if x.shape[0] != 1:
                raise ValueError("polynomial model expects 1D x")
            x = x[0]
        # coeffs are c0..cN, polyval needs highest-first:
        return np.polyval(list(coeffs)[::-1], x)

    def guess(x, y):
        g = np.zeros(degree + 1, dtype=float)
        g[0] = float(np.mean(np.asarray(y, float)))
        return g

    # expression templates (optional but useful)
    expr_terms = []
    latex_terms = []
    for i in range(degree + 1):
        if i == 0:
            expr_terms.append("c0")
            latex_terms.append("c0")
        elif i == 1:
            expr_terms.append("c1*x")
            latex_terms.append(r"c1 x")
        else:
            expr_terms.append(f"c{i}*x**{i}")
            latex_terms.append(rf"c{i} x^{i}")

    expr = " + ".join(expr_terms)
    latex = " + ".join(latex_terms)

    return Model(
        name=f"poly{degree}",
        func=f,
        param_names=param_names,
        guess=guess,
        expr=expr,
        latex=latex,
    )