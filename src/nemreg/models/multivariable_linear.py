import numpy as np
from nemreg.core.model import Model


def multivariable_linear(d: int):
    """
    Multivariable linear:
        y = b + a1*x1 + a2*x2 + ... + ad*xd

    d = number of features (1–3)
    """

    if d < 1 or d > 3:
        raise ValueError("d must be between 1 and 3")

    param_names = ("b",) + tuple(f"a{i+1}" for i in range(d))

    def f(x, *params):
        x = np.asarray(x, dtype=float)

        if x.ndim != 2 or x.shape[0] != d:
            raise ValueError(f"Expected x shape ({d}, n)")

        b = params[0]
        a = np.array(params[1:])
        return b + a @ x

    def guess(x, y):
        return np.zeros(d + 1)
    
    # ---- build symbolic template ----
    terms = ["b"]
    latex_terms = ["b"]

    for i in range(d):
        terms.append(f"a{i+1}*x{i+1}")
        latex_terms.append(rf"a_{i+1} x_{i+1}")

    expr = " + ".join(terms)
    latex = " + ".join(latex_terms)

    return Model(
        name=f"multi_variable_{d}",
        func=f,
        param_names=param_names,
        guess=guess,
        expr=expr,
        latex=latex
    )