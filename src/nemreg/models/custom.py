import numpy as np
import sympy as sp
from nemreg.core.model import Model


def custom(
    formula: str,
    *,
    var="x",
    vars=None,
    param_order=None,
    name="custom",
):
    """
    Create a custom model from a user-written formula.

    Parameters
    ----------
    formula : str
        Example: "a*x**2 + b*x + c" or "A*exp(-k*x) + B"
    var : str
        Convenience for single-variable models. Default "x".
    vars : list[str] | tuple[str] | None
        Multi-variable form. Example: ["x1","x2"] or ["t","x"].
        If provided, overrides `var`.
    param_order : list[str] | None
        Optional explicit ordering of parameters (symbols) to fit.
        If None, all symbols in expression except vars are parameters,
        sorted by name.
    name : str
        Model name.

    Returns
    -------
    Model
        A nemreg Model compatible with curve_fit: func(x, *params) -> y
    """

    # ---- choose variables ----
    if vars is None:
        vars = [var]
    else:
        vars = list(vars)

    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("custom(): formula must be a non-empty string")

    if not all(isinstance(v, str) and v.strip() for v in vars):
        raise ValueError("custom(): vars/var must be non-empty string(s)")

    d = len(vars)
    if d < 1 or d > 3:
        raise ValueError("custom(): supports 1–3 variables (vars length must be 1..3)")

    # ---- build sympy symbols for variables ----
    var_syms = [sp.Symbol(v) for v in vars]
    var_set = set(var_syms)

    # allow common functions in user formulas
    # (sympify will recognize many, but we also provide explicit locals)
    locals_map = {
        **{v: s for v, s in zip(vars, var_syms)},
        "pi": sp.pi,
        "E": sp.E,
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "Abs": sp.Abs,
    }

    # ---- parse expression ----
    try:
        expr = sp.sympify(formula, locals=locals_map)
    except Exception as e:
        raise ValueError(
            "custom(): could not parse formula.\n"
            f"  formula: {formula!r}\n"
            f"  error: {type(e).__name__}: {e}\n"
            "Tips:\n"
            "  - Use ** for powers (x**2), not ^\n"
            "  - Use exp(x), log(x), sin(x), cos(x), etc.\n"
            "  - Make sure your variable name matches var/vars."
        ) from e

    # ---- determine parameter symbols ----
    free_syms = set(expr.free_symbols)

    # check that variables actually appear
    if not (free_syms & var_set):
        raise ValueError(
            "custom(): your formula does not contain the specified variable(s).\n"
            f"  vars={vars}\n"
            f"  formula={formula!r}\n"
            "Did you misspell the variable name?"
        )

    inferred_params = sorted([s for s in free_syms if s not in var_set], key=lambda s: s.name)

    if param_order is not None:
        if not isinstance(param_order, (list, tuple)) or not all(isinstance(p, str) for p in param_order):
            raise ValueError("custom(): param_order must be a list of parameter names (strings)")
        param_syms = [sp.Symbol(p) for p in param_order]

        # validate: all ordered params exist in expression
        missing = [p for p in param_syms if p not in free_syms]
        if missing:
            miss_names = [m.name for m in missing]
            raise ValueError(
                "custom(): param_order includes parameter(s) not found in the formula: "
                + ", ".join(miss_names)
            )

        # validate: ensure no other free symbols besides vars and param_order
        extras = [s for s in free_syms if (s not in var_set) and (s not in param_syms)]
        if extras:
            extra_names = [e.name for e in sorted(extras, key=lambda s: s.name)]
            raise ValueError(
                "custom(): formula contains extra symbol(s) not listed in param_order: "
                + ", ".join(extra_names)
                + "\nEither add them to param_order or remove them from the formula."
            )
    else:
        param_syms = inferred_params

    if len(param_syms) == 0:
        raise ValueError(
            "custom(): no parameters found to fit.\n"
            "Your formula only contains variables and constants.\n"
            "Example with parameters: 'a*x + b' (parameters: a, b)"
        )

    if len(param_syms) > 20:
        raise ValueError(
            f"custom(): too many parameters detected ({len(param_syms)}). "
            "This will be unstable. Consider simplifying your formula or use param_order."
        )

    param_names = tuple(s.name for s in param_syms)

    # ---- compile numeric function ----
    # order: vars first, then parameters
    try:
        fn = sp.lambdify([*var_syms, *param_syms], expr, modules="numpy")
    except Exception as e:
        raise ValueError(
            "custom(): failed to compile the formula to a numeric function.\n"
            f"  error: {type(e).__name__}: {e}"
        ) from e

    # ---- adapt to curve_fit signature f(x, *params) ----
    def f(x, *params):
        # x can be (n,) or (1,n) for 1D; (d,n) for d>1
        x = np.asarray(x, dtype=float)

        if d == 1:
            if x.ndim == 2:
                if x.shape[0] != 1:
                    raise ValueError("custom(): expected 1D x, got shape (d,n) with d!=1")
                x0 = x[0]
            elif x.ndim == 1:
                x0 = x
            else:
                raise ValueError("custom(): x must be 1D (n,) or 2D (1,n)")
            try:
                return np.asarray(fn(x0, *params), dtype=float)
            except Exception as e:
                raise ValueError(
                    "custom(): evaluation failed.\n"
                    f"  formula={formula!r}\n"
                    f"  vars={vars}\n"
                    f"  params={param_names}\n"
                    f"  error={type(e).__name__}: {e}"
                ) from e

        else:
            # multi-variable: expect (d,n)
            if x.ndim != 2 or x.shape[0] != d:
                raise ValueError(f"custom(): expected x shape ({d}, n) for vars={vars}, got {x.shape}")

            xs = [x[i] for i in range(d)]
            try:
                return np.asarray(fn(*xs, *params), dtype=float)
            except Exception as e:
                raise ValueError(
                    "custom(): evaluation failed.\n"
                    f"  formula={formula!r}\n"
                    f"  vars={vars}\n"
                    f"  params={param_names}\n"
                    f"  error={type(e).__name__}: {e}"
                ) from e

    # ---- default guess: ones ----
    def guess(x, y):
        return np.ones(len(param_syms), dtype=float)

    # ---- build expr + latex strings ----
    # Plain expression string (normalized by sympy)
    expr_str = str(expr)

    # Pretty LaTeX
    try:
        latex_str = sp.latex(expr)
    except Exception:
        latex_str = None

    return Model(
        name=name,
        func=f,
        param_names=param_names,
        guess=guess,
        expr=expr_str,
        latex=latex_str,
    )