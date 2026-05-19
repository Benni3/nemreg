import nemreg as nr


def test_polynomial_expr_exists():
    model = nr.models.polynomial(3)

    assert model.expr is not None
    assert model.latex is not None

    assert isinstance(model.expr, str)
    assert isinstance(model.latex, str)

    # basic structure check
    assert "c0" in model.expr
    assert "x**3" in model.expr or "x^3" in model.expr


def test_multivariable_linear_expr():
    model = nr.models.multivariable_linear(d=2)

    assert model.expr is not None
    assert model.latex is not None

    assert "b" in model.expr
    assert "a1" in model.expr
    assert "x1" in model.expr
    assert "x2" in model.expr


def test_multivariable_polynomial_expr():
    model = nr.models.multivariable_polynomial(d=2, degree=2)

    assert model.expr is not None
    assert model.latex is not None

    # should contain interaction terms like x1*x2 or similar
    assert "x1" in model.expr
    assert "x2" in model.expr


def test_custom_expr():
    model = nr.models.custom("A*x**2 + B*x + C")

    assert model.expr is not None
    assert model.latex is not None

    assert "A" in model.expr
    assert "x" in model.expr