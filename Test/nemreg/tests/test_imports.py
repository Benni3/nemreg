# tests/test_imports.py
import nemreg as nr

def test_import_path_is_src():
    # You wanted to verify this exact file path
    assert nr.__file__.endswith("/Test/nemreg/src/nemreg/__init__.py")

def test_public_api_exists():
    # these should be available via lazy getattr in __init__.py
    _ = nr.Dataset
    _ = nr.Model
    _ = nr.FitResult
    _ = nr.fit
    _ = nr.plot
    _ = nr.models