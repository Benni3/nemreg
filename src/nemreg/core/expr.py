from __future__ import annotations
from typing import Sequence, Optional
import re

def format_expr(template: Optional[str], param_names: Sequence[str], params: Sequence[float], digits: int = 4) -> Optional[str]:
    """Replace parameter symbols in template with numbers."""
    if template is None:
        return None
    s = template
    for name, val in zip(param_names, params):
        # replace whole-word occurrences only (prevents replacing 'a1' inside 'a10')
        s = re.sub(rf"\b{re.escape(name)}\b", f"{val:.{digits}g}", s)
    return s

def default_latex_from_expr(expr: Optional[str]) -> Optional[str]:
    """Very small 'good enough' converter for common cases."""
    if expr is None:
        return None
    return (expr
            .replace("*", r"\,")
            .replace("**", "^"))