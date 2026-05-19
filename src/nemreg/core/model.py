from dataclasses import dataclass
from typing import Callable, Sequence, Optional, Union
import numpy as np

ArrayLike = Union[np.ndarray, float]

@dataclass(frozen=True)
class Model:
    """
    Mathematical model definition.

    func must have signature:
        f(x, *params) -> y
    """

    name: str
    func: Callable[..., np.ndarray]
    param_names: Sequence[str]
    guess: Optional[Callable[[np.ndarray, np.ndarray], Sequence[float]]] = None
    expr: Optional[str] = None        # python-ish string: "A*x + B"
    latex: Optional[str] = None

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def __call__(self, x: ArrayLike, *params) -> np.ndarray:
        """
        Allows calling the model like:
            model(x, *params)
        """
        return self.func(x, *params)

    def initial_guess(self, x: np.ndarray, y: np.ndarray):
        """
        Guess priority:
        1) Model-specific guess function
        2) Fallback: ones
        """
        if self.guess is not None:
            return np.asarray(self.guess(x, y), dtype=float)

        return np.ones(self.n_params, dtype=float)