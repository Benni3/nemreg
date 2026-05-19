# nemreg/latex.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union, Any

from nemreg.core.dataset import Dataset
from nemreg.core.result import FitResult


# -------------------------
# Public API
# -------------------------

ReportID = Union[int, str]


@dataclass(frozen=True)
class LatexExportResult:
    """Standardized return so all templates feel the same."""
    id: str
    paths: Dict[str, Any]  # typically {"tex": Path, "pdf": Path?, "assets": Path}


def latex(
    report_id: ReportID,
    dataset: Dataset,
    result: FitResult,
    *,
    out_dir: str = "nemreg_report",
    filename_stem: Optional[str] = None,
    run_pdflatex: bool = True,
    **template_kwargs,
) -> LatexExportResult:
    """
    Export a LaTeX report using a numbered template.

    Usage
    -----
    >>> import nemreg as nr
    >>> from nemreg.latex import latex
    >>> res = nr.fit(ds, nr.models.linear())
    >>> out = latex(1, ds, res, out_dir="out", title="My title")

    Parameters
    ----------
    report_id:
        1 -> template latex1 (one-page report)
        You can add more: 2 -> latex2, 3 -> latex3, ...
    dataset, result:
        NEMREG objects.
    out_dir:
        Folder where .tex/.pdf/assets will be written.
    filename_stem:
        Base name for output files. Default: "report_<id>".
    run_pdflatex:
        If True, will try to compile PDF if pdflatex exists.
    template_kwargs:
        Forwarded to the selected template's options class (or exporter).

    Returns
    -------
    LatexExportResult
    """
    key = str(report_id).strip().lower()

    if key in ("1", "latex1", "one", "onepage", "one_page"):
        return _run_latex1(
            dataset,
            result,
            out_dir=out_dir,
            filename_stem=filename_stem or "report_1",
            run_pdflatex=run_pdflatex,
            **template_kwargs,
        )

    raise ValueError(
        f"Unknown LaTeX template id={report_id!r}. "
        "Available: 1"
    )


# -------------------------
# Template runners (private)
# -------------------------

def _run_latex1(
    dataset: Dataset,
    result: FitResult,
    *,
    out_dir: str,
    filename_stem: str,
    run_pdflatex: bool,
    **template_kwargs,
) -> LatexExportResult:
    """
    latex(1) -> nemreg.latex1.export_one_page_report(...)
    """
    # Import INSIDE to keep import-time side effects minimal
    from nemreg.export.latex1 import LatexReportOptions, export_one_page_report

    # Build options using kwargs (unknown keys will raise TypeError -> good!)
    opts = LatexReportOptions(**template_kwargs)

    paths = export_one_page_report(
        dataset,
        result,
        out_dir=out_dir,
        filename_stem=filename_stem,
        opts=opts,
        run_pdflatex=run_pdflatex,
    )
    return LatexExportResult(id="1", paths=paths)