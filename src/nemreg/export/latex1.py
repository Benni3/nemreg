# nemreg/export/latex1.py
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

from nemreg.core.dataset import Dataset
from nemreg.core.result import FitResult


# ============================================================
# Options
# ============================================================

def _notes_block(opts: LatexReportOptions) -> str:
    """
    If notes_text is provided, split into lines and render as:
        \textit{...} \\
    Otherwise render N blank italic lines.
    """
    raw = (opts.notes_text or "").strip()

    if raw:
        # Allow user to separate lines with \n
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        # If user wrote one long line, keep it as one line
        if not lines:
            lines = [raw]
        return "\n".join([rf"\textit{{{_safe_tex_text(ln)}}} \\" for ln in lines])

    # fallback placeholders
    return "\n".join([r"\textit{} \\" for _ in range(max(1, opts.notes_block_lines))])

@dataclass
class LatexReportOptions:
    # Title shown at top
    title: str = "TITLE"

    # Notes block
    notes_block_lines: int = 4
    notes_text: str = "" 

    # Digits in report
    digits: int = 4

    # Plot settings
    figsize: Tuple[float, float] = (6.0, 4.0)
    dpi: int = 160

    # Optional “real life model” overlay on the plot
    optional_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
    optional_func_label: str = "real life model"   # shown in legend

    # Figure scaling (horizontal stretch only)
    figure_scale_x: float = 1.0
    figure_scale_y: float = 1.0

    plot_title: Optional[str] = None
    plot_xlabel: Optional[str] = None
    plot_ylabel: Optional[str] = None
    data_label: str = "data"
    fit_label: str = "fit"
    real_life_model_label: Optional[str] = None


# ============================================================
# Helpers
# ============================================================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _axis_label(label: str, unit: str) -> str:
    unit = (unit or "").strip()
    return f"{label} [{unit}]" if unit else label


def _safe_tex_text(s: Any) -> str:
    """Escape for LaTeX TEXT context (do not use for math)."""
    if s is None:
        return ""
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _yesno(v: bool) -> str:
    return "Yes" if bool(v) else "No"


def _fmt(v: float, digits: int) -> str:
    try:
        return f"{float(v):.{digits}g}"
    except Exception:
        return "-"


def _pm(val: float, err: float, digits: int) -> str:
    return rf"{_fmt(val, digits)} $\pm$ {_fmt(err, digits)}"


def _clean_expr(expr: str, variables: str = "x") -> str:
    """
    Clean expression formatting and prepend function notation.

    Example output:
        f(x) = 2.514 x - 0.9543
    """
    if not expr:
        return "-"

    # Fix ugly sign formatting
    expr = expr.replace("+ -", "- ")
    expr = expr.replace("+-", "-")

    # Remove accidental double spaces
    expr = " ".join(expr.split())

    return f"f({variables}) = {expr}"


def _as_1d_x(dataset: Dataset) -> np.ndarray:
    X = np.asarray(dataset.x, float)
    if X.ndim == 2:
        return X[:, 0].ravel()
    return X.ravel()


# ============================================================
# Dataset + Result info
# ============================================================

def _dataset_info(dataset: Dataset) -> Dict[str, str]:
    X = np.asarray(dataset.x, float)
    y = np.asarray(dataset.y, float).ravel()
    n = int(y.size)
    d = int(X.shape[1]) if X.ndim == 2 else 1

    info: Dict[str, str] = {}
    info["Name"] = dataset.name or "-"
    info["n (samples)"] = str(n)
    info["d (features)"] = str(d)
    info["x label"] = _axis_label(getattr(dataset, "xlabel", "x"), getattr(dataset, "xunit", ""))
    info["y label"] = _axis_label(getattr(dataset, "ylabel", "y"), getattr(dataset, "yunit", ""))
    info["yerr"] = "Provided" if getattr(dataset, "yerr", None) is not None else "None"
    info["xerr"] = "Provided" if getattr(dataset, "xerr", None) is not None else "None"

    try:
        if X.ndim == 2:
            info["x min/max"] = ", ".join(
                [f"x{i+1}: [{X[:, i].min():.4g}, {X[:, i].max():.4g}]" for i in range(d)]
            )
        else:
            info["x min/max"] = f"[{X.min():.4g}, {X.max():.4g}]"
    except Exception:
        info["x min/max"] = "-"

    try:
        info["y min/max"] = f"[{y.min():.4g}, {y.max():.4g}]"
    except Exception:
        info["y min/max"] = "-"

    return info


def _result_info(result: FitResult) -> Dict[str, Any]:
    stats = result.stats or {}
    meta = result.meta or {}

    params: List[Dict[str, Any]] = []
    try:
        for name, val, err in zip(result.param_names, result.popt, result.perr):
            params.append({"name": str(name), "val": float(val), "err": float(err)})
    except Exception:
        params = []

    return {
        "model": str(getattr(result, "model_name", "model")),
        "r2": float(stats.get("r2", float("nan"))),
        "rmse": float(stats.get("rmse", float("nan"))),
        "chi2": float(stats.get("chi2", float("nan"))),
        "chi2_red": float(stats.get("chi2_red", float("nan"))),
        "dof": int(stats.get("dof", -1)),
        "weighted": bool(meta.get("weighted", False)),
        "absolute_sigma": bool(meta.get("absolute_sigma", False)),
        "method": str(meta.get("method", "None")),
        "maxfev": meta.get("maxfev", ""),
        "bounds": meta.get("bounds", None) is not None,
        "p0": meta.get("p0", None),
        "params": params,
    }


# ============================================================
# Plot: plot_data_fit.png (data + fit + optional)
# ============================================================

def _plot_data_fit_png(dataset: Dataset, result: FitResult, out_dir: Path, opts: LatexReportOptions) -> Path:
    fig, ax = plt.subplots(figsize=opts.figsize)

    X = np.asarray(dataset.x, float)
    y = np.asarray(dataset.y, float).ravel()
    x = _as_1d_x(dataset)

    # ----------------------------
    # Resolve labels (user > dataset > default)
    # ----------------------------

    title = opts.plot_title if opts.plot_title is not None else "Data + Regression"

    xlabel = (
        opts.plot_xlabel
        if opts.plot_xlabel is not None
        else _axis_label(getattr(dataset, "xlabel", "x"), getattr(dataset, "xunit", ""))
    )

    ylabel = (
        opts.plot_ylabel
        if opts.plot_ylabel is not None
        else _axis_label(getattr(dataset, "ylabel", "y"), getattr(dataset, "yunit", ""))
    )

    data_label = opts.data_label
    fit_label = opts.fit_label

    # optional model label priority
    real_model_label = (
        opts.real_life_model_label
        if opts.real_life_model_label is not None
        else opts.optional_func_label
    )

    # ----------------------------
    # Plot data
    # ----------------------------
    ax.plot(x, y, "o", label=data_label)

    # ----------------------------
    # Fit curve
    # ----------------------------
    if X.ndim == 2 and X.shape[1] == 1:
        xg = np.linspace(float(np.min(x)), float(np.max(x)), 500)
        yg = np.asarray(result.predict(xg), float).ravel()
        ax.plot(xg, yg, "-", label=fit_label)

    else:
        try:
            yhat = np.asarray(result.yhat, float).ravel()
        except Exception:
            yhat = np.asarray(result.predict(X.T), float).ravel()

        order = np.argsort(x)
        ax.plot(x[order], yhat[order], "-", label=f"{fit_label} (obs)")

    # ----------------------------
    # Optional "real life model"
    # ----------------------------
    if opts.optional_func is not None:
        try:
            if X.ndim == 2 and X.shape[1] == 1:
                xg = np.linspace(float(np.min(x)), float(np.max(x)), 500)
                yg_ref = np.asarray(opts.optional_func(xg), float).ravel()
                ax.plot(xg, yg_ref, "--", label=real_model_label)
            else:
                yg_ref = np.asarray(opts.optional_func(X.T), float).ravel()
                order = np.argsort(x)
                ax.plot(x[order], yg_ref[order], "--", label=real_model_label)
        except Exception:
            pass

    # ----------------------------
    # Apply axis labels
    # ----------------------------
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(True)
    ax.legend(loc="best")

    path = out_dir / "plot_data_fit.png"

    fig.tight_layout()
    fig.savefig(path, dpi=opts.dpi)
    plt.close(fig)

    return path

# ============================================================
# LaTeX builder (matches your template)
# ============================================================

def build_report_tex(dataset: Dataset, result: FitResult, opts: LatexReportOptions) -> str:
    ds = _dataset_info(dataset)
    ri = _result_info(result)

    # Expression (prefer latex_str)
    expr = ""
    if hasattr(result, "latex_str"):
        try:
            expr = str(result.latex_str(digits=opts.digits))
        except Exception:
            expr = ""
    if not expr and hasattr(result, "expr_str"):
        try:
            expr = str(result.expr_str(digits=opts.digits))
            expr = expr.replace("*", " ")
        except Exception:
            expr = ""
    expr = _clean_expr(expr) if expr else "-"

    # Notes block text
    notes_lines = _notes_block(opts)

    # Caption EXACT style requested
    # "Graph showing <regression model> gegression of <Dataset_name> + Optional(real life model) of Title"
    model_name = ri["model"]
    data_name = dataset.name or "-"
    cap = f"Graph showing {model_name} regression of {data_name}"
    if opts.optional_func is not None:
        cap += f" + {opts.optional_func_label} of {opts.title}"
    cap_tex = _safe_tex_text(cap)

    # Fit summary rows
    p0_str = "(not stored)" if ri["p0"] is None else str(ri["p0"])
    fit_rows = "\n".join(
        [
            rf"Model & {_safe_tex_text(model_name)} \\",
            rf"$R^2$ & {_fmt(ri['r2'], opts.digits)} \\",
            rf"RMSE & {_fmt(ri['rmse'], opts.digits)} \\",
            rf"$\chi^2$ & {_fmt(ri['chi2'], opts.digits)} \\",
            rf"$\chi^2_{{\mathrm{{red}}}}$ & {_fmt(ri['chi2_red'], opts.digits)} \\",
            rf"dof & {ri['dof']} \\",
            rf"weighted & {_yesno(ri['weighted'])} \\",
            rf"absolute sigma & {_yesno(ri['absolute_sigma'])} \\",
            rf"method & {_safe_tex_text(ri['method'])} \\",
            rf"maxfev & {_safe_tex_text(ri['maxfev'])} \\",
            rf"bounds & {_yesno(ri['bounds'])} \\",
            rf"initial guess p0 & {_safe_tex_text(p0_str)} \\",
        ]
    )

    # Parameters rows
    if ri["params"]:
        param_rows = "\n".join(
            [rf"{_safe_tex_text(p['name'])} & {_pm(p['val'], p['err'], opts.digits)} \\" for p in ri["params"]]
        )
    else:
        param_rows = r"- & - \\"

    # Dataset rows
    ds_rows = "\n".join([rf"{_safe_tex_text(k)} & {_safe_tex_text(v)} \\" for k, v in ds.items()])

    # Emit LaTeX in the same structure you posted
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.7in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{graphicx}}
\usepackage{{caption}}
\usepackage{{subcaption}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{multicol}}
\usepackage{{amsmath}}
\usepackage{{xcolor}}

\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{4pt}}

\begin{{document}}

\begin{{center}}
{{\LARGE \textbf{{{_safe_tex_text(opts.title)}}}}}\\
\end{{center}}
\vspace{{4pt}}
\hrule
\vspace{{10pt}}

\textbf{{Notes}}\\

{notes_lines}

\vspace{{8pt}}

\begin{{figure}}[h!]
    \centering
    \scalebox{{{opts.figure_scale_x}}}[{opts.figure_scale_y}]{{%
        \includegraphics{{plot_data_fit.png}}%
    }}
    \caption{{{cap_tex}}}
    \label{{fig:placeholder}}
\end{{figure}}

\begin{{multicols}}{{2}}

\textbf{{Fit summary}}\\[-2pt]
\begin{{tabularx}}{{\linewidth}}{{@{{}}lX@{{}}}}
\toprule
Key & Value \\
\midrule
{fit_rows}
\bottomrule
\end{{tabularx}}

\vspace{{15pt}}

\[
{expr}
\]

\columnbreak

\textbf{{Fitted parameters}}\\[-2pt]
\begin{{tabularx}}{{\linewidth}}{{@{{}}lX@{{}}}}
\toprule
Parameter & Estimate \\
\midrule
{param_rows}
\bottomrule
\end{{tabularx}}

\vspace{{8pt}}

\textbf{{Dataset info}}\\[-0pt]
\vspace{{2pt}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}lX@{{}}}}
\toprule
Key & Value \\
\midrule
{ds_rows}
\bottomrule
\end{{tabularx}}

\end{{multicols}}

\end{{document}}
""".strip()

    return tex


# ============================================================
# Export
# ============================================================

def export_one_page_report(
    dataset: Dataset,
    result: FitResult,
    *,
    out_dir: str | Path = "nemreg_report",
    filename_stem: str = "nemreg_report",
    opts: Optional[LatexReportOptions] = None,
    run_pdflatex: bool = True,
) -> Dict[str, Path]:
    """
    Outputs:
      out_dir/<stem>.tex
      out_dir/plot_data_fit.png
      out_dir/<stem>.pdf (if pdflatex available + run_pdflatex=True)
    """
    opts = opts or LatexReportOptions()
    out_dir = Path(out_dir).resolve()
    _ensure_dir(out_dir)

    # 1) plot
    png_path = _plot_data_fit_png(dataset, result, out_dir, opts)

    # 2) tex
    tex_str = build_report_tex(dataset, result, opts)
    tex_path = out_dir / f"{filename_stem}.tex"
    tex_path.write_text(tex_str, encoding="utf-8")

    out: Dict[str, Path] = {"tex": tex_path, "png": png_path, "assets": out_dir}

    # 3) pdf
    if run_pdflatex:
        pdflatex = shutil.which("pdflatex")
        if pdflatex is None:
            out["pdf"] = out_dir / f"{filename_stem}.pdf"
            return out

        cmd = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
        for _ in range(2):
            subprocess.run(
                cmd,
                cwd=out_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        out["pdf"] = out_dir / f"{filename_stem}.pdf"

    return out