from __future__ import annotations

from pathlib import Path
import os
import sys
import shutil
import subprocess
import platform
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt

import nemreg as nr
from nemreg.export.latex import latex


x=np.linspace(-6,6,80); rng=np.random.default_rng(7)
trueA=2.5; trueB=-1.0; sigma=0.6
y=trueA*x+trueB+rng.normal(0,sigma,x.size); yerr=sigma

DIG=4; RUN=True; OUT="out_full_test"; STEM="report_demo"
title="Linear regression demonstration"; notes_text="Synthetic dataset with Gaussian noise, ipsum lorem, ipsum lorem, ipsum lorem, ipsum lorem, ipsum lorem, ipsum lorem, ipsum lorem, ipsum lorem"
figsize=(7,4); dpi=200; sx=1.0; sy=1.0
label_data="measurements"; label_fit="fit"; label_model="true model"
xlabel="x variable"; ylabel="observed y"
ds=nr.Dataset(x=x,y=y,yerr=yerr,xlabel=xlabel,ylabel=ylabel,name="Demo dataset")
m=nr.models.linear()

res=nr.fit(ds,m,p0=None,
           bounds=(-np.inf,np.inf),
           maxfev=100000,
           method=None,
           absolute_sigma=True)

nr.plot(mode="result_data",
        dataset=ds,
        result=res,
        grid=True,
        legend=True,
        marker="o",
        markersize=5,
        label=label_data)

expr=res.expr_str(DIG); latex_expr=res.latex_str(DIG)
print(res.summary(digits=DIG))
print("f(variables) =",expr)
print("latex:",latex_expr)
real=lambda t: trueA*np.asarray(t)+trueB

export=latex(
    1, ds, res,
    out_dir=OUT, filename_stem=STEM, run_pdflatex=RUN,
    title=title,
    notes_block_lines=4,
    notes_text=notes_text,
    digits=DIG,
    figsize=figsize, dpi=dpi,
    optional_func=real, optional_func_label=label_model,
    figure_scale_x=sx, figure_scale_y=sy
)

print({k:str(v) for k,v in export.paths.items()})
(__import__("os").system(f"open '{export.paths.get('pdf','')}'") if export.paths.get("pdf") else None)