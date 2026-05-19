"""
This file contains all the imports the project uses or may use in the future

All other files import this file using 'from imports import *'

For version of the different libraries, please reffere to 'Requirments'
"""

# Core numerical
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Scientific computing
from scipy.optimize import curve_fit
from scipy.stats import linregress
import scipy as sp

# Symbolic math
import sympy as sym

# Uncertainty handling
import uncertainties as unc
from uncertainties import unumpy as unp

# Machine learning / advanced regression
from sklearn.linear_model import LinearRegression

# Nice printing
sym.init_printing()

# Dataclass
from dataclasses import dataclass, field
from typing import Optional, List, Union