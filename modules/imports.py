import os
import sys
sys.path.append('/feynman/work/dap/lcs/vt272285/final_codes/LDT_2cell_l1_norm')
# Standard Libraries
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.interpolate import CubicSpline, UnivariateSpline, interp1d
import logging # Good practice for messages/debugging


from scipy.optimize import root
from scipy.integrate import simps, quad, trapezoid
import scipy.special as sp
import scipy
import pyccl as ccl
import scipy.special as sp
import mpmath as mp
from scipy.optimize import newton

import healpy as hp
import multiprocessing as mp
import astropy.units as u
import mpmath as mp
import unittest
from scipy.stats import qmc
import pandas as pd


try:
    # --- halo‑model pieces ---
    from pyccl.halos import MassDef200m
    from pyccl.halos.massdef import MassDef200m
    from pyccl.halos.hmfunc.tinker10 import MassFuncTinker10
    from pyccl.halos.hbias.tinker10 import HaloBiasTinker10
    from pyccl.halos.concentration.duffy08 import ConcentrationDuffy08
    from pyccl.halos.profiles.nfw import HaloProfileNFW
    from pyccl.halos.halo_model import HMCalculator
    from pyccl.halos.pk_4pt import halomod_Tk3D_cNG   # full (1h+2h+3h+4h) trispectrum
    
    from modules.cosmology_module import *
    from modules.calculations_module import *
    from modules.covariance import *
    from modules.variance_module import *
    from modules.variables_module import *
    from modules.ratefunction_module import *
    from modules.takahashi_loader import *
    # from modules.test_cosmology import *
    from modules.computePDF_module import *
    from modules.criticalpoints_module import * 
    
except ImportError as e:
    logging.error(f"Failed to import one or more project modules: {e}")

# Basic logging setup (optional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Common imports loading...")

print("Executing common imports from imports.py...") # Simple confirmation 