import os
import sys

sys.path.append("/feynman/work/dap/lcs/vt272285/final_codes/LDT_2cell_l1_norm")
import logging  # Good practice for messages/debugging
import unittest
from functools import lru_cache

# import multiprocessing as mp
import astropy.units as u
import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mpmath as mp
# Standard Libraries
import numpy as np
import pandas as pd
import pyccl as ccl
import scipy
import scipy.special as sp
from jax.numpy.fft import rfft2
from numpy import newaxis
from scipy.integrate import quad, simps, trapezoid
from scipy.interpolate import CubicSpline, UnivariateSpline, interp1d
from scipy.optimize import newton, root
from scipy.stats import qmc

try:
    # --- halo‑model pieces ---
    from pyccl.halos import MassDef200m
    from pyccl.halos.concentration.duffy08 import ConcentrationDuffy08
    from pyccl.halos.halo_model import HMCalculator
    from pyccl.halos.hbias.tinker10 import HaloBiasTinker10
    from pyccl.halos.hmfunc.tinker10 import MassFuncTinker10
    from pyccl.halos.massdef import MassDef200m
    from pyccl.halos.pk_4pt import \
        halomod_Tk3D_cNG  # full (1h+2h+3h+4h) trispectrum
    from pyccl.halos.profiles.nfw import HaloProfileNFW

    from src.filters import *
    from src.calculations_module import *
    from src.calculations_from_simulations import *
    
    # from modules.test_cosmology import *
    from src.computePDF_module import *
    from src.cosmology_module import *
    from src.covariance import *
    from src.variance_module import *
    
    from src.criticalpoints_module import *
    from src.variables_module import *
    
    from src.ratefunction_module import *
    from src.takahashi_loader import *
    from src.ldt2celll1norm import *
    

except ImportError as e:
    logging.error(f"Failed to import one or more project modules: {e}")

# Basic logging setup (optional)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Common imports loading...")

print("Executing common imports from imports.py...")  # Simple confirmation
