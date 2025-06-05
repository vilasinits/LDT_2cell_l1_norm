import numpy as np
import scipy.integrate as simps
import pyccl as ccl
from .filters import top_hat_window, uHat_starlet_analytical 
from .covariance import compute_pk_with_cv
from .utils import apply_pixel_window

class Variance:
    """
    A class to compute linear and nonlinear variance using power spectrum interpolators and a specific cosmological model.
    
    Attributes:
        cosmo (Cosmology): An instance of a cosmology class providing necessary cosmological functions and parameters.
        PK_interpolator_linear (Interpolator): An interpolator instance for linear power spectrum calculations.
        PK_interpolator_nonlinear (Interpolator): An interpolator instance for nonlinear power spectrum calculations.
        model (str): The name of the cosmological model to be used for variance calculations.
        
    """

    def __init__(
        self,
        cosmo,
        z_values,
        z_values_critical,
        filter_type,
        variability=True,
        delta_A0=0.05,
    ):
        """
        Initializes the Variance class with cosmology and parameters for P(k) calculation.
        Calculates the non-linear power spectrum, including cosmic variance noise if volume is specified.
        
        Parameters:
            cosmo (Cosmology_function): An instance of the cosmology class.
            z_values (array-like): Redshifts for lensing planes/primary calculations.
            volume (float, optional): Volume for cosmic variance calculation in (Mpc/h)^3. Defaults to None (no CV noise).
            delta_A0 (float, optional): Parameter for additional non-Gaussian noise. Defaults to 1.9.
        """
        self.cosmo = cosmo
        self.filter_type = filter_type
        # Calculate the non-linear power spectrum internally
        self.crit_point_z = z_values_critical
        z_values_all = np.append(z_values, self.crit_point_z)
        # self.pk_nonlin = compute_pk_with_cv(cosmo, volume, z_values_all, delta_A0)
        self.pk_nonlin = compute_pk_with_cv(
            cosmo,
            cosmo.z_nz,
            z_values_all,
            cosmo.n_norm,
            delta_A0=delta_A0,
            variability=variability,
        )
        powspec = []
        a_nz = 1 / (1 + z_values_all)
        for zval in z_values_all:
            p = self.pk_nonlin[zval]
            powspec.append(p)
        powspec = np.array(powspec)
        # Step 2: Sort all arrays

        sort_idx = np.argsort(a_nz)
        a_nz_sorted = a_nz[sort_idx]
        P_sorted = powspec[sort_idx]

        tracer = ccl.WeakLensingTracer(
            self.cosmo.cosmoccl, dndz=(cosmo.z_nz, cosmo.n_nz)
        )
        ell_edges = np.linspace(10, 1e3, 61)

        self.ell = 0.5 * (ell_edges[1:] + ell_edges[:-1])
        # Create Pk2D for the mean
        pk2d_ = ccl.Pk2D(
            a_arr=a_nz_sorted,
            lk_arr=np.log(cosmo.k_values),
            pk_arr=P_sorted,
            is_logp=False,
            extrap_order_lok=1,
            extrap_order_hik=1,
        )


        w = apply_pixel_window(self.ell, theta_deg=10.0, npix=1200)
        self.cls = (
            ccl.angular_cl(
                self.cosmo.cosmoccl, tracer, tracer, self.ell, p_of_k_a=pk2d_
            )
            * (w ** 2)
            * self.cosmo.h ** 1
        )

        print("Variance module initialized...")


    def linear_sigma2(self, redshift, R1, R2=None):
        """
        Calculates the linear variance σ² for given scales and redshift, considering the specified model adjustments.
        
        Parameters:
            redshift (float): The redshift at which to evaluate the variance.
            R1 (float): The first scale radius.
            R2 (float, optional): The second scale radius. Defaults to R1 if not specified.
            
        Returns:
            float: The linear variance σ² at the given scales and redshift.
        """
        if R2 is None:
            R2 = R1
        else:
            R2 = R2

        # pk = self.PK_interpolator_linear.P(redshift, self.cosmo.k_values)
        pk = (
            ccl.linear_matter_power(
                self.cosmo.cosmoccl, self.cosmo.k_values, 1 / (1 + redshift)
            )
            * self.cosmo.h ** 3
        )
        if self.filter_type == "tophat":
            w1_2D = top_hat_window(self.cosmo.k_values * R1)
            w2_2D = top_hat_window(self.cosmo.k_values * R2)
            w2 = w1_2D * w2_2D
        elif self.filter_type == "starlet":
            w1_2D = uHat_starlet_analytical(self.cosmo.k_values * R1)
            w2_2D = w1_2D  # self.uHat_starlet_analytical(self.cosmo.k_values * R2)
            w2 = -w1_2D * w2_2D
        constant = 1.0 / 2.0 / np.pi
        integrand = self.cosmo.k_values * pk * w2 * constant
        return simps(integrand, x=self.cosmo.k_values)

    def nonlinear_sigma2(self, redshift, R1, R2=None):
        """
        Calculates the nonlinear variance σ² for given scales and redshift, considering the specified model adjustments.
        
        Parameters:
            redshift (float): The redshift at which to evaluate the variance.
            R1 (float): The first scale radius.
            R2 (float, optional): The second scale radius. Defaults to R1 if not specified.
            
        Returns:
            float: The nonlinear variance σ² at the given scales and redshift.
        """
        if R2 is None:
            R2 = R1
        else:
            R2 = R2

        pk = self.pk_nonlin[redshift]
        k = self.cosmo.k_values * self.cosmo.h
        if self.filter_type == "tophat":
            w1_2D = top_hat_window(k * R1)
            w2_2D = top_hat_window(k * R2)
            w2 = w1_2D * w2_2D
        elif self.filter_type == "starlet":
            w1_2D = uHat_starlet_analytical(k * R1)
            w2_2D = w1_2D  # self.uHat_starlet_analytical(k * R2)
            w2 = w1_2D * w2_2D
        constant = 1.0 / 2.0 / np.pi
        integrand = k * pk * w2 * constant
        return simps(integrand, x=k)

    def get_sig_slice(self, z, R1, R2):
        """
        Calculates the slice variance σ² for the given scales and redshift in the nonlinear regime.
        
        Parameters:
            z (float): The redshift at which to evaluate the slice variance.
            R1 (float): The first scale radius.
            R2 (float): The second scale radius.
            
        Returns:
            float: The slice variance σ² at the given scales and redshift.
        """
        # chi = self.cosmo.get_chi(z)
        if self.filter_type == "tophat":
            sigslice = (
                self.nonlinear_sigma2(z, R1)
                + self.nonlinear_sigma2(z, R2)
                - 2.0 * self.nonlinear_sigma2(z, R1, R2)
            )
            return sigslice
        elif self.filter_type == "starlet":
            sigslice = (
                self.nonlinear_sigma2(z, R1)
                # + self.nonlinear_sigma2(z, R2)
                # - 2.0 * self.nonlinear_sigma2(z, R1, R2)
            )
            return sigslice
