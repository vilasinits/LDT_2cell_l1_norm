import numpy as np
import camb

class Cosmology_function:
    """
        Initializes the cosmology based on the parameters
        H0 = hubble constant in km/s/Mpc
        Ob = Omega baryon
        Oc = Omega cdm
        Omnu = Omega massive neutrinos
        Om = Omega matter = Ob + Oc + Omnu
        Ol = Omega lambda 

    """
    def __init__(self, h, H0, Ob, Oc, mnu, ns, As_, zs,**kwargs):
        self.h = h  # Assuming H0 is in km/s/Mpc
        self.Ob = Ob
        self.Oc = Oc
        self.mnu = mnu
        self.Omnu = mnu / 93.14 / self.h / self.h
        self.Om = Ob + Oc + (mnu / 93.14 / self.h / self.h)
        self.Oc = self.Om - Ob - self.Omnu
        self.ns = ns
        self.As_ = As_
        self.As = As_ * 1e-9
        self.Ol = 1.0 - self.Om
        
        
        self.H0 = H0
        self.speed_light = 299792.458
        self.zsource = zs
        self.zini = 0.0
        self.zmax = 5      
        self.pars = self._set_params()
        self.results = camb.get_results(self.pars)

    def _set_params(self):
        return camb.set_params(H0=self.H0 * self.h, omch2=self.Oc * (self.h ** 2.),
                               ombh2=self.Ob * (self.h ** 2.), NonLinear=camb.model.NonLinear_both,
                               mnu=self.mnu, omnuh2=self.Omnu * self.h * self.h, As=self.As, ns=self.ns,
                               halofit_version='takahashi')

    def get_chi(self, redshift):
        return self.results.comoving_radial_distance(redshift, tol=0.0000001) * self.h

    def get_z_from_chi(self, chi):
        return self.results.redshift_at_comoving_radial_distance(chi / self.h)

    def get_lensing_weight(self, chi, chi_source):
        z = self.get_z_from_chi(chi)
        return 1.5 * self.Om * (self.speed_light ** -2.) * ((self.H0) ** 2.) * chi * (1 - (chi / chi_source)) * (1 + z)
    
    def get_lensing_weight_array(self, chis, chi_source):
        z_values = self.get_z_from_chi(chis)
        lensing_weight = np.zeros_like(chis)
        for i in range(len(chis)):
            lensing_weight[i] = self.get_lensing_weight(chis[i], chi_source)

        return z_values, lensing_weight
        
    def get_matter_power_interpolator(self, nonlinear=False, kmin=1e-3, kmax=150, nk=300):
        self.k_values = np.logspace(np.log10(kmin), np.log10(kmax), nk, base=10)

        if nonlinear:
            PK_interpolator = camb.get_matter_power_interpolator(self.pars, nonlinear=True,
                hubble_units=True, k_hunit=True, kmax=kmax, zmax=self.zmax)
        else:
            PK_interpolator = camb.get_matter_power_interpolator(self.pars, nonlinear=False,
                hubble_units=True, k_hunit=True, kmax=kmax, zmax=self.zmax)

        # You can use PK_interpolator.P(z, k) to get the power spectrum at redshift z and wavenumber k
        return PK_interpolator
