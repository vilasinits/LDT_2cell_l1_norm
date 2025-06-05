from imports import *


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

    def __init__(
        self, h, H0, Ob, Oc, mnu, ns, zs, w, wa, kmin, kmax, nz_file, **kwargs
    ):
        self.h = h  # H0 is in km/s/Mpc
        self.Ob = Ob
        self.Oc = Oc
        self.mnu = mnu
        self.Omnu = mnu / 93.14 / self.h / self.h
        self.Om = Ob + Oc + (mnu / 93.14 / self.h / self.h)
        self.Oc = self.Om - Ob - self.Omnu
        self.ns = ns
        if "sigma8" in kwargs:
            self.sig8 = kwargs["sigma8"]
        else:
            self.sig8 = None
        if "As" in kwargs:
            self.As = kwargs["As"]
        else:
            self.As = None
        # self.sig8 = sig8
        self.Ol = 1.0 - self.Om
        self.w = w
        self.wa = wa
        self.H0 = H0
        self.speed_light = 299792.458
        self.zsource = zs
        self.zini = 0.0
        self.zmax = 5
        self.cosmoccl = self._set_params()
        self.kmin = kmin
        self.kmax = kmax
        self.nk = 120
        self.k_values = np.logspace(self.kmin, self.kmax, self.nk)

        self.nz_file = nz_file
        print(
            "the min and max values of k are: ",
            self.k_values[0],
            self.k_values[-1],
            "and length is: ",
            len(self.k_values),
        )

    def _set_params(self):
        if self.sig8 is not None:
            return ccl.Cosmology(
                h=self.h,
                Omega_c=self.Oc,
                Omega_b=self.Ob,
                sigma8=self.sig8,
                n_s=self.ns,
                w0=self.w,
                wa=self.wa,
                transfer_function="boltzmann_camb",
            )
        elif self.As is not None:
            return ccl.Cosmology(
                h=self.h,
                Omega_c=self.Oc,
                Omega_b=self.Ob,
                A_s=self.As,
                n_s=self.ns,
                w0=self.w,
                wa=self.wa,
                transfer_function="boltzmann_camb",
            )

    def get_chi(self, redshift):
        a = 1 / (1 + redshift)
        return ccl.comoving_radial_distance(self.cosmoccl, a) * self.h

    def getH(self, redshift):
        a = 1 / (1 + redshift)
        return ccl.h_over_h0(self.cosmoccl, a) * self.h * 100

    def get_z_from_chi(self, chi):
        a = ccl.scale_factor_of_chi(self.cosmoccl, chi / self.h)
        return (1 / a) - 1

    def get_lensing_weight(self, chi, chi_source):
        z = self.get_z_from_chi(chi)
        return (
            1.5
            * self.Om
            * (self.speed_light ** -2.0)
            * ((self.H0) ** 2.0)
            * chi
            * (1 - (chi / chi_source))
            * (1 + z)
        )

    def get_lensing_weight_array(self, chis, chi_source):
        z_values = self.get_z_from_chi(chis)
        lensing_weight = np.zeros_like(chis)
        for i in range(len(chis)):
            lensing_weight[i] = self.get_lensing_weight(chis[i], chi_source)

        plt.figure()
        plt.plot(z_values, lensing_weight)
        plt.xlabel("z")
        plt.ylabel("Lensing weight")
        plt.show()

        return z_values, lensing_weight

    def get_lensing_weight_array_nz(self, chis):
        """
        Compute the lensing weight for an array of lens distances using integrated n(z).
        Returns the corresponding lens redshifts and lensing weights.
        """
        if self.nz_file is None:
            raise ValueError(
                "nz_file must be provided for integrated lensing weight calculations."
            )

        # Load and normalize n(z)
        nz = np.load(self.nz_file)
        self.z_nz = nz[:, 0]
        self.n_nz = nz[:, 1]
        self.n_norm = self.n_nz / trapezoid(self.n_nz, self.z_nz)

        # Pre-compute the comoving distance for each source redshift
        a = 1 / (1 + self.z_nz)

        chi_nz = ccl.comoving_radial_distance(self.cosmoccl, a) * self.h
        dz_dw = np.gradient(self.z_nz, chi_nz)  # Compute dz/dw'
        q_s = self.n_norm * dz_dw

        lensing_weight = np.zeros_like(chis)
        z_values = self.get_z_from_chi(chis)

        for i, chi in enumerate(chis):
            chi = chi  # / self.h
            # only consider sources that are behind the lens (chi_source > chi)
            mask = chi_nz > chi
            if np.sum(mask) == 0:
                lensing_weight[i] = 0.0
            else:
                chi_nz_sel = chi_nz[mask]
                q_s_sel = q_s[mask]
                a_sel = 1 / (1 + self.z_nz[mask])  # Scale factor a(w')

                # The integrand: (1 - chi/chi_source) weighted by q_s
                integrand = (chi * (chi_nz_sel - chi) / chi_nz_sel) * (q_s_sel / a_sel)
                # Perform numerical integration
                integral = trapezoid(integrand, chi_nz_sel)

                prefactor = 1.5 * self.Om * ((self.H0 / self.speed_light) ** 2)
                # Use standard prefactor (z here is the *lens* redshift)
                lensing_weight[i] = prefactor * integral

        # plt.figure()
        # plt.plot(z_values, lensing_weight) #*self.h*self.h)
        # plt.xlabel('z')
        # plt.ylabel('Lensing weight')
        # plt.show()

        return z_values, lensing_weight  # *self.h*self.h
