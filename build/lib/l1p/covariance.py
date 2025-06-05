from imports import *


def compute_pk_with_cv(
    cosmo, z_values, z_values_all, nz_values, delta_A0=0.05, variability=True
):

    if variability:
        print("Variability is ON")
        # Sky area in steradians
        sky_area_deg2 = 100
        sky_area_sr = sky_area_deg2 * (4 * np.pi) / 41253

        # Compute comoving distance and H(z)
        chi = cosmo.get_chi(z_values)
        H = cosmo.getH(z_values)
        # Volume element weighted by n(z)
        dV_dz = sky_area_sr * nz_values * chi ** 2 * cosmo.speed_light / H
        V_eff = np.abs(np.trapz(dV_dz, nz_values))  # / (cosmo.h**3))
        print(f"Effective volume = {V_eff:.2e} (Mpc/h)^3")
        # Number of realizations
        n_real = 1
        # -------------------------------------------------
        # 2. halo‑model ingredients & trispectrum
        # -------------------------------------------------
        mdef = MassDef200m  #'200m'
        mf = MassFuncTinker10(mass_def=mdef)
        hbias = HaloBiasTinker10(mass_def=mdef)
        conc = ConcentrationDuffy08(mass_def=mdef)

        profile = HaloProfileNFW(mass_def=mdef, concentration=conc)
        hmc = HMCalculator(mass_function=mf, halo_bias=hbias)
        P_samples_all = []
        print("halo model calculator initialized")

    P_means_all = []

    # ------------------------
    # Loop over redshift samples
    # ------------------------
    a_all = 1.0 / (1.0 + z_values_all)  # scale factors
    for a in a_all:
        print(
            f"calculating for z = {1/a -1:.2f}, the final z: {1/a_all[-1] - 1:.2f}",
            end="\r",
        )
        k = cosmo.k_values  # * cosmo.h  # k in h/Mpc, shape = [Nk]

        # Get P(k, z) from CCL
        P_mean = ccl.nonlin_matter_power(cosmo.cosmoccl, k, a)  # * (cosmo.h**3)
        # print("mean P shape: ", P_mean.shape)
        if variability:
            tk3d = halomod_Tk3D_cNG(
                cosmo.cosmoccl,
                hmc,
                profile,
                lk_arr=np.log(k),
                a_arr=np.array([a]),
                use_log=False,
            )
            Tmat = tk3d(k, a)

            cov_total = compute_total_covariance(
                tk3d, k, a, P_mean, V_eff, delta_A0=delta_A0
            )

            eigvals, eigvecs = np.linalg.eigh(cov_total)
            eigvals = np.clip(eigvals, 0, None)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

            P_samples = P_mean[:, None] + L @ np.random.randn(len(k), n_real)
            P_samples_all.append(P_samples[:, 0])
        else:
            P_means_all.append(P_mean)

    if variability:
        P_samples_all = np.array(P_samples_all)
        # Build pk dictionary by redshift
        pk_dict = {z: P_samples_all[i, :] for i, z in enumerate(z_values_all)}
        return pk_dict
    else:
        P_means_all = np.array(P_means_all)
        # Build pk dictionary by redshift
        pk_dict = {z: P_means_all[i, :] for i, z in enumerate(z_values_all)}
        return pk_dict


def compute_total_covariance(
    tk3d, k, a, pk_mean, volume, delta_A0=1.0, response=None, sigma_b_squared=None
):
    """
    Compute total covariance matrix: Gaussian + non-Gaussian (trispectrum) + SSC.

    Parameters
    ----------
    cosmo : pyccl.Cosmology
    tk3d : pyccl.Tk3D
    k : array_like
        Wavenumbers [h/Mpc]
    a : float
        Scale factor
    pk_mean : array_like
        Mean P(k, a)
    volume : float
        Effective survey volume [Mpc^3/h^3]
    delta_A0 : float
        Optional shot noise component
    response : array_like or None
        If None, uses dlnP/dlnk as approximation.
    sigma_b_squared : float or None
        Variance of background mode (from survey window function).

    Returns
    -------
    cov_total : 2D ndarray
        Full covariance matrix with SSC.
    """
    # 1. Gaussian
    delta_k = np.gradient(k)
    Nk = (volume * (k ** 2) * delta_k) / (2 * (np.pi ** 2))
    sigma2 = (2 / Nk) + (0.5 ** 2)
    cov_gauss = np.diag(sigma2 * pk_mean ** 2)

    # 2. Non-Gaussian
    Tmat = tk3d(k, a)
    cov_ng = 0.5 * (Tmat + Tmat.T) / volume  # / volume

    # 3. Super-sample covariance
    if response is None:
        # Default: dlnP/dlnk
        dlnk = np.gradient(np.log(k))
        dlnP = np.gradient(np.log(pk_mean))
        slope = dlnP / dlnk
        response = 68.0 / 21.0 - (1.0 / 3.0) * slope

    if sigma_b_squared is None:
        # Default: estimate from top-hat window (approx)
        R_survey = (3 * volume / (4 * np.pi)) ** (1 / 3)
        k_window = np.linspace(1e-5, 1e2, 100)
        W = (
            3
            * (
                np.sin(k_window * R_survey)
                - k_window * R_survey * np.cos(k_window * R_survey)
            )
            / ((k_window * R_survey) ** 3)
        )
        pk_window = np.interp(k_window, k, pk_mean)
        integrand = (k_window ** 2) * pk_window * (W ** 2)
        sigma_b_squared = np.trapz(integrand, k_window) / (2 * (np.pi ** 2))

    cov_ssc = (
        np.outer(response, response)
        * pk_mean[:, None]
        * pk_mean[None, :]
        * sigma_b_squared
    ) / (volume ** 2)

    # Total covariance
    cov_total = cov_gauss   + (1.0 * cov_ssc) +  .8 * cov_ng 

    return cov_total
