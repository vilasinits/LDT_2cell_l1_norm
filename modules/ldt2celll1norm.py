from imports import *


def setup_cosmology_and_variance(
    cosmology_params,
    zs_source,
    kmin_pk,
    kmax_pk,
    nz_file,
    n_lensing_planes,
    filter_type,
    variability,
    theta1_arcmin,
):
    """Sets up the cosmology and variance calculations for lensing analysis.

    Args:
        cosmology_params : _Description of cosmology parameters, e.g., h, Om, Oc, sig8, w0, wa, etc._
        zs_source : _source redshift for lensing calculations_
        kmin_pk : _minimum k value for power spectrum calculations_
        kmax_pk : _maximum k value for power spectrum calculations_
        nz_file : _optional file containing redshift distribution data for lensing weight calculations_
        n_lensing_planes : _number of lensing planes to consider in the calculations_
        filter_type : _type_ of filter to apply for lensing calculations, e.g., 'top_hat', 'gaussian', etc._
        variability : _boolean indicating whether to include variability in the calculations_
        theta1_arcmin : _angular scale in arcminutes for lensing calculations_

    Returns:
        The `VariablesGenerator` instance containing all necessary variables for further calculations.
    """
    print("--- Running Cosmology Setup ---")

    # Extract from dict
    h = cosmology_params["h"]
    Oc = cosmology_params["Oc"]
    Om = cosmology_params["Om"]
    sig8 = cosmology_params["sig8"]
    w0 = cosmology_params["w0"]
    wa = cosmology_params.get("wa", 0.0)
    Ob = Om - Oc
    H0 = 100.0 * h
    ns = cosmology_params.get("ns", 0.963)
    mnu = cosmology_params.get("mnu", 0.0)

    print("Cosmological Parameters:")
    print(
        f"  h={h:.4f}, Om={Om:.4f}, Oc={Oc:.4f}, Ob={Ob:.4f}, sigma8={sig8:.4f}, w0={w0:.4f}, wa={wa:.4f}"
    )

    # 2. Initialize cosmology
    print("Initializing Cosmology...")
    cosmo = Cosmology_function(
        h=h,
        H0=H0,
        Ob=Ob,
        Oc=Oc,
        mnu=mnu,
        ns=ns,
        zs=zs_source,
        w=w0,
        wa=wa,
        kmin=kmin_pk,
        kmax=kmax_pk,
        nz_file=nz_file,
        sigma8=sig8,
    )

    # 3. Lensing geometry
    print("Calculating lensing geometry...")
    chi_source = cosmo.get_chi(zs_source)
    chis = np.linspace(0.3, chi_source - 5, n_lensing_planes)
    dchis = np.ones(len(chis)) * (chis[1] - chis[0]) / 2
    dchis[1:-1] *= 2.0

    if nz_file is not None:
        z_array, lensing_weight = cosmo.get_lensing_weight_array_nz(chis)
    else:
        z_array, lensing_weight = cosmo.get_lensing_weight_array(chis, chi_source)

    print(f"  Source Redshift: {zs_source}, Comoving Distance: {chi_source:.2f} Mpc/h")
    print(f"  Number of lensing planes: {n_lensing_planes}")

    crit_point_z = np.linspace(0.04, zs_source * 0.25, 5)

    # 4. Variance
    print("Initializing Variance and P(k)...")
    variance = Variance(
        cosmo=cosmo,
        z_values=z_array,
        z_values_critical=crit_point_z,
        filter_type=filter_type,
        variability=variability,
        delta_A0=0.7,
    )

    # 5. VariablesGenerator
    print("Initializing VariablesGenerator...")
    variables = VariablesGenerator(
        cosmo=cosmo,
        variance=variance,
        zs=zs_source,
        theta1=theta1_arcmin,
        nz_file=nz_file,
        nplanes=n_lensing_planes,
        chis=chis,
        dchis=dchis,
        z_array=z_array,
        lensing_weight=lensing_weight,
    )

    variables.recal_value = 1.0
    variables.crit_point_z = crit_point_z
    return variables


def find_critical_points_for_cosmo(variables, ngrid_critical=90):
    """Finds critical points in the lensing potential based on the provided variables.

    Args:
        variables (_type_): _description_
        ngrid_critical (int, optional): _description_. Defaults to 90.

    Returns:
        tuple: Smallest positive and largest negative critical point values.
    """
    print("Finding critical points...")
    criticalpoints = CriticalPointsFinder(variables, ngrid=ngrid_critical, plot=False)
    critical_values_list = []

    for z_crit in variables.crit_point_z:
        crit_vals = criticalpoints.get_critical_points(z_crit)
        if crit_vals is not None and len(crit_vals) >= 2:
            critical_values_list.append(crit_vals[:2])

    if not critical_values_list:
        print("  Warning: No critical points found in the specified redshift range.")
        return None, None

    # Flatten values
    flat_values = []
    for item in critical_values_list:
        if isinstance(item, (np.ndarray, list)):
            flat_values.extend(np.ravel(item))
        else:
            flat_values.append(item)

    flat_values = np.array(flat_values)
    flat_values = flat_values[~np.isnan(flat_values)]

    # Compute smallest positive and largest negative values
    positive_values = flat_values[flat_values > 0]
    negative_values = flat_values[flat_values < 0]

    smallest_positive = np.min(positive_values) if positive_values.size > 0 else None
    largest_negative = np.max(negative_values) if negative_values.size > 0 else None

    print("Smallest positive value:", smallest_positive)
    print("Largest negative value:", largest_negative)
    print(
        "Smallest distance pair of critical points:",
        smallest_positive,
        largest_negative,
    )

    return smallest_positive, largest_negative
