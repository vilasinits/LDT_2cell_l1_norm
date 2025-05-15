from imports import *

class VariablesGenerator:
    """
    Stores configuration and calculates the final map variance (sigmasq_map).
    Relies on injected cosmology, variance, and pre-calculated plane information.

    Attributes:
        cosmo (Cosmology_function): Injected cosmology object.
        variance (Variance): Injected variance object.
        zs (float): Source redshift.
        theta1_radian (float): Angular scale in radians converted from arcminutes.
        theta2_radian (float): Twice the angular scale of theta1 in radians.
        nz_file (str): Path to redshift distribution file.
        nplanes (int): Number of lensing planes.
        chis (np.ndarray): Comoving distances to lensing planes.
        dchis (np.ndarray): Differential comoving distances for integration.
        z_array (np.ndarray): Redshifts corresponding to chis.
        lensing_weight (np.ndarray): Lensing weights at each plane.
        sigmasq_map (float): The calculated mass map variance.
        recal_value (float): Recalibration value.
    
    Methods:
        __init__(self, cosmo, variance, zs, volume, theta1, nz_file, nplanes, chis, dchis, z_array, lensing_weight):
            Initializes the VariablesGenerator with injected objects and configuration.
    """

    def __init__(self, cosmo, variance, zs,  theta1, nz_file, nplanes, chis, dchis, z_array, lensing_weight):
        """
        Initializes the VariablesGenerator with injected objects and pre-calculated plane info.
        Calculates sigmasq_map.

        Parameters:
            cosmo (Cosmology_function): An initialized cosmology object.
            variance (Variance): An initialized variance object.
            zs (float): Source redshift.
            volume (float): Simulation volume.
            theta1 (float): Angular scale in arcminutes.
            nz_file (str): Path to n(z) file.
            nplanes (int): Number of planes.
            chis (np.ndarray): Comoving distances to planes.
            dchis (np.ndarray): Differential comoving distances.
            z_array (np.ndarray): Redshifts of planes.
            lensing_weight (np.ndarray): Lensing weights at planes.
        """
        self.cosmo = cosmo
        self.variance = variance
        self.zs = zs
        # self.volume = volume
        self.theta1_radian = (theta1 * u.arcmin).to(u.radian).value  # Convert theta1 from arcmin to radians
        self.theta2_radian = 2. * self.theta1_radian  # Double the angular scale for theta2
        self.nz_file = nz_file
        self.nplanes = nplanes
        self.chi_source = self.cosmo.get_chi(self.zs)
        self.chis = chis
        self.dchis = dchis
        self.z_array = z_array
        self.lensing_weight = lensing_weight
        self.lambdas = np.linspace(-100, 100, 30)
        # Calculate sigmasq_map directly here
        lensing_weight_squared = self.lensing_weight ** 2.
        print("Calculating sigmasq_map...")
        print("the shapes are: ", self.dchis.shape, lensing_weight_squared.shape)
        self.sigmasq_map = np.sum(self.dchis * lensing_weight_squared * np.array([
            self.variance.get_sig_slice(z, chi * self.theta1_radian, chi * self.theta2_radian)
            for z, chi in zip(self.z_array, self.chis)
        ]))
        self.recal_value = 1.  

        print("VariablesGenerator Initialized:")
        print(f"  Source Redshift (zs): {self.zs}")
        print(f"  Angular Scale (theta1): {theta1} arcmin")
        print(f"  Number of Planes: {self.nplanes}")
        print(f"  Using n(z) file: {self.nz_file is not None}")
        print(f"  Calculated Map Variance (sigmasq_map): {self.sigmasq_map}")

