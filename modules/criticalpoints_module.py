import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
import matplotlib.pyplot as plt
from ratefunction_module import *


class CriticalPointsFinder:
    """
    A class designed to identify critical points where the rate function's convexity changes in a cosmological context. 
    This is achieved through analyzing the Hessian matrix of the rate function across a grid of values, 
    identifying zero crossings in its determinant to locate changes in convexity.
    """
    
    
    """
    The rate function $I(x)$ characterizes the exponential decay rate of the probabilities of certain outcomes as the system size increases. The rate function is required to be convex, which ensures that the study of rare events through large deviation principles can be approached effectively through minimization techniques.

    Cumulant Generating Function and Legendre-Fenchel Transform:
    The CGF, denoted by $\Lambda(\theta)$, is foundational for deriving the rate function through the Legendre-Fenchel transform. This transform connects the CGF and the rate function as follows:

    $I(x) = \sup_{\theta} \{\theta x - \Lambda(\theta)\}$

    This equation ensures that the rate function $I(x)$ is convex, inheriting this property from the convex CGF $\Lambda(\theta)$. The supremum operation over $\theta$ highlights that $I(x)$ represents the tightest upper bound of the linear functions defined by $\theta x - \Lambda(\theta)$.

    Convexity of the Rate Function
    The convexity of the rate function $I(x)$ implies the following inequality for any two points $x_1$ and $x_2$ in its domain and any $\lambda \in [0, 1]$:

    $I(\lambda x_1 + (1 - \lambda)x_2) \leq \lambda I(x_1) + (1 - \lambda)I(x_2)$

    This inequality defines the convexity of the rate function, critical for analyzing rare events in large deviation theory.

    In this function we try to use the hessian determinants of the rate function to calculate the points where it is 0 and use that to find the value of lambda which we use in our calculation

    """
    def __init__(self, variables, ngrid=50, plot=False):
        """
        Initializes the CriticalPointsFinder with cosmology and variance objects, 
        and optionally configures plotting.

        Parameters:
            variables (VariablesGenerator): An instance containing all necessary cosmological parameters and variables.
            ngrid (int): The number of grid points to use for delta value calculations.
            plot (bool): Flag to enable plotting of critical points.
        """
        self.variables = variables
        self.plot = plot
        print(f"Setting ngrid = {ngrid}. Increase this for more accuracy, but note that computation becomes slower!")
        self.delta1_vals = np.linspace(-0.99, 2.0, ngrid)
        self.delta2_vals = np.linspace(-0.99, 2.0, ngrid)
        self.D1, self.D2 = np.meshgrid(self.delta1_vals, self.delta2_vals, indexing='ij')

    def get_hessian(self, x):
        """Calculates the Hessian matrix of a function."""
        x_grad = np.gradient(x) 
        hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
        for k, grad_k in enumerate(x_grad):
            tmp_grad = np.gradient(grad_k) 
            for l, grad_kl in enumerate(tmp_grad):
                hessian[k, l, :, :] = grad_kl
        return hessian

    def find_zero_crossing_point(self, x1, y1, x2, y2, determinant_value1, determinant_value2):
        """Finds the zero crossing point between two points based on the determinant values."""
        t = abs(determinant_value1) / (abs(determinant_value1) + abs(determinant_value2))
        zero_crossing_x = x1 + t * (x2 - x1)
        zero_crossing_y = y1 + t * (y2 - y1)
        return zero_crossing_x, zero_crossing_y

    def find_zero_crossings(self, determinant):
        """Identifies zero crossings in the determinant grid."""
        zero_crossings = []
        for i in range(determinant.shape[0] - 1):
            for j in range(determinant.shape[1] - 1):
                if determinant[i, j] * determinant[i, j+1] <= 0:
                    newx, newy = self.find_zero_crossing_point(self.D1[i, j], self.D2[i, j], self.D1[i, j+1], self.D2[i, j+1], determinant[i, j], determinant[i, j+1]) 
                    zero_crossings.append((newx, newy))
                if determinant[i, j] * determinant[i+1, j] <= 0:
                    newx, newy = self.find_zero_crossing_point(self.D1[i, j], self.D2[i, j], self.D1[i+1, j], self.D2[i+1, j], determinant[i, j], determinant[i+1, j]) 
                    zero_crossings.append((newx, newy))
        return zero_crossings
    
    def get_critical_points(self, z):
        """Calculates critical points for the given redshift z and plots them if requested."""
        chi_value = self.variables.cosmo.get_chi(z)
        recal_value = self.variables.recal_value
        theta1 = self.variables.theta1_radian
        theta2 = self.variables.theta2_radian
        chi_source = self.variables.chi_source

        deld = 1e-4
        rate_function = np.vectorize(lambda d1, d2: get_psi_2cell(self.variables.variance, chi_value, recal_value, z, d1, d2, theta1, theta2))(self.D1, self.D2)
        lw = self.variables.cosmo.get_lensing_weight(chi_value, chi_source)
        hessian = np.array(self.get_hessian(rate_function))
        determinants = np.array((hessian[0, 0, :, :] * hessian[1, 1, :, :]) - (hessian[0, 1, :, :] * hessian[1, 0, :, :]))
        zero_crossings = np.array(self.find_zero_crossings(determinants))
        drf1, drf2 = [], []
        for x, y in zero_crossings:
            drf1.append(get_psi_derivative_delta1(deld, self.variables.variance, chi_value, recal_value, z, x, y, theta1, theta2) / lw)
            drf2.append(get_psi_derivative_delta2(deld, self.variables.variance, chi_value, recal_value, z, x, y, theta1, theta2) / lw)
        drf1, drf2 = np.array(drf1), np.array(drf2)
        
        sorted_indices = np.argsort(drf1[:,0])
        # Sort drf1 and drf2 using the sorted indices
        sorted_drf1 = drf1[sorted_indices,0]
        sorted_drf2 = drf2[sorted_indices,0]
        
        drf_spline = CubicSpline(sorted_drf1[:], sorted_drf2[:])
        drf1_new = np.linspace(-6000,6000, 5000)
        drf2_new = drf_spline(drf1_new)
        sum_derivatives = (drf1_new + drf2_new) 
        # Fit spline to the sum of derivatives
        spline1 = UnivariateSpline(drf1_new, sum_derivatives, s=0)
        sorted_indices = np.argsort(drf2_new)
        # Find the value of x where the spline is 0
        critical_points1 = spline1.roots()
        
        print("The approximate critical points at redshift z: ", z, " are: ", -critical_points1) 
        if self.plot:
            plt.plot(drf1_new, sum_derivatives,label=z) 
            plt.scatter(critical_points1,spline1(critical_points1),color='r') 
            plt.xlim(-2000,2000)
            plt.ylim(-2000,2000)
            plt.grid(visible=True, which='both', axis='both')
        return -critical_points1