import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import newton
from ratefunction_module import *

class computePDF:
    """
    A class to compute the Probability Distribution Function (PDF) for kappa using various
    cosmological and variance parameters contained within an instance of VariablesGenerator.
    """
    
    """
    The Laplace transform of a function \(f(t)\), where \(t \geq 0\), is defined as:
    \[ F(s) = \mathcal{L}\{f(t)\} = \int_{0}^{\infty} e^{-st} f(t) \, dt \]
    where \(s\) is a complex number (\(s = \sigma + i\omega\)), and \(F(s)\) is the transform of \(f(t)\) in the \(s\)-domain. The inverse Laplace transform, used to recover \(f(t)\) from \(F(s)\), is given by the Bromwich integral:
    \[ f(t) = \mathcal{L}^{-1}\{F(s)\} = \frac{1}{2\pi i} \int_{\gamma-i\infty}^{\gamma+i\infty} e^{st} F(s) \, ds \]
    The integration is performed along a vertical line in the complex plane, where \(\gamma\) is a real number chosen so that the contour passes to the right of all singularities of \(F(s)\).

    The exponential \(e^{st}\) can oscillate indefinitely when \(s\) is purely imaginary (\(s = i\omega\)), which doesn't always guarantee convergence. By allowing \(s\) to have a real part (\(\sigma\)), the exponential \(e^{-\sigma t}\) ensures that the integral converges for functions \(f(t)\) that grow less quickly than \(e^{\sigma t}\). This choice of \(s = \sigma + i\omega\) leverages the complex plane's properties to ensure convergence and apply complex analysis tools.

    By integrating \(F(s)\) multiplied by \(e^{st}\) over \(\omega\) from \(-\infty\) to \(+\infty\), with \(\sigma = \gamma\) fixed, the computation and interpretation of the inverse transform are simplified. This parallels the method of finding Fourier transforms, which are closely related to Laplace transforms but are defined on the imaginary axis.

    For PDF computation, especially as described, the principle is somewhat analogous. The transformation that links the characteristic or cumulant generating function with the PDF can be inverted using an integral performed along an imaginary path in the complex plane. This approach uses the oscillatory nature of the complex exponential to reconstruct the PDF from its transform:
    \[ \text{PDF}(\kappa) = \frac{1}{\pi} \int_{0}^{\infty} \text{Im}\left(e^{-\lambda \kappa + \phi(\lambda)}\right) \, d\lambda \]
    where \(\lambda\) corresponds to the imaginary part (\(i\omega\)) of the complex variable, and \(\phi(\lambda)\) plays a role similar to \(F(s)\), encapsulating the transform of the PDF to be inverted.
    """
    
    def __init__(self, variables, plot_scgf=False):
        """
        Initializes the computePDF with variables from VariablesGenerator.

        Parameters:
            variables (VariablesGenerator): An instance containing all necessary cosmological parameters and variables.
            plot_scgf (bool): Flag to enable plotting of the scaled cumulant generating function (SCGF).
        """
        self.variables = variables
        self.plot_scgf = plot_scgf
        self.pdf_values, self.kappa_values = self.compute_pdf_values()

    def get_scgf(self):
        """
        Computes the scaled cumulant generating function (SCGF) using parameters from the VariablesGenerator instance.
        """
        # Utilizing variables from the VariablesGenerator instance
        scgf = get_scaled_cgf(self.variables.theta1_radian, self.variables.theta2_radian, self.variables.z_array,
                              self.variables.chis, self.variables.dchis, self.variables.lensing_weight,
                              self.variables.lambdas, self.variables.recal_value, self.variables.variance)
        return scgf

    def compute_phi_values(self):
        """
        Computes phi values for the lambda range specified in the VariablesGenerator instance.
        Optionally plots the SCGF if plot_scgf is True.
        """
        scgf = self.get_scgf()
        scgf_spline = CubicSpline(self.variables.lambdas, scgf[:,0], axis=0)
        dscgf = scgf_spline(self.variables.lambdas, 1)
        if self.plot_scgf:
            plt.figure(figsize=(4,4))
            plt.plot(self.variables.lambdas, scgf)
            plt.show()

        tau_effective = np.sqrt(2.*(self.variables.lambdas * dscgf - scgf[:,0]))
        x_data = np.sign(self.variables.lambdas) * tau_effective
        y_data = dscgf

        coeffs = np.polyfit(x_data, y_data, 9)
        p = np.poly1d(coeffs)
        dp = p.deriv()
        print(p.coeffs[-2]**2)
        lambda_new = 1j * np.arange(0, 40000, 10)

        taus = np.zeros_like(lambda_new, dtype=np.complex128)

        def vectorized_equation(tau, lambda_):
            return tau - dp(tau) * lambda_

        for n, lambda_ in enumerate(lambda_new):
            initial_guess = np.sqrt(1j * (10 ** (-12))) if n == 0 else taus[n-1]
            taus[n] = newton(vectorized_equation, x0=initial_guess, args=(lambda_,))

        phi_values = lambda_new * p(taus) - ((taus**2) / 2.)
        return lambda_new, phi_values

    def compute_pdf_for_kappa(self, kappa, lambda_new, phi_values):
        """
        Computes the PDF for a given kappa value using the computed phi values by applying bromwhich integral.
        """
        delta_lambda = np.abs(lambda_new[1] - lambda_new[0]) * 1j
        lambda_weight = np.full(len(lambda_new), delta_lambda)
        lambda_weight[0] = lambda_weight[-1] = delta_lambda / 2.

        integral_sum = np.sum(np.exp(-lambda_new * kappa + phi_values) * lambda_weight)
        pdf_kappa = np.imag(integral_sum / (1. * np.pi))  # Corrected the denominator to 2*np.pi for proper normalization

        return pdf_kappa.real

    def compute_pdf_values(self):
        """
        Computes PDF values for a range of kappa values.
        """
        kappa_values = np.linspace(-0.1, 0.1, 2000)
        lambda_new, phi_values = self.compute_phi_values()
        pdf_values = [self.compute_pdf_for_kappa(kappa, lambda_new, phi_values) for kappa in kappa_values]
        return pdf_values, kappa_values
