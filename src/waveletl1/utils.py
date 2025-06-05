import numpy as np


def apply_pixel_window(ells, theta_deg=10.0, npix=1200):
    """
    Apply pixel window function to theoretical Cls.

    Parameters:
    - cls: array of C_ell values (same length as ells)
    - ells: array of multipoles (ell values)
    - theta_deg: total angular size of the map (in degrees)
    - npix: number of pixels on one side of the square map

    Returns:
    - cls_smoothed: Cls multiplied by the pixel window function
    """
    # Convert pixel size to radians
    theta_pix_rad = np.deg2rad(theta_deg / npix)

    # Pixel window function W(ell) = sinc(ell * theta_pix / 2)^2
    # np.sinc(x) = sin(pi*x)/(pi*x), so we must divide by pi
    arg = ells * theta_pix_rad / 2
    W_ell = np.sinc(arg / np.pi) ** 2

    return W_ell


def fourier_coordinate(x, y, map_size):
    return (((map_size // 2) + 1) * x) + y


def get_moments(kappa_values, pdf_values):
    """
    Calculates the moments (mean, variance, skewness, kurtosis) of a probability distribution function.

    Parameters:
        kappa_values (numpy.ndarray): A 1D array of kappa values.
        pdf_values (numpy.ndarray): A 1D array of PDF values corresponding to `kappa_values`.

    Returns:
        tuple: Contains mean, variance, skewness, kurtosis, and normalization of the PDF.
    """
    norm = np.trapz(pdf_values, kappa_values)
    normalized_pdf_values = pdf_values / norm
    mean_kappa = np.trapz(kappa_values * normalized_pdf_values, kappa_values)
    variance = np.trapz(
        (kappa_values - mean_kappa) ** 2 * normalized_pdf_values, kappa_values
    )
    third_moment = np.trapz(
        (kappa_values - mean_kappa) ** 3 * normalized_pdf_values, kappa_values
    )
    fourth_moment = np.trapz(
        (kappa_values - mean_kappa) ** 4 * normalized_pdf_values, kappa_values
    )
    S_3 = third_moment / variance**2.0
    K = fourth_moment / variance**2 - 3
    return mean_kappa, variance, S_3, K, norm


def get_l1_from_pdf(counts, bins):
    """
    Calculates the L1 norm from a probability distribution function represented as a histogram.

    Parameters:
        counts (numpy.ndarray): The counts or heights of the histogram bins.
        bins (numpy.ndarray): The values of the bins.

    Returns:
        numpy.ndarray: L1 norm of the PDF represented by the histogram.
    """
    return counts * np.abs(bins)
