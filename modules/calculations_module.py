from imports import *

def calculate_moments(x, P):
    """
    Calculates the statistical moments (mean, variance, skewness) and normalization of a probability distribution.

    Parameters:
        x (numpy.ndarray): A 1D array of variable values.
        P (numpy.ndarray): A 1D array of probability densities corresponding to `x`.

    Returns:
        tuple: Contains mean, variance, skewness, and normalization of the distribution.
    """
    norm = simps(P, x)
    mean_x = simps(x * P, x) / norm
    variance_x = simps(x**2 * P, x) / norm - mean_x**2
    skewness_x = (simps(x**3 * P, x) / norm - 3 * mean_x * variance_x - mean_x**3) / variance_x**1.5
    return mean_x, variance_x, skewness_x, norm

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
    variance = np.trapz((kappa_values - mean_kappa)**2 * normalized_pdf_values, kappa_values)
    third_moment = np.trapz((kappa_values - mean_kappa)**3 * normalized_pdf_values, kappa_values)
    fourth_moment = np.trapz((kappa_values - mean_kappa)**4 * normalized_pdf_values, kappa_values)
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

def find_smallest_pair(critical_values):
    """
    Finds the pair of points with the smallest Euclidean distance between them from a set of critical values.

    Parameters:
        critical_values (numpy.ndarray): An array of critical points.

    Returns:
        tuple: The pair of points with the smallest distance and their Euclidean distance.
    """
    num_points = critical_values.shape[0]
    if num_points < 2:
        return None, float('inf')  # No pair exists

    smallest_distance = float('inf')
    smallest_pair = None

    for i in range(num_points - 1):
        for j in range(i + 1, num_points):
            distance = np.linalg.norm(critical_values[i] - critical_values[j])
            if distance < smallest_distance:
                smallest_distance = distance
                smallest_pair = (critical_values[i], critical_values[j])

    return smallest_pair


def top_hat_filter(k,R):
    """
    Calculates the top-hat window function for a given radius.
    
    Parameters:
        R (float or numpy.ndarray): The scale (or array of scales) at which to calculate the window function.
        
    Returns:
        numpy.ndarray: The top-hat window function values at the given scale(s).
    """
    return 2. * scipy.special.j1(k*R) /(k* R)

def get_W2D_FL(window_radius, map_shape, filter_type, L=505):
    """
    Constructs a 2D Fourier-space window function for a top-hat filter.
    
    Parameters:
        window_radius : float
            The top-hat window radius in physical units (must be consistent with L).
        map_shape     : tuple
            Shape of the map (assumed square, e.g. (600,600)).
        L             : float, optional
            Physical size of the map (default is 505, as used for SLICS).
    
    Returns:
        2D numpy array representing the Fourier-space window.
    """
    N = map_shape[0]
    dx = N / N
    # Generate Fourier frequencies.
    kx = np.fft.fftshift(np.fft.fftfreq(N, dx))
    ky = np.fft.fftshift(np.fft.fftfreq(N, dx))
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k2 = kx**2 + ky**2
    # Convert to radial wavenumber (with 2pi factor).
    k = 2 * np.pi * np.sqrt(k2)
    # Avoid division by zero at the center.
    ind = int(N / 2)
    k[ind, ind] = 1e-7
    if filter_type == 'tophat':
        return top_hat_filter(k, window_radius)
    elif filter_type == 'starlet':
        print("Getting starlet W2D_FL")
        return uHat_starlet_analytical(k, window_radius)

def get_smoothed_app_pdf(mass_map, window_radius, binedges, filter_type, L=505):
    """
    Applies top-hat smoothing in Fourier space at two scales and returns the PDF of the difference map.
    
    The map is filtered with a top-hat window of radius R and 2R, then the difference is computed.
    
    Parameters:
        mass_map     : 2D numpy array.
        window_radius: The smoothing scale (R) in physical units.
        binedges     : Bin edges for the histogram.
        L            : Physical size of the map (default 505 MPC/h).
    
    Returns:
        tuple : (bin_edges, pdf_counts, difference_map)
    """
    N = mass_map.shape[0]
    # Compute the Fourier-space top-hat windows.
    # print("Getting W2D_FL")
    if filter_type == 'tophat':
        W2D_1 = get_W2D_FL(window_radius, mass_map.shape, 'tophat', L)
        W2D_2 = get_W2D_FL(window_radius * 2, mass_map.shape, 'tophat', L)
        
        # Fourier transform the input mass map.
        field_ft = np.fft.fftshift(np.fft.fftn(mass_map))
        
        # Apply the window functions in Fourier space.
        smoothed_ft1 = field_ft * W2D_1
        smoothed_ft2 = field_ft * W2D_2
        
        # Inverse Fourier transform to get back to real space.
        smoothed1 = np.fft.ifftn(np.fft.ifftshift(smoothed_ft1)).real
        smoothed2 = np.fft.ifftn(np.fft.ifftshift(smoothed_ft2)).real
        
        # Compute the difference map.
        difference_map = smoothed2 - smoothed1
    elif filter_type == "starlet":
        W2D_1 = get_W2D_FL(window_radius, mass_map.shape, 'starlet', L)
        print("got the starlet W2D_1")
        # Fourier transform the input mass map.
        field_ft = np.fft.fftshift(np.fft.fftn(mass_map))
        # Apply the window functions in Fourier space.
        smoothed_ft1 = field_ft * W2D_1        
        # Inverse Fourier transform to get back to real space.
        smoothed1 = np.fft.ifftn(np.fft.ifftshift(smoothed_ft1)).real
        # Compute the difference map.
        difference_map = -smoothed1
    
    counts, _ = np.histogram(difference_map, bins=binedges, density=True)
    return binedges, counts, difference_map

    


import numpy as np
import scipy.special as sp
import mpmath as mp
from functools import lru_cache

# Fast memoized scalar S function
@lru_cache(maxsize=None)
def S_scalar(n: int, b: float) -> float:
    if n < -1:
        raise ValueError("n cannot be smaller than -1.")

    J0 = sp.j0(b)
    J1 = sp.j1(b)

    if n == 0:
        return b * J1
    elif n == -1:
        return b * float(mp.hyp1f2(0.5, 1, 1.5, -b**2 / 4))
    else:
        return b**(n+1) * J1 + n * b**n * J0 - n**2 * S_scalar(n-2, b)

# Wrapper to handle arrays
def S(n: int, b):
    b = np.asarray(b)
    if b.ndim == 0:
        return S_scalar(n, float(b))
    else:
        vec_func = np.vectorize(lambda x: S_scalar(n, float(x)))
        return vec_func(b)

# Fast uHat_starlet_analytical
def uHat_starlet_analytical(eta, R):
    """
    Computes the analytical Hankel transform of the starlet U-filter.

    Args:
        eta (np.ndarray or float): Dimensionless argument \( \hat{u} \).

    Returns:
        float or np.ndarray: Computed \( \hat{u} \).
    """
    print("Calculating uHat_starlet_analytical (optimized version)")
    
    eta = np.asarray(eta) * R
    eta_safe = np.clip(eta, 2e-2, 100)  # Stability for small eta

    # Precompute all needed S values
    b_half = 0.5 * eta_safe
    b_one = eta_safe
    b_two = 2.0 * eta_safe

    S0_half = S(0, b_half)
    S1_half = S(1, b_half)
    S2_half = S(2, b_half)
    S3_half = S(3, b_half)

    S0_one = S(0, b_one)
    S1_one = S(1, b_one)
    S2_one = S(2, b_one)
    S3_one = S(3, b_one)

    S0_two = S(0, b_two)
    S1_two = S(1, b_two)
    S2_two = S(2, b_two)
    S3_two = S(3, b_two)

    # Compute factors
    factor1 = (0.125 * eta_safe**3 * S0_half 
               - 0.75 * eta_safe**2 * S1_half 
               + 1.5 * eta_safe * S2_half 
               - S3_half)
    print("done factor1")
    factor2 = (eta_safe**3 * S0_one 
               - 3 * eta_safe**2 * S1_one 
               + 3 * eta_safe * S2_one 
               - S3_one)
    print("done factor2")
    factor3 = (8 * eta_safe**3 * S0_two 
               - 12 * eta_safe**2 * S1_two 
               + 6 * eta_safe * S2_two 
               - S3_two)
    print("done factor3")
    # Final result
    result = (2 * np.pi) * (-128/9 * factor1 + 4 * factor2 - 1/9 * factor3) / eta_safe**5

    return result
