import jax
import jax.numpy as jnp
from jax.numpy.fft import rfft2
from .utils import fourier_coordinate
import numpy as np

def calculate_Cls(map, angle, ell_min, ell_max, n_bins):
    """
    map: the image from which the angular power spectra (Cls) has to be calculated
    angle: side angle in the units of degree
    ell_min: the minimum multipole moment to get the Cls
    ell_max: the maximum multipole moment to get the Cls
    n_bins: number of bins in the ells
    """
    angle = jnp.array(angle)
    ell_min = jnp.array(ell_min)
    ell_max = jnp.array(ell_max)
    n_bins = jnp.array(n_bins, int)

    # Calculate the Fourier Transforms
    map_ft = rfft2(map)  ## rfft2
    map_ft = map_ft.flatten()
    ell_edges = jnp.linspace(ell_min, ell_max, num=n_bins + 1)

    # Define pixel physical size in Fourier space
    lpix = 360 / angle
    # Initialize arrays to store power and hits for each ell bin
    power_l = jnp.zeros(n_bins)
    hits = jnp.zeros(n_bins)

    def loop_body(j, val):
        i, power_l, hits = val
        lx = jnp.minimum(i, map.shape[1] - i) * lpix
        ly = j * lpix
        l = jnp.sqrt(lx**2.0 + ly**2.0)
        pixid = fourier_coordinate(i, j, map.shape[0])
        bin_idx = jnp.digitize(l, ell_edges)  # - 1
        power_l = power_l.at[bin_idx].add(jnp.abs(map_ft[pixid] ** 2.0))
        hits = hits.at[bin_idx].add(1)
        return i, power_l, hits

    def outer_loop_body(i, val):
        _, power_l, hits = val
        _, power_l, hits = jax.lax.fori_loop(
            0, map.shape[0], loop_body, (i, power_l, hits)
        )
        return i, power_l, hits

    _, power_l, hits = jax.lax.fori_loop(
        0, map.shape[1], outer_loop_body, (0, power_l, hits)
    )

    # Calculate Cls based on the accumulated power and hits
    cls_values = jnp.where(hits > 0, power_l / hits, 0.0)  # Ensure no division by zero
    cls_values = jnp.maximum(cls_values, 0)  # Clip negative values to zero if any
    ell_bins = 0.5 * (ell_edges[1:] + ell_edges[:-1])
    normalization = (angle * np.pi / 180.0 / (map.shape[0] * map.shape[0])) ** 2.0
    return (
        jnp.array(ell_edges),
        jnp.array(ell_bins),
        jnp.array(cls_values * normalization),
    )
