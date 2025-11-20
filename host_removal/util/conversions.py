import astropy.units as u
from astropy.table import QTable
from scipy.interpolate import interp1d


def nm_to_A(wave):
    return wave.to(u.Angstrom)


def normalise_data(data, val):
    data = data / val
    return data


def align_spec_wave(wave_align, spec, method="linear", keys=["x", "y"], **kwargs):

    # Create a function describing the spectrum
    spec_func = interp1d(spec[keys[0]], spec[keys[1]], kind=method, fill_value="extrapolate")

    # Replace wavelengths with aligned and Interpolate flux values at these new
    # wavelengths
    flux_wave_aligned = spec_func(wave_align)

    spec_aligned = QTable(names=[keys[0], keys[1]], data=[wave_align, flux_wave_aligned])  # Need to specify keys[0, 1] as "keys" can have more items
    return spec_aligned
