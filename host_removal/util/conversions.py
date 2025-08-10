import astropy.units as u
from scipy.interpolate import interp1d


def nm_to_A(wave):
    return wave.to(u.Angstrom)


def normalise_data(data, val):
    data = data / val
    return data


def align_spec_wave(spec1, spec2, method="linear", keys1=["wave", "flux"], keys2=["wave", "flux"]):

    spec2_interp = spec1[keys1].copy()
    spec2_func = interp1d(spec2[keys2[0]], spec2[keys2[1]], kind=method, fill_value="extrapolate")

    # Interpolate y-values
    spec2_interp[keys1[1]] = spec2_func(spec1[keys1[0]])

    return spec2_interp