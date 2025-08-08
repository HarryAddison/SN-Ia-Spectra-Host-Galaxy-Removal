import astropy.units as u
from scipy.interpolate import interp1d


def nm_to_A(wave):
    return wave.to(u.Angstrom)


def normalise_data(data, val):
    data = data / val
    return data


def align_spec_wave(spec1, spec2, method="linear", keys=["wave", "flux"]):
    spec2_interp = spec1[keys].copy()
    spec2_func = interp1d(spec2[keys[0]], spec2[keys[1]], kind=method, fill_value="extrapolate")

    # Interpolate y-values
    spec2_interp["flux"] = spec2_func(spec1[keys[0]])

    return spec2_interp