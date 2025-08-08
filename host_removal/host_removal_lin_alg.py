import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import QTable
from scipy.interpolate import interp1d
from scipy.optimize import lsq_linear


# -----------------------------
# Helper functions
# -----------------------------

def model_gal_spec(eigenspec, eigenvalues):
    model_spec = np.dot(eigenvalues, eigenspec)
    return model_spec


def load_gal_eigenspec(num_eigenspec=10):

    eigenspec = []
    for i in range(num_eigenspec):
        path = f"../data/galaxy-eigenspectra/galaxyKL_eigSpec_{i+1}.dat"
        eigenspec.append(QTable.read(path, format="ascii", names=["wave", "flux"]))

    return eigenspec


def find_files(dir, pattern):
    matching_files = [dir+f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.startswith(pattern)]
    return matching_files


def load_sn_templates(obs_phase, phase_diff=5, dir="../data/sn-templates/"):

    sn_templates = []
    for i in range((phase_diff * 2) + 1):
        phase = obs_phase - phase_diff + i  # gives phases between obs_phase +- phase_diff
        file_paths = find_files(dir, pattern=f"sn_template_phase_{phase}")
        for path in list(file_paths):
            # TODO Remove the four lines below and uncomment the last line
            # These are temporary as I was testing using a random template I had.
            # In actuality I want to use templates are at z=0 and are normalised.
            template = QTable.read(path)
            template["wave"] /= 1.01
            template["flux"] /= max(template["flux"])
            sn_templates.append(template)
            # sn_templates.append(QTable.read(path))

    return sn_templates


def align_spec_wave(spec1, spec2, method="linear", keys=["wave", "flux"]):
    spec2_interp = spec1.copy()
    spec2_func = interp1d(spec2[keys[0]], spec2[keys[1]], kind=method, fill_value="extrapolate")

    # Interpolate y-values
    spec2_interp["flux"] = spec2_func(spec1[keys[0]])

    return spec2_interp


if __name__ == "__main__":

    obs_spec = QTable.read("/vol/ph/astro_data/haddison/TiDES-spectral-analysis/TiDES-spextractor-testing/host-contamination-testing/data/mock-spectra/l1-spectra/l1_spectrum_1.fits")
    obs_phase = 0
    z = 0.01
    obs_spec["wave"] /= (1 + z)
    max_flux = max(obs_spec["flux"])
    obs_spec["flux"] /= max_flux
    obs_spec["flux_err"] /= max_flux
    wl_mask = (obs_spec["wave"].value > 4000) & (obs_spec["wave"].value < 7000)

    sn_templates = load_sn_templates(obs_phase)
    sn_templates_aligned_flux = []
    for spec in sn_templates:
        sn_templates_aligned_flux.append(align_spec_wave(obs_spec[wl_mask], spec)["flux"])

    gal_eigenspec = load_gal_eigenspec()
    gal_eigenspec_aligned_flux = []
    for spec in gal_eigenspec:
        gal_eigenspec_aligned_flux.append(align_spec_wave(obs_spec[wl_mask], spec)["flux"])

    # Define polynomial basis (centered at 6000 Å)
    wavelength = obs_spec[wl_mask]["wave"].value
    p0 = np.ones_like(wavelength)
    p1 = wavelength - 6000
    p2 = (wavelength - 6000) ** 2

    # TODO Add a loop over SN templates here.
    # Design matrix: [SN*p0, SN*p1, SN*p2, galaxy_pc1, galaxy_pc2...]
    design_matrix = np.vstack([
        sn_templates_aligned_flux * p0,
        sn_templates_aligned_flux * p1,
        sn_templates_aligned_flux * p2,
        gal_eigenspec_aligned_flux]).T

    # Weighted least squares solution
    weights = 1 / obs_spec[wl_mask]["flux_err"].value
    a = design_matrix# * weights[:, np.newaxis]
    b = obs_spec[wl_mask]["flux"].value# * weights

    # Set bounds of parameters. Only require that the eigenvalues are positive.
    lower_bounds = np.concatenate([[-np.inf] * 3, [0.0] * len(gal_eigenspec_aligned_flux)])
    upper_bounds = np.full(a.shape[1], np.inf)
    
    result = lsq_linear(a, b, bounds=(lower_bounds, upper_bounds), verbose=2)
    coeffs = result.x
    print(result.cost)

    # Extract fitted SN and galaxy components
    sn_fit = (coeffs[0] * p0 + coeffs[1] * p1 + coeffs[2] * p2) * sn_templates_aligned_flux
    galaxy_fit = np.dot(coeffs[3:], gal_eigenspec_aligned_flux)
    fit_total = sn_fit + galaxy_fit
    
    # --- Plotting Results ---
    fig, axes = plt.subplots(2,1, sharex=True)
    axes[0].plot(obs_spec["wave"], obs_spec["flux"], label="Observed")
    axes[0].plot(wavelength, np.ravel(fit_total), label="Model")
    axes[0].plot(wavelength, np.ravel(sn_fit), label="Model (SN)")
    axes[0].plot(wavelength, np.ravel(galaxy_fit), label="Model (galaxy)")
    for i, spec in enumerate(gal_eigenspec_aligned_flux):
        axes[0].plot(wavelength, np.dot(coeffs[3+i], spec), label=f"Model (eigenspec {i})")
    axes[0].legend()

    axes[1].plot(obs_spec["wave"][wl_mask], obs_spec["flux"][wl_mask] - np.ravel(fit_total), label="Residual (Observed - Model)")
    axes[1].axhline(0, 0, 1, c="k")
    plt.legend()

    plt.show()
