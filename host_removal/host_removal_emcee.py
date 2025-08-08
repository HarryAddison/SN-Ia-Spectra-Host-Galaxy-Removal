import os
import numpy as np
import corner
import emcee
import matplotlib.pyplot as plt
from astropy.table import QTable
from scipy.interpolate import interp1d


def s_lambda(wavelengths, lambda1, c1, c2):
    delta = wavelengths - lambda1
    # Second degree polynomial defined such that at lambda1 the polynomial = 1
    # and that it is < 1 at any other x. (i.e c Must be negative)
    return 1 + c1 * delta + c2 * delta**2


def model_gal_spec(eigenspec, eigenvalues):
    model_spec = np.dot(eigenvalues, eigenspec)
    return model_spec


def model_spectrum(theta, wavelengths, f_sn, gal_eigenspec, lambda1=6600.0):
    n_gal = len(gal_eigenspec)
    a_0 = theta[0]
    a_gal = theta[1:1 + n_gal]
    c1 = theta[-2]
    c2 = theta[-1]

    s = s_lambda(wavelengths, lambda1, c1, c2)
    sn_component = a_0 * s * f_sn
    gal_component = model_gal_spec(gal_eigenspec, a_gal)
    return sn_component + gal_component, sn_component, gal_component


def log_prior(theta, wavelengths, gal_eigenspec, lambda1=6600.0):
    a_0 = theta[0]
    a_gal = theta[1:-2]
    c1 = theta[-2]
    c2 = theta[-1]

    if a_0 <= 0:
        return -np.inf
    if c2 > 1e-7:
        return -np.inf
    if abs(c1) > 1e-5:
        return -np.inf

    gal_spec = model_gal_spec(gal_eigenspec, a_gal)
    if np.any(gal_spec < 0):
        return -np.inf

    s = s_lambda(wavelengths, lambda1, c1, c2)
    if np.any(s > 1.0 + 1e-4):
        return -np.inf

    return 0


def log_likelihood(theta, obs_spec, f_sn, gal_eigenspec, keys=["x", "y", "z"]):
    model, _, _ = model_spectrum(theta, obs_spec[keys[0]].value, f_sn, gal_eigenspec)
    # chi2 = np.sum(((obs_spec[keys[1]].value - model) / obs_spec[keys[2]].value) ** 2)
    weights = np.ones_like(obs_spec[keys[0]].value)
    # h_alpha_mask = (obs_spec[keys[0]].value > 6540) & (obs_spec[keys[0]].value < 6600)
    # h_beta_mask = (obs_spec[keys[0]].value > 4830) & (obs_spec[keys[0]].value < 5050)
    # weights[h_alpha_mask] = 10
    # weights[h_beta_mask] = 10
    chi2 = np.sum(weights * (((model - obs_spec[keys[1]])**2) / obs_spec[keys[1]]))

    return -0.5 * chi2


def log_posterior(theta_norm, obs_spec, f_sn, gal_eigenspec, theta_scale, theta_initial, keys=["x", "y", "z"]):
    theta = theta_norm * theta_scale + theta_initial
    lp = log_prior(theta, obs_spec[keys[0]].value, gal_eigenspec)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, obs_spec, f_sn, gal_eigenspec, keys)


def run_emcee(obs_spec, f_sn, gal_eigenspec, n_walkers=100, keys=["x", "y", "z"], **kwargs):

    n_dim = 3 + len(gal_eigenspec)  # sn scale +  2 poly coeff + n gal eigenvalues

    # Initialize walkers
    # SN scale, gal eignval x10, polynomial coefficient x2
    # initial = np.array([0.75, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1e-8, -1e-12])
    # scale = np.array([0.5, 3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1e-8, 1e-13])
    initial = np.array([0.75, 1.0, 0.5, 0.5, 1e-6, -1e-8])
    scale = np.array([0.5, 3, 0.5, 0.5, 1e-5, 1e-9])
    p0 = np.empty((n_walkers, n_dim))
    for i in range(n_walkers):
        while True:
            trial = initial + scale * np.random.randn(n_dim)
            if np.all(trial[:-1] > 0):  # All except the last must be > 0
                p0[i] = trial
                break
    p0_norm = (p0 - initial) / scale

    # Run emcee
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior,
                                    args=(obs_spec, f_sn, gal_eigenspec, scale, initial, keys))
    sampler.run_mcmc(p0_norm, 5000, progress=True)

    # Store result
    samples_norm = sampler.get_chain(discard=3000, flat=True)
    samples = initial + scale * samples_norm
    log_prob = sampler.get_log_prob(discard=3000, flat=True)

    #TODO remove if its annoying...
    n_gal = len(gal_eigenspec)
    labels = [r"$a_{\mathrm{SN}}$"] + [f"$a_{{g,{i+1}}}$" for i in range(n_gal)] + ["c1", "c2"]
    fig = corner.corner(samples, labels=labels, show_titles=True, quantiles=[0.16, 0.5, 0.84],
                        title_kwargs={"fontsize": 12}, label_kwargs={"fontsize": 14})
    plt.tight_layout()

    samples_plot = sampler.get_chain(flat=False)
    samples_plot = initial + scale * samples_plot
    fig, axes = plt.subplots(n_dim, figsize=(10, 2 * n_dim), sharex=True)
    for i in range(n_dim):
        for j in range(n_walkers):
            axes[i].plot(samples_plot[:, j, i], alpha=0.4, c="k")
        axes[i].set_ylabel(f"param {i}")
    axes[-1].set_xlabel("Step number")
    plt.tight_layout()

    return samples, log_prob


def find_best_fit(obs_spec, sn_templates, gal_eigenspec, **kwargs):

    best_log_prob = -np.inf
    best_params = None
    best_sn_template = None

    for j, f_sn in enumerate(sn_templates):
        print(f"Running MCMC for SN template {j}")
        samples, log_prob = run_emcee(obs_spec, f_sn, gal_eigenspec, **kwargs)

        max_ind = np.unravel_index(np.argmax(log_prob), log_prob.shape)
        theta_max = samples[max_ind]
        log_prob_max = log_prob[max_ind]

        if log_prob_max > best_log_prob:
            best_log_prob = log_prob_max
            best_params = theta_max
            best_sn_template = f_sn

    return best_params, best_sn_template


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
    obs_spec["flux"] /= max(obs_spec["flux"])
    wl_mask = (obs_spec["wave"].value > 4000) & (obs_spec["wave"].value < 7000)

    sn_templates = load_sn_templates(obs_phase)
    sn_templates_aligned_flux = []
    for spec in sn_templates:
        sn_templates_aligned_flux.append(align_spec_wave(obs_spec[wl_mask], spec)["flux"])

    gal_eigenspec = load_gal_eigenspec()[:3]
    gal_eigenspec_aligned_flux = []
    for spec in gal_eigenspec:
        gal_eigenspec_aligned_flux.append(align_spec_wave(obs_spec[wl_mask], spec)["flux"])

    best_params, best_sn_template = find_best_fit(obs_spec[wl_mask], sn_templates_aligned_flux,
                                                  gal_eigenspec_aligned_flux, **{"keys": ["wave", "flux", "flux_err"]})

    model_spec, sn_component, gal_component = model_spectrum(best_params, obs_spec["wave"][wl_mask].value, best_sn_template, gal_eigenspec_aligned_flux)
    
    fig, axes = plt.subplots(2,1, sharex=True)
    axes[0].plot(obs_spec["wave"], obs_spec["flux"], label="Observed")
    axes[0].plot(obs_spec["wave"][wl_mask], model_spec, label="Model")
    axes[0].plot(obs_spec["wave"][wl_mask], sn_component, label="Model (SN)")
    axes[0].plot(obs_spec["wave"][wl_mask], gal_component, label="Model (galaxy)")
    n_gal = len(gal_eigenspec)
    for i, eigenval in enumerate(best_params[1:1 + n_gal]):
        spec = gal_eigenspec_aligned_flux[i] * eigenval
        axes[0].plot(obs_spec["wave"][wl_mask], spec, label=f"Model (galaxy component {i+1})")
    axes[0].legend()

    axes[1].plot(obs_spec["wave"][wl_mask], obs_spec["flux"][wl_mask] - model_spec, label="Residual (Observed - Model)")
    axes[1].axhline(0, 0, 1, c="k")
    plt.legend()

    plt.show()
