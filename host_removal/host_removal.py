'''
Host galaxy removal class
'''
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import solve
from scipy.optimize import lsq_linear
from host_removal.util.conversions import nm_to_A, align_spec_wave
from host_removal.util.input_output import load_gal_eigenspec, load_sn_templates


class HostGalaxyRemoval:
    '''
    Models the given SN Ia spectrum using SALT SN Ia templates and
    galaxy eigenspectra.
    The SN Ia spectrum must be in rest frame (i.e deredshifted), normalised
    such that the flux is between 0 and 1.  
    '''
    def __init__(self, obs_spec, sn_rest_phase, fit_wl_bounds=[4000, 6000], spec_keys=["x", "y", "z"], min_eigenspec=3, max_eigenspec=10, **kwargs):
        self.obs_spec = obs_spec
        self.sn_phase = sn_rest_phase
        self.spec_keys = spec_keys
        self.fit_lower_wl = fit_wl_bounds[0]
        self.fit_upper_wl = fit_wl_bounds[1]
        self.min_eigenspec = min_eigenspec
        self.max_eigenspec = max_eigenspec

        wl_mask = (self.obs_spec[self.spec_keys[0]].value > self.fit_lower_wl) & (self.obs_spec[self.spec_keys[0]].value < self.fit_upper_wl)
        self.obs_spec_trimmed = self.obs_spec[wl_mask]

        self.sn_templates = None
        self.gal_eigenspec = None
        self.gal_eigenvals = None
        self.design_matrix = None
        self.weights = None
        self.gal_model = None
        self.sn_model = None
        self.spec_model = None
        self.spec_model_params = None
        self.obs_spec_gal_subtracted = None


    def fit_spectrum(self):
        if self.sn_templates is None:
            self._obtain_sn_templates()
        if self.gal_eigenspec is None:
            self._obtain_gal_eigenspec()
        if self.weights is None:
            self._get_default_weights()

        best_bic = np.inf
        for sn_template in self.sn_templates:
            for num_eigenspec in range(self.min_eigenspec, min(len(self.gal_eigenspec), self.max_eigenspec)  + 1):
                gal_eigenspec = self.gal_eigenspec[:num_eigenspec]

                design_matrix = self._design_matrix(sn_template, gal_eigenspec)
                design_matrix_weighted = design_matrix * self.weights[:, None]
                spec_weighted = self.obs_spec_trimmed[self.spec_keys[1]].value * self.weights

                # Normal equations
                m = design_matrix_weighted.T @ design_matrix_weighted
                b = design_matrix_weighted.T @ spec_weighted
                x = solve(m, b)

                # Construct the fitted model spectra (host, sn). Use the unweighted design matrix.
                sn_model = np.ravel(design_matrix[:, num_eigenspec:] @ x[num_eigenspec:])
                sn_model_flux_tot = np.trapz(sn_model, self.obs_spec_trimmed[self.spec_keys[0]])
                gal_model = np.ravel(design_matrix[:, :num_eigenspec] @ x[:num_eigenspec])
                gal_model_flux_tot = np.trapz(gal_model, self.obs_spec_trimmed[self.spec_keys[0]])

                # SN and galaxy models must independently have flux > 0
                if gal_model_flux_tot < 0:
                    continue
                if np.any(sn_model < 0):
                    continue

                spec_model = np.ravel(design_matrix @ x)
                chi2 = np.sum(((self.obs_spec_trimmed[self.spec_keys[1]].value - spec_model) / self.weights)**2)
                bic = chi2  + (3 + num_eigenspec) * np.log(len(self.weights))  # Chi^2 + number of model parameters * ln(number of data points)

                if bic < best_bic:
                    best_bic = bic
                    self.design_matrix = design_matrix
                    self.sn_model_params = x[num_eigenspec:]
                    self.gal_eigenvals = list(x[:num_eigenspec]) + [0] * (len(self.gal_eigenspec) - num_eigenspec)  # Add 0s for all unfitted eigenspectra
                    self.sn_model = sn_model * self.obs_spec_trimmed[self.spec_keys[1]].unit #TODO "Unit handling issue"
                    self.gal_model = gal_model * self.obs_spec_trimmed[self.spec_keys[1]].unit
                    self.spec_model = spec_model
        print(f"Best bic for entire fitting: {best_bic}")


    def remove_galaxy_contamination(self):
        '''
        Remove the fitted galaxy spectrum from the observed data and return
        the value.
        If the galaxy spectrum has not yet been determined then run the fitting
        process.
        '''

        if self.gal_model is None:
            self.fit_spectrum()

        self.obs_spec_gal_subtracted = self.obs_spec_trimmed.copy()
        self.obs_spec_gal_subtracted[self.spec_keys[1]] = self.obs_spec_gal_subtracted[self.spec_keys[1]] - self.gal_model
        return self.obs_spec_gal_subtracted


    def plot_fit(self, plot_host_free_spec=True, plot_gal_components=True, show=True):
        n_subplots = 2
        axes_ind = 2
        if plot_gal_components:
            n_subplots += 1
        if plot_host_free_spec:
            n_subplots += 1
        
        fig, axes = plt.subplots(n_subplots, 1, sharex=True)
        
        if self.spec_model is not None:
            axes[0].plot(self.obs_spec[self.spec_keys[0]], self.obs_spec[self.spec_keys[1]], label="Observed Spectrum")
            axes[0].plot(self.obs_spec_trimmed[self.spec_keys[0]], self.spec_model, label="Model")
            axes[0].plot(self.obs_spec_trimmed[self.spec_keys[0]], self.sn_model, label="Model (SN)")
            axes[0].plot(self.obs_spec_trimmed[self.spec_keys[0]], self.gal_model, label="Model (galaxy)")
            axes[0].legend()
            # Residual plot
            axes[1].axhline(0, 0, 1, c="k")
            axes[1].plot(self.obs_spec_trimmed[self.spec_keys[0]], self.obs_spec_trimmed[self.spec_keys[1]] - self.spec_model, label="Residual (Observed - Model)")
            axes[1].legend()

            if plot_host_free_spec and self.obs_spec_gal_subtracted is not None:
                axes[axes_ind].plot(self.obs_spec_gal_subtracted[self.spec_keys[0]], self.obs_spec_gal_subtracted[self.spec_keys[1]], label="Host Free Observed Spectrum")
                axes[axes_ind].plot(self.obs_spec_gal_subtracted[self.spec_keys[0]], self.gal_model, label="Model (galaxy)")
                axes[axes_ind].legend()
                axes_ind += 1

            if plot_gal_components:
                axes[axes_ind].plot(self.obs_spec_trimmed[self.spec_keys[0]], self.gal_model, label="Model (galaxy)")
                if len(self.gal_eigenvals) > 0:
                    for i, eigenspec in enumerate(self.gal_eigenspec):
                        axes[axes_ind].plot(self.obs_spec_trimmed[self.spec_keys[0]],
                                            np.dot(self.gal_eigenvals[i], (eigenspec[self.spec_keys[1]] * self.weights)),  # Must weight eigenspec as eigenvals were found using wieghts
                                            label=f"Model (eigenspec: {i+1}, eigenval: {self.gal_eigenvals[i]:.2E})")
                    axes[axes_ind].legend()
        else:
            print("Fitting failed! \nCould not plot model spectra.")

        if show:
            plt.show()


    def _obtain_sn_templates(self):
        '''
        Load in the SN templates and align them to the same wavelengths
        as the SN spectrum.
        '''
        sn_templates = load_sn_templates(self.sn_phase)
        sn_templates_aligned = []
        for spec in sn_templates:
            for z_diff in [-0.035, -0.03, -0.025, -0.02, -0.015, -0.01, -0.005, -0.0025, -0.001, 0, 0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035]:#[-0.035, -0.025, -0.015, -0.005, 0, 0.005, 0.015, 0.025, 0.035, 0.045]:
                if z_diff > 0:
                    spec["wave"] /= (1 + z_diff)  # Blueshift the spectrum
                    spec_aligned = align_spec_wave(self.obs_spec_trimmed, spec, keys1=self.spec_keys)  # Note: spec_aligned has same keys as self.spec_keys
                elif z_diff < 0:
                    spec["wave"] *= (1 + abs(z_diff))  # Redshift the spectrum
                    spec_aligned = align_spec_wave(self.obs_spec_trimmed, spec, keys1=self.spec_keys)
                elif z_diff == 0:
                    spec_aligned = align_spec_wave(self.obs_spec_trimmed, spec, keys1=self.spec_keys)
                sn_templates_aligned.append(spec_aligned)
        self.sn_templates = sn_templates_aligned


    def _obtain_gal_eigenspec(self):
        '''
        Load in the galaxy eigenspectra and align them to the same wavelengths
        as the SN spectrum.
        '''
        gal_eigenspec = load_gal_eigenspec()
        gal_eigenspec_aligned = []
        for spec in gal_eigenspec:
            gal_eigenspec_aligned.append(align_spec_wave(self.obs_spec_trimmed, spec, keys1=self.spec_keys))
        self.gal_eigenspec = gal_eigenspec_aligned


    def _wdot(self, x,y, sigma):
        return np.sum(x * y / (sigma**2))


    def _define_sn_template_polynomial(self, wl_fixed=6600):
        '''
        Define the polynomial used to scale the SN template
        polynomial is setup so that at the given wavelength, wl_fix,
        the polynomial is always 1.
        Therefore, the polynomial s = p0 + c1 * p1 + c2 * p3**2
        Where c0 is 1, c1 and c2 are coefficients to be determined, p1 is
        x term, and p2 is x^2 term. 
        The x and x^2 terms are defined with respect to wl_fixed
        (i.e p1 = d_wl, p2 = d_wl**2).
        '''
        wl = self.obs_spec_trimmed[self.spec_keys[0]].value
        d_wl = wl - wl_fixed

        p0 = np.ones_like(wl)
        p1 = d_wl
        p2 = d_wl**2

        return p0, p1, p2


    def _design_matrix(self, sn_template, gal_eigenspec):
        gal_eigenspec_flux = [spec[self.spec_keys[1]] for spec in gal_eigenspec]
        sn_poly = self._define_sn_template_polynomial()
        design_matrix = np.vstack([gal_eigenspec_flux, 
                                   sn_template[self.spec_keys[1]] * sn_poly[0],
                                   sn_template[self.spec_keys[1]] * sn_poly[1],
                                   sn_template[self.spec_keys[1]] * sn_poly[2]]).T
        return design_matrix


    def _get_default_weights(self):
        error_weights = 1 / self.obs_spec_trimmed[self.spec_keys[2]]
        # halpha_weights = 1 + (5 * np.exp(-0.5 * ((self.obs_spec_trimmed[self.spec_keys[0]].value - 6563) / 3)**2))
        # self.weights = error_weights * halpha_weights
        self.weights = error_weights
