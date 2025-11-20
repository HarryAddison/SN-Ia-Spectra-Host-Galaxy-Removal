'''
Host galaxy removal class
'''
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
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
    def __init__(self, obs_spec, sn_rest_phase, fit_wl_bounds=[4000*u.Angstrom, 6000*u.Angstrom], keys=["x", "y", "z"], min_eigenspec=3, max_eigenspec=10, **kwargs):
        self.obs_spec = obs_spec
        self.sn_phase = sn_rest_phase
        self.keys = keys
        self.min_eigenspec = min_eigenspec
        self.max_eigenspec = max_eigenspec

        for i, bound in enumerate(fit_wl_bounds):
            if type(bound) is str:
                fit_wl_bounds[i] = u.Quantity(bound)
            if type(fit_wl_bounds[i]) is not u.Quantity:
                raise TypeError("The provided fit_wl_bounds does not contain units. Bounds must be provided with units.")
        self.fit_wl_bounds = fit_wl_bounds
        self.weights = None
        self.sn_templates = None
        self._obtain_sn_templates()
        self.gal_eigenspec = load_gal_eigenspec()

        self.com_wl_range = None
        self._find_common_wavelengths()
        if fit_wl_bounds[0] < self.com_wl_range[0]:
            raise ValueError("The lower fitting wavelength bound given is less than the minimum wavelength of the observed spectrum or the templates used in the fitting. Please increase the minimum wavelength used in the fitting.")
        if fit_wl_bounds[1] > self.com_wl_range[1]:
            raise ValueError("The upper fitting wavelength bound given is greater than the maximum wavelength of the observed spectrum or the templates used in the fitting. Please decrease the maximum wavelength used in the fitting.")


        # Align the observed spectrum, sn templates, and galaxy eigenvectors to common wavelengths
        self._align_wavelengths()

        self._get_default_weights()

        self.wl_fit_mask = (self.obs_spec[self.keys[0]] > self.fit_wl_bounds[0]) & (self.obs_spec[self.keys[0]] < self.fit_wl_bounds[1])
        self.gal_eigenvals = None
        self.design_matrix = None
        self.gal_model = None
        self.sn_model = None
        self.spec_model = None
        self.spec_model_params = None
        self.obs_spec_gal_subtracted = None


    def fit_spectrum(self):

        best_bic = np.inf
        for sn_template in self.sn_templates:
            for num_eigenspec in range(self.min_eigenspec, min(len(self.gal_eigenspec), self.max_eigenspec)  + 1):
                gal_eigenspec = self.gal_eigenspec[:num_eigenspec]

                design_matrix = self._design_matrix(sn_template, gal_eigenspec)
                design_matrix_weighted = design_matrix * self.weights[:, None]
                spec_weighted = self.obs_spec[self.keys[1]].value * self.weights

                # Normal equations (apply wavelength mask to limit fit to the given region)
                m = design_matrix_weighted[self.wl_fit_mask, :].T @ design_matrix_weighted[self.wl_fit_mask, :]
                b = design_matrix_weighted[self.wl_fit_mask, :].T @ spec_weighted[self.wl_fit_mask]
                x = solve(m, b)

                # Construct the fitted model spectra (host, sn). Use the unweighted design matrix.
                # Only contruct model in fitted wavelength range using "wl_mask"
                sn_model = np.ravel(design_matrix[self.wl_fit_mask, num_eigenspec:] @ x[num_eigenspec:])
                gal_model = np.ravel(design_matrix[self.wl_fit_mask, :num_eigenspec] @ x[:num_eigenspec])
                gal_model_flux_tot = np.trapz(gal_model, self.obs_spec[self.keys[0]][self.wl_fit_mask])

                # SN and galaxy models must independently have flux > 0
                if gal_model_flux_tot < 0:
                    continue
                if np.any(sn_model < 0):
                    continue

                spec_model = np.ravel(design_matrix[self.wl_fit_mask] @ x) * self.obs_spec[self.keys[1]].unit
                chi2 = np.sum(((self.obs_spec[self.keys[1]][self.wl_fit_mask] - spec_model) * self.weights[self.wl_fit_mask])**2)
                bic = chi2  + (3 + num_eigenspec) * np.log(len(spec_model))  # Chi^2 + number of model parameters * ln(number of data points)

                if bic < best_bic:
                    best_bic = bic
                    self.design_matrix = design_matrix
                    self.sn_model_params = x[num_eigenspec:]
                    self.gal_eigenvals = list(x[:num_eigenspec]) + [0] * (len(self.gal_eigenspec) - num_eigenspec)  # Add 0s for all unfitted eigenspectra
                    
                    # Full wavelength models
                    sn_model_flux = np.ravel(design_matrix[:, num_eigenspec:] @ x[num_eigenspec:]) * self.obs_spec[self.keys[1]].unit
                    gal_model_flux = np.ravel(design_matrix[:, :num_eigenspec] @ x[:num_eigenspec]) * self.obs_spec[self.keys[1]].unit
                    spec_model_flux = np.ravel(design_matrix @ x) * self.obs_spec[self.keys[1]].unit

                    self.sn_model = QTable(names=self.keys[:2], data=[self.obs_spec[self.keys[0]], sn_model_flux])
                    self.gal_model = QTable(names=self.keys[:2], data=[self.obs_spec[self.keys[0]], gal_model_flux])
                    self.spec_model = QTable(names=self.keys[:2], data=[self.obs_spec[self.keys[0]], spec_model_flux])
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

        self.obs_spec_gal_subtracted = self.obs_spec.copy()
        self.obs_spec_gal_subtracted[self.keys[1]] = self.obs_spec_gal_subtracted[self.keys[1]] - self.gal_model[self.keys[1]]
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
            axes[0].plot(self.obs_spec[self.keys[0]], self.obs_spec[self.keys[1]], label="Observed Spectrum")
            axes[0].plot(self.spec_model[self.keys[0]], self.spec_model[self.keys[1]], label="Model")
            axes[0].plot(self.sn_model[self.keys[0]], self.sn_model[self.keys[1]], label="Model (SN)")
            axes[0].plot(self.gal_model[self.keys[0]], self.gal_model[self.keys[1]], label="Model (galaxy)")
            axes[0].legend()
            # Residual plot
            axes[1].axhline(0, 0, 1, c="k")
            axes[1].plot(self.obs_spec[self.keys[0]], self.obs_spec[self.keys[1]] - self.spec_model[self.keys[1]], label="Residual (Observed - Model)")
            axes[1].legend()

            if plot_host_free_spec and self.obs_spec_gal_subtracted is not None:
                flux_ratio = np.trapz(self.gal_model[self.keys[1]][self.wl_fit_mask], self.gal_model[self.keys[0]][self.wl_fit_mask]) / np.trapz(self.sn_model[self.keys[1]][self.wl_fit_mask], self.sn_model[self.keys[0]][self.wl_fit_mask])
                axes[axes_ind].plot(self.obs_spec_gal_subtracted[self.keys[0]], self.obs_spec_gal_subtracted[self.keys[1]], label="Host Free Observed Spectrum")
                axes[axes_ind].plot(self.gal_model[self.keys[0]], self.gal_model[self.keys[1]], label=f"Model (galaxy) ({flux_ratio} x SN flux)")
                axes[axes_ind].legend()
                axes_ind += 1

            if plot_gal_components:
                axes[axes_ind].plot(self.gal_model[self.keys[0]], self.gal_model[self.keys[1]], label="Model (galaxy)")
                if len(self.gal_eigenvals) > 0:
                    for i, eigenspec in enumerate(self.gal_eigenspec):
                        axes[axes_ind].plot(eigenspec[self.keys[0]],
                                            np.dot(self.gal_eigenvals[i], eigenspec[self.keys[1]]),
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
        sn_templates_shifted = []
        for spec in sn_templates:
            for z_diff in [-0.035, -0.03, -0.025, -0.02, -0.015, -0.01, -0.005, -0.0025, -0.001, 0, 0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035]:
                spec_shifted = spec.copy()
                if z_diff > 0:
                    spec_shifted["wave"] /= (1 + z_diff)  # Blueshift the spectrum
                elif z_diff < 0:
                    spec_shifted["wave"] *= (1 + abs(z_diff))  # Redshift the spectrum
                sn_templates_shifted.append(spec_shifted)
        self.sn_templates = sn_templates_shifted


    def _find_common_wavelengths(self):

        # Find the common wavelengths between the observed spectrum, galaxy eigenspectra,
        # and SN templates.
        # Galaxy eigenspectra and SN templates always have same wavelengths.

        # Common wl range of SN templates
        sn_min_wl = max(np.min([spec[self.keys[0]] for spec in self.sn_templates], axis=1)) * self.sn_templates[0][self.keys[0]].unit
        sn_max_wl = min(np.max([spec[self.keys[0]] for spec in self.sn_templates], axis=1)) * self.sn_templates[0][self.keys[0]].unit

        # wl range of eigenspec (Use first eigenspec)
        gal_min_wl = min(self.gal_eigenspec[0][self.keys[0]])
        gal_max_wl = max(self.gal_eigenspec[0][self.keys[0]])

        # wl range of observed spectrum
        obs_min_wl = min(self.obs_spec[self.keys[0]])
        obs_max_wl = max(self.obs_spec[self.keys[0]])

        # Common wavelength range between SN template, galaxy eigenspec and observed spectrum
        min_wl = max([sn_min_wl, gal_min_wl, obs_min_wl])
        max_wl = min([sn_max_wl, gal_max_wl, obs_max_wl])

        self.com_wl_range = [min_wl, max_wl]

    
    def _align_wavelengths(self):

        wl_mask = (self.obs_spec[self.keys[0]] > self.com_wl_range[0]) & (self.obs_spec[self.keys[0]] < self.com_wl_range[1])
        common_wls = self.obs_spec[wl_mask][self.keys[0]]

        self.obs_spec = self.obs_spec[wl_mask]

        gal_eigenspec_aligned = []
        for spec in self.gal_eigenspec:
            gal_eigenspec_aligned.append(align_spec_wave(common_wls, spec, keys=self.keys))
        self.gal_eigenspec = gal_eigenspec_aligned

        sn_templates_aligned = []
        for spec in self.sn_templates:
            sn_templates_aligned.append(align_spec_wave(common_wls, spec, keys=self.keys))
        self.sn_templates = sn_templates_aligned


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
        wl = self.obs_spec[self.keys[0]].value
        d_wl = wl - wl_fixed

        p0 = np.ones_like(wl)
        p1 = d_wl
        p2 = d_wl**2

        return p0, p1, p2


    def _design_matrix(self, sn_template, gal_eigenspec):
        gal_eigenspec_flux = [spec[self.keys[1]].value for spec in gal_eigenspec]
        sn_poly = self._define_sn_template_polynomial()
        design_matrix = np.vstack([gal_eigenspec_flux, 
                                   sn_template[self.keys[1]].value * sn_poly[0],
                                   sn_template[self.keys[1]].value * sn_poly[1],
                                   sn_template[self.keys[1]].value * sn_poly[2]]).T
        return design_matrix


    def _get_default_weights(self):
        error_weights = 1 / self.obs_spec[self.keys[2]]
        # halpha_weights = 1 + (5 * np.exp(-0.5 * ((self.obs_spec_trimmed[self.keys[0]].value - 6563) / 3)**2))
        # self.weights = error_weights * halpha_weights
        self.weights = error_weights
