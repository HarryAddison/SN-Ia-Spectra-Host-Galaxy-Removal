'''
Host galaxy removal class
'''

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
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
    def __init__(self, sn_spec, sn_rest_phase, fit_wl_bounds=[3500, 7500], keys=["x", "y", "z"], **kwargs):
        self.sn_spec = sn_spec
        self.sn_phase = sn_rest_phase
        self.sn_keys = keys
        self.fit_lower_wl = fit_wl_bounds[0]
        self.fit_upper_wl = fit_wl_bounds[1]

        wl_mask = (self.sn_spec[self.sn_keys[0]].value > self.fit_lower_wl) & (self.sn_spec[self.sn_keys[0]].value < self.fit_upper_wl)
        self.sn_spec_trimmed = self.sn_spec[wl_mask]

        self.sn_templates = None
        self.gal_eigenspec = None
        self.gal_eigenvals = None
        self.gal_model = None
        self.sn_model = None
        self.spec_model = None
        self.spec_model_params = None
        self.sn_spec_no_host = None


    def fit_spectrum(self):
        if self.sn_templates is None:
            self._obtain_sn_templates()
        if self.gal_eigenspec is None:
            self._obtain_gal_eigenspec()

        best_chi = np.inf
        for sn_template in self.sn_templates:
            for i in range(2):  # Run the loop twice, with and without galaxy model.
                if i == 0:
                    lsq_result, design_matrix = self._lsq_fitting_with_gal(sn_template)
                if i == 1:
                    lsq_result, design_matrix = self._lsq_fitting_without_gal(sn_template)

                better_fit, chi2, spec_model = self._evaluate_lsq_fit(lsq_result, design_matrix, best_chi)
                if better_fit:
                    best_chi = chi2
                    self.spec_model = spec_model * self.sn_spec_trimmed[self.sn_keys[1]].unit  #TODO "Unit handling issue"
                    self.sn_model =  np.ravel(design_matrix[:, :3] @ lsq_result.x[:3]) * self.sn_spec_trimmed[self.sn_keys[1]].unit  #TODO "Unit handling issue"
                    self.gal_eigenvals = lsq_result.x[3:]
                    self.gal_model = np.ravel(design_matrix[:, 3:] @ self.gal_eigenvals)  * self.sn_spec_trimmed[self.sn_keys[1]].unit  #TODO "Unit handling issue"
                    self.spec_model_params = design_matrix


    def remove_galaxy_contamination(self):
        '''
        Remove the fitted galaxy spectrum from the observed data and return
        the value.
        If the galaxy spectrum has not yet been determined then run the fitting
        process.
        '''

        if self.gal_model is None:
            self.fit_spectrum()

        self.sn_spec_no_host = self.sn_spec_trimmed.copy()
        self.sn_spec_no_host[self.sn_keys[1]] = self.sn_spec_no_host[self.sn_keys[1]] - self.gal_model
        return self.sn_spec_no_host


    def plot_fit(self, plot_host_free_spec=True, plot_gal_components=True, show=True):
        n_subplots = 2
        axes_ind = 2
        if plot_gal_components:
            n_subplots += 1
        if plot_host_free_spec:
            n_subplots += 1
        
        fig, axes = plt.subplots(n_subplots, 1, sharex=True)
        
        if self.spec_model is not None:
            axes[0].plot(self.sn_spec[self.sn_keys[0]], self.sn_spec[self.sn_keys[1]], label="Observed Spectrum")
            axes[0].plot(self.sn_spec_trimmed[self.sn_keys[0]], self.spec_model, label="Model")
            axes[0].plot(self.sn_spec_trimmed[self.sn_keys[0]], self.sn_model, label="Model (SN)")
            axes[0].plot(self.sn_spec_trimmed[self.sn_keys[0]], self.gal_model, label="Model (galaxy)")
            axes[0].legend()
            # Residual plot
            axes[1].axhline(0, 0, 1, c="k")
            axes[1].plot(self.sn_spec_trimmed[self.sn_keys[0]], self.sn_spec_trimmed[self.sn_keys[1]] - self.spec_model, label="Residual (Observed - Model)")
            axes[1].legend()

            if plot_host_free_spec and self.sn_spec_no_host is not None:
                axes[axes_ind].plot(self.sn_spec_no_host[self.sn_keys[0]], self.sn_spec_no_host[self.sn_keys[1]], label="Host Free Observed Spectrum")
                axes[axes_ind].plot(self.sn_spec_no_host[self.sn_keys[0]], self.gal_model, label="Model (galaxy)")
                axes[axes_ind].legend()
                axes_ind += 1

            if plot_gal_components:
                axes[axes_ind].plot(self.sn_spec_trimmed[self.sn_keys[0]], self.gal_model, label="Model (galaxy)")
                if len(self.gal_eigenvals) > 0:
                    for i, eigenspec in enumerate(self.gal_eigenspec):
                        axes[axes_ind].plot(self.sn_spec_trimmed[self.sn_keys[0]],
                                            np.dot(self.gal_eigenvals[i], eigenspec[self.sn_keys[1]]),
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
                    spec_aligned = align_spec_wave(self.sn_spec_trimmed, spec, keys1=self.sn_keys)  # Note: spec_aligned has same keys as self.sn_keys
                elif z_diff < 0:
                    spec["wave"] *= (1 + abs(z_diff))  # Redshift the spectrum
                    spec_aligned = align_spec_wave(self.sn_spec_trimmed, spec, keys1=self.sn_keys)
                elif z_diff == 0:
                    spec_aligned = align_spec_wave(self.sn_spec_trimmed, spec, keys1=self.sn_keys)
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
            gal_eigenspec_aligned.append(align_spec_wave(self.sn_spec_trimmed, spec, keys1=self.sn_keys))
        self.gal_eigenspec = gal_eigenspec_aligned


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
        wl = self.sn_spec_trimmed[self.sn_keys[0]].value
        d_wl = wl - wl_fixed

        p0 = np.ones_like(wl)
        p1 = d_wl
        p2 = d_wl**2

        return p0, p1, p2


    def _design_matrix_with_gal(self, sn_template):
        poly = self._define_sn_template_polynomial()
        gal_eigenspec_fluxes = np.array([spec[self.sn_keys[1]] for spec in self.gal_eigenspec])
        design_matrix = np.vstack([sn_template[self.sn_keys[1]] * poly[0], sn_template[self.sn_keys[1]] * poly[1],
                                   sn_template[self.sn_keys[1]] * poly[2], gal_eigenspec_fluxes]).T
        return design_matrix
    

    def _design_matrix_without_gal(self, sn_template):
        poly = self._define_sn_template_polynomial()
        design_matrix = np.vstack([sn_template[self.sn_keys[1]] * poly[0], sn_template[self.sn_keys[1]] * poly[1],
                                   sn_template[self.sn_keys[1]] * poly[2]]).T
        return design_matrix


    def _lsq_fitting_with_gal(self, sn_template):

        design_matrix = self._design_matrix_with_gal(sn_template)
        target_vec = self.sn_spec_trimmed[self.sn_keys[1]].value

        # Require that the galaxy eigenvalues are positive.
        lower_bounds = np.concatenate([[-np.inf] * 3, [0.0] * len(self.gal_eigenspec)])
        upper_bounds = np.full(design_matrix.shape[1], np.inf)

        lsq_result = lsq_linear(design_matrix, target_vec, bounds=(lower_bounds, upper_bounds))

        return lsq_result, design_matrix


    def _lsq_fitting_without_gal(self, sn_template):

        design_matrix = self._design_matrix_without_gal(sn_template)
        target_vec = self.sn_spec_trimmed[self.sn_keys[1]].value

        lower_bounds = np.full(design_matrix.shape[1], -np.inf)
        upper_bounds = np.full(design_matrix.shape[1], np.inf)

        lsq_result = lsq_linear(design_matrix, target_vec, bounds=(lower_bounds, upper_bounds))

        return lsq_result, design_matrix


    def _evaluate_lsq_fit(self, lsq_result, design_matrix, best_chi):
        '''
        return: a, b, c
                a = Is the fit better than the given current best
                    (i.e better than best_chi)
                b = new best chi2
                c = new best model
        '''

        if not lsq_result.success:
            return False, None, None

        model_fit = np.ravel(design_matrix @ lsq_result.x)
        chi2 = np.sum(((self.sn_spec_trimmed[self.sn_keys[1]].value - model_fit) / self.sn_spec_trimmed[self.sn_keys[2]].value) ** 2)

        if chi2 < best_chi:
            return True, chi2, model_fit
        else:
            return False, None, None
