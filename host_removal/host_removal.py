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

        if self.sn_spec[self.sn_keys[0]].unit == u.nm:
            self.min_wl = nm_to_A(self.min_wl)

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
        for i, sn_template in enumerate(self.sn_templates):
            print("\nSN template:", i)
            #TODO If the code is slow at iterating over many SN templates
            # then I can multiproccess this step.
            lsq_result, design_matrix = self._lsq_fitting(sn_template)

            eval, chi2, spec_model = self._evaluate_lsq_fit(lsq_result, design_matrix, best_chi)
            if eval:
                best_chi = chi2
                self.spec_model = spec_model
                self.sn_model =  np.ravel(design_matrix[:, :3] @ lsq_result.x[:3])
                self.gal_eigenvals = lsq_result.x[3:]
                self.gal_model = np.ravel(design_matrix[:, 3:] @ self.gal_eigenvals)
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
        self.sn_spec_no_host[self.sn_keys[1]] -= self.gal_model

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
            sn_templates_aligned.append(align_spec_wave(self.sn_spec_trimmed, spec, keys1=self.sn_keys))
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


    def _design_matrix(self, sn_template):
        poly = self._define_sn_template_polynomial()
        gal_eigenspec_fluxes = np.array([spec[self.sn_keys[1]] for spec in self.gal_eigenspec])
        design_matrix = np.vstack([sn_template[self.sn_keys[1]] * poly[0], sn_template[self.sn_keys[1]] * poly[1],
                                   sn_template[self.sn_keys[1]] * poly[2], gal_eigenspec_fluxes]).T
        return design_matrix


    def _lsq_fitting(self, sn_template):

        design_matrix = self._design_matrix(sn_template)
        target_vec = self.sn_spec_trimmed[self.sn_keys[1]].value

        # Require that the galaxy eigenvalues are positive.
        lower_bounds = np.concatenate([[-np.inf] * 3, [0.0] * len(self.gal_eigenspec)])
        upper_bounds = np.full(design_matrix.shape[1], np.inf)

        lsq_result = lsq_linear(design_matrix, target_vec, bounds=(lower_bounds, upper_bounds))

        return lsq_result, design_matrix


    def _evaluate_lsq_fit(self, lsq_result, design_matrix, best_chi):

        if not lsq_result.success:
            return False, None, None

        model_fit = np.ravel(design_matrix @ lsq_result.x)
        chi2 = np.sum(((self.sn_spec_trimmed[self.sn_keys[1]].value - model_fit) / self.sn_spec_trimmed[self.sn_keys[2]].value) ** 2)

        if chi2 < best_chi:
            return True, chi2, model_fit
