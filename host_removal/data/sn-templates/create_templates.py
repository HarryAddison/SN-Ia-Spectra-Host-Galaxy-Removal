from qmost_mock_spectra.modules.input_output import load_config
from qmost_mock_spectra.modules.parameter_grid import create_sn_ia_param_grid
from qmost_mock_spectra.modules.extinction import get_mwebv
from qmost_mock_spectra.modules.template_spectrum import template_spectrum
from qmost_mock_spectra.modules.l1_spectrum import l1_spectrum
import numpy as np
import astropy.units as u


if __name__ == "__main__":

    config = load_config()

    param_grid = create_sn_ia_param_grid(config)

    param_grid["id"] = np.arange(1, 1+ len(param_grid), 1)
    param_grid["rest_phase"] = param_grid["observer_phase"] / (1 + param_grid["z"]) * u.day

    param_grid["mwebv"] = 0.0 * u.mag

    for i, params in enumerate(param_grid):
        # Set the save paths for figures and data
        # The name is based on certain input parameters that vary:
        # - Phase: rest frame phase (in this case its equal to observer_phase as the SN is at z=0 and t0=0)
        # - id: ID of the SN, which can be crossmatched to the parameter output file.
        template_path = f"{config['template_spec_dir']}/sn_template_phase_{params['observer_phase']}_id_{params['id']}.fits"
        template_spec = template_spectrum(params, template_path)

    # Save the parameters grid of the spectra to produced.
    param_grid.write(config["param_grid_path"], overwrite=True)

    print("Full parameter grid saved.")
