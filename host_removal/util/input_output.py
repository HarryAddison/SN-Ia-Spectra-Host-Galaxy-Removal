import numpy as np
from astropy.table import QTable
from importlib.resources import files
from pathlib import Path


def load_gal_eigenspec(num_eigenspec=10):

    eigenspec = []
    for i in range(num_eigenspec):
        path = files('host_removal.data').joinpath(f"galaxy-eigenspectra/galaxyKL_eigSpec_{i+1}.dat")
        eigenspec.append(QTable.read(path, format="ascii", names=["wave", "flux"]))
    return eigenspec


def find_matching_files(dir, pattern):
    return [f for f in dir.iterdir() if f.is_file() and f.name.startswith(pattern)]


def load_sn_templates(obs_phase, phase_diff=5):
    '''
    phase diff == integer
    '''
    template_dir  = path = files('host_removal.data').joinpath(f"sn-templates/templates")
    obs_phase = np.round(obs_phase)
    phase_diff = np.round(phase_diff)

    sn_templates = []
    for i in range((phase_diff * 2) + 1):
        phase = obs_phase - phase_diff + i  # gives phases between obs_phase +- phase_diff
        file_paths = list(find_matching_files(template_dir, pattern=f"sn_template_phase_{phase:.1f}"))
        for path in file_paths:
            template = QTable.read(path)
            template["flux"] /= max(template["flux"])
            sn_templates.append(template)

    return sn_templates
