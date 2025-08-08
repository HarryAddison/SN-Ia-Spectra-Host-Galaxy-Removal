import os
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
    template_dir  = path = files('host_removal.data').joinpath(f"sn-templates")

    sn_templates = []
    for i in range((phase_diff * 2) + 1):
        phase = obs_phase - phase_diff + i  # gives phases between obs_phase +- phase_diff
        file_paths = find_matching_files(template_dir, pattern=f"sn_template_phase_{phase}")
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
