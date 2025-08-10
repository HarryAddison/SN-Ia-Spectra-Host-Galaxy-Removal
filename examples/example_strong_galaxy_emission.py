from host_removal import HostGalaxyRemoval
from astropy.table import QTable

# Load in the observed SN Ia spectrum
sn_spec = QTable.read("./data/example_observed_spectrum.fits")

# Define the needed properties of the SN.
rest_phase = 0
z = 0.01

# Convert the SN spectrum to rest frame (i.e de-redshift it)
sn_spec["wave"] /= (1 + z)

# Normalise the SN spectrum
max_flux = max(sn_spec["flux"])
sn_spec["flux"] /= max_flux
sn_spec["flux_err"] /= max_flux

# Initialise the HostGalaxyRemoval object for this SN spectrum
hgr = HostGalaxyRemoval(sn_spec, rest_phase, sn_keys=["wave", "flux", "flux_err"])

# Run the fitting procedure to get the galaxy model
hgr.fit_spectrum()

# Plot the results of the fitting
hgr.plot_fit(plot_gal_components=True, show=True)
