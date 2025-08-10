from host_removal import HostGalaxyRemoval
from astropy.table import QTable

# Load in the observed SN Ia spectrum
spec = QTable.read("./data/kinney_sc_template.fits")

# Define the needed properties of the SN.
# (There is no rest phase of a SN as the template is hust a galaxy
# The code needs a rest phase so we will define it as 0.)
rest_phase = 0

# Normalise the SN spectrum
max_flux = max(spec["FLUX"])
spec["FLUX"] /= max_flux

# The kinney templates dont have errors so we will just set it to 1.
# (This only affects the calculated chi^2 value and not the fitting.)
spec["FLUX_ERR"] = 1

# Initialise the HostGalaxyRemoval object for this SN spectrum
hgr = HostGalaxyRemoval(spec, rest_phase, sn_keys=["WAVELENGTH", "FLUX", "FLUX_ERR"])

# Run the fitting procedure to get the galaxy model
hgr.fit_spectrum()

# Plot the results of the fitting
hgr.plot_fit(plot_gal_components=True, show=True)
