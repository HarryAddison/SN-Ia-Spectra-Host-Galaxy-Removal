# Host galaxy removal

Code to remove host-galaxy contamination from SN spectra.
SN templates and galaxy eigenspectra are used to create a model spectrum.
The difference between the model and the observed spectrum is minimised to
find the model host galaxy spectrum, which is then subtracted from the
observed spectrum.  


# Installation:

1. Clone the repository
2. Go to the downloaded repository directory
3. When in the repository directory do $ pip3 install .
4. Import the package in python as "host_removal" (See examples)