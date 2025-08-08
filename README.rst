# Host galaxy removal

Code to remove host-galaxy contamination from SN spectra.
SN templates and galaxy eigenspectra are used to create a model spectrum.
The difference between the model and the observed spectrum is minimised to
find the model host galaxy spectrum, which is then subtracted from the
observed spectrum.  
