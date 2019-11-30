import numpy as np
from astropy.cosmology import Planck15

from rsr import RSR

wavelength = np.array([
    0.3551, 
    0.4686, 
    0.6165, 
    0.7481, 
    0.8931
])  # um

softening = np.array([
    1.4e-10,
    0.9e-10,
    1.2e-10,
    1.8e-10,
    7.4e-10
])

zeropoint = 3631e-23 * 299792458000000 / wavelength**2  # erg/s/um/cm^2

AB_correction = np.array([
    0.04,
    0.,
    0.,
    0.,
    -0.02
])

def mag_to_lum(mag, mag_unc, z):
    dist = Planck15.luminosity_distance(z).value * 3.08568e+24  # cm
    lum = -(mag - AB_correction)/2.5 + np.log10(zeropoint) + np.log10(wavelength) + np.log10(4*np.pi*dist**2)
    lum_unc = mag_unc / 2.5
    return lum, lum_unc


rsr = []

for file in ["U","G","R","I","Z"]:
    data = np.loadtxt(f"../data/rsr/{file}")
    rsr_ = RSR(data[:,0]/1e4, data[:,1])
    rsr_.normalise()
    rsr.append(rsr_)