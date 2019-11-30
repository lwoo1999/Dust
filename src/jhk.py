import numpy as np
from astropy.cosmology import Planck15

from rsr import RSR

wavelength = np.array([
    1.235,
    1.662,
    2.159
])  # um

zeropoint = np.array([
    3.129E-13,
    1.133E-13,
    4.283E-14
])  # W/cm^2/um

def mag_to_lum(mag, mag_unc, z):
    dist = Planck15.luminosity_distance(z).value * 3.08568e+24  # cm
    lum = -mag/2.5 + np.log10(zeropoint) + np.log10(wavelength) + np.log10(4*np.pi*dist**2) + 7 # J -> erg
    lum_unc = mag_unc / 2.5
    return lum, lum_unc

rsr = []

for file in ["J","H","K"]:
    data = np.loadtxt(f"../data/rsr/{file}")
    rsr_ = RSR(data[:,0], data[:,1])
    rsr_.normalise()
    rsr.append(rsr_)