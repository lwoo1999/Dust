import numpy as np
from astropy.cosmology import Planck15

from rsr import RSR

wavelength = np.array([
    3.4, 
    4.6, 
    12, 
    22
])  # um

zeropoint = np.array([
    8.1787e-15,
    2.4150e-15,
    6.5151e-17,
    5.0901e-18
])  # W/cm^2/um

def mag_to_lum(mag, mag_unc, z):
    dist = Planck15.luminosity_distance(z).value * 3.08568e+24  # cm
    lum = -mag/2.5 + np.log10(zeropoint) + np.log10(wavelength) + np.log10(4*np.pi*dist**2) + 7 # J -> erg
    lum_unc = mag_unc / 2.5
    return lum, lum_unc

rsr = []

for file in ["W1","W2","W3", "W4"]:
    data = np.loadtxt(f"../data/rsr/{file}")
    rsr_ = RSR(data[:,0], data[:,1])
    rsr_.normalise()
    rsr.append(rsr_)