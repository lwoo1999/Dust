import numpy as np
import jhk
import wise
import ugriz

def prepare_data(data):
    z = data["redshift"]
    wavelength = []
    rsr = []
    lum = []
    lum_unc = []

    lum_ugriz, lum_ugriz_unc = ugriz.mag_to_lum(data["ugriz_dered"], data["ugriz_err"], z)
    for i in range(5):
        if data["ugriz_err"][i] != 0:
            wavelength.append(ugriz.wavelength[i]/(1+z))
            rsr.append(ugriz.rsr[i].redshift(z))
            lum.append(lum_ugriz[i])
            lum_unc.append(lum_ugriz_unc[i])

    lum_jhk, lum_jhk_unc = jhk.mag_to_lum(data["jhk"], data["jhk_err"], z)
    for i in range(3):
        if data["jhk_err"][i] != 0:
            wavelength.append(jhk.wavelength[i]/(1+z))
            rsr.append(jhk.rsr[i].redshift(z))
            lum.append(lum_jhk[i])
            lum_unc.append(lum_jhk_unc[i])

    lum_wise, lum_wise_unc = wise.mag_to_lum(data["wise1234"], data["wise1234_err"], z)
    for i in range(4):
        if data["wise1234_err"][i] != 0:
            wavelength.append(wise.wavelength[i]/(1+z))
            rsr.append(wise.rsr[i].redshift(z))
            lum.append(lum_wise[i])
            lum_unc.append(lum_wise_unc[i])

    return rsr, wavelength, lum, lum_unc