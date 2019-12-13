import numpy as np
from model import dust_models, prepare_data, blackbody, disk_powlaw, ec, get_band
from analysis import *

def simulate(data, params_, dust_amp, logcf_mean, logcf_std):
    rsr, wavelength, lum, lum_unc = prepare_data(data)
    *params, residual, mod = params_
    dust_model = dust_models[int(mod)]
    loglbol, loglnir, logcf = analysis(params_)

    new_logcf = dust_amp * logcf_std + logcf_mean
    factor = new_logcf - logcf

    band = get_band(dust_model, params)

    disk_amp, dust_temp, dust_lbol, av, cold_dust = params
    band_ = get_band(dust_model, [
        disk_amp,
        dust_temp,
        dust_lbol + factor,
        av,
        cold_dust + factor
    ])

    lum_ = band_(rsr, wavelength) + lum - band(rsr, wavelength)

    return fit_for_storage((rsr, wavelength, lum_, lum_unc), prepare_data=lambda x: x)