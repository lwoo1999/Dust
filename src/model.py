import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import integrate
import scipy.optimize as opt

from numba import jit, float64

from prepare_data import prepare_data

# smc extinction
smc = np.loadtxt("../data/smc")

# um -> A(λ)/A(V)
ec = interp1d(1/smc[:, 0], smc[:, 1], bounds_error=False, fill_value=0)

# disk
disk = np.loadtxt("../data/disk")
disk_spec = interp1d(disk[0], disk[1])

# log(lambda) -> log(lambda*L_lambda)
def disk_powlaw_(x):
    if x > disk[0,0]:
        return disk[1,0] - x + disk[0,0]
    elif x < disk[0,-1]:
        return disk[1,-1]
    else:
        return disk_spec(x)

# lambda -> lambda*L_lambda
@np.vectorize
def disk_powlaw(x, amp):
    return amp * 10 ** disk_powlaw_(np.log10(x))


# lambda -> lambda*L_lambda
@jit(float64[:](float64[:], float64, float64), nopython=True, parallel=True)
def blackbody(x, temp, lbol):
    # constants
    h = 6.626e-34  # m^2*kg/s
    k = 1.381e-23  # m^2*kg/s^2/K
    c = 299792458  # m/s
    sigma = 5.6704e-8  # J/s/m^2/K^4

    lam = x * 1e-6  # convert to m

    B_lambda = (2*h*c**2/lam**5)/(np.exp(h*c/lam/k/temp)-1)

    bb = np.pi*B_lambda*lbol/sigma/temp**4  # J/s/m
    bb *= 1e-6 * x  # erg/s
    return bb

# dust
m1 = np.genfromtxt("../data/dust/model1.csv")
m2 = np.genfromtxt("../data/dust/model2.csv")
m3 = np.genfromtxt("../data/dust/model3.csv")
# m4 = np.genfromtxt("./model4.csv")
wav = np.genfromtxt("../data/dust/wavelength.csv")

dust_models = [
    interp1d(wav, m1),
    interp1d(wav, m2),
    interp1d(wav, m3),
    # interp1d(wav, m4),
]

# total model
def get_sed(dust_model, params):
    disk_amp, dust_temp, dust_lbol, av, cold_dust = params
    disk_amp = 10**disk_amp
    dust_lbol = 10**dust_lbol
    cold_dust = 10**cold_dust

    def sed(x):
        bb = blackbody(x, dust_temp, dust_lbol)
        pl = disk_powlaw(x, disk_amp)
        cd = cold_dust * dust_model(x)
        ext = np.exp(ec(x) * av)

        return (bb + pl + cd) * ext
    
    return sed

def get_band(dust_model, params):
    sed = get_sed(dust_model, params)

    @np.vectorize
    def band(rsr, wavelength):
        return wavelength*integrate.trapz(rsr.ys*sed(rsr.xs)/rsr.xs, x=rsr.xs)

    return band

def get_residual(dust_model, rsr, wavelength, lum, lum_unc):
    
    def residual(params):
        band = get_band(dust_model, params)
        res = sum(((np.log10(band(rsr, wavelength)) - lum) / lum_unc)**2)
        return res
    
    return residual


initial_params = [
    np.log10(0.5e46),  # log(disk_amp)
    1500.,             # dust_temp
    np.log10(5e45),    # log(dust_lbol)
    -0.2,              # av
    np.log10(1e46)     # log(cold_dust)
]

bounds = opt.Bounds(
    # disk_amp, dust_temp, dust_lbol, av,      cold_dust
    [-np.inf,   500.,     -np.inf,   -np.inf, -np.inf],
    [ np.inf,   2000.,     np.inf,    0.,      np.inf]
)

def fit(data, method=None, options=None):
    rsr, wavelength, lum, lum_unc = prepare_data(data)
    res = np.inf
    ret = None
    mod = None

    for i, dust_model in enumerate(dust_models):
        residual = get_residual(dust_model, rsr, wavelength, lum, lum_unc)
        opt_res = opt.minimize(residual, initial_params, bounds=bounds, options=options)
        if opt_res.fun < res:
            res = opt_res.fun
            ret = opt_res.x
            mod = i

    return ret, res, mod


def fit_for_storage(data, method=None, options=None):
    ret, res, mod = fit(data, method=method, options=options)
    return np.append(ret, [res, mod])


def show(data, params_):
    _, wavelength, lum, lum_unc = prepare_data(data)
    *params, residual, mod = params_
    dust_model = dust_models[mod]

    sed = get_sed(dust_model, params)

    x = np.logspace(np.log10(np.min(wavelength))-0.2, np.log10(np.max(wavelength))+0.2, 100)
    plt.figure(figsize=(8, 4))

    plt.plot(x, np.log10(sed(x)))
    plt.plot(x, np.log10(dust_model(x)) + params[4])
    plt.plot(x, np.log10(blackbody(x, params[1], 10**params[2])))
    plt.plot(x, np.log10(disk_powlaw(x, 10**params[0])))

    plt.errorbar(wavelength, lum, yerr=lum_unc, fmt="kx")

    plt.xscale("log")
    plt.ylim((np.log10(sed(x).min()) - 0.2, np.log10(sed(x).max()) + 0.2))

    plt.title(f"residual={residual}")

    plt.show(block=True)

def fit_and_show(data, method=None, options=None):
    params, residual, dust_model = fit(data, method=method, options=options)

    show(data, [*params, residual, dust_model])
