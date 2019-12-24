import numpy as np
from astropy.io import fits

file = fits.open("../data/dr7_bh_Nov19_2013.fits")
data = file[1].data
file.close()


def for_hb():
    ret = data
    ret = ret[ret["z_hw"] < 0.875]
    ret = ret[ret["ew_narrow_hb"] != 0]
    ret = ret[ret["ew_broad_hb"] != 0]
    ret = ret[
        np.apply_along_axis(all, 1, ret["WISE1234"] / ret["WISE1234_ERR"] > 3)
        ]
    return ret


def for_mgii():
    ret = data
    ret = ret[ret["z_hw"] < 1.858]
    ret = ret[ret["z_hw"] > 0.786]
    ret = ret[ret["ew_mgii"] != 0]
    ret = ret[
        np.apply_along_axis(all, 1, ret["WISE1234"] / ret["WISE1234_ERR"] > 3)
        ]
    return ret


def for_oiii():
    ret = data
    ret = ret[ret["z_hw"] < 0.8]
    ret = ret[ret["ew_oiii_5007"] != 0]
    ret = ret[
        np.apply_along_axis(all, 1, ret["WISE1234"] / ret["WISE1234_ERR"] > 3)
        ]
    return ret


def for_civ():
    ret = data
    ret = ret[ret["z_hw"] > 2.228]
    ret = ret[ret["z_hw"] < 4.165]
    ret = ret[ret["ew_civ"] != 0]
    ret = ret[
        np.apply_along_axis(all, 1, ret["WISE1234"] / ret["WISE1234_ERR"] > 3)
        ]
    return ret