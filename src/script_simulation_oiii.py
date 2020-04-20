import numpy as np

from simulation import simulate, gen_by_covm
from data import for_oiii
from analysis import analysis

oiii = for_oiii()
oiii = oiii[oiii["loglbol"] > 46]

oiii_res = np.loadtxt("../res/oiii.res")
oiii_ans = np.array([analysis(i) for i in oiii_res])

logcfs = oiii_ans[:,2]
logcfs_mean = np.mean(logcfs)
logcfs_std = np.std(logcfs)

loglbols = oiii_ans[:,0]
loglbols_mean = np.mean(loglbols)
loglbols_std = np.std(loglbols)

logbhs = oiii["logbh_hb_vp06"]
logbhs_mean = np.mean(logbhs)
logbhs_std = np.std(logbhs)

logews = np.log10(oiii["ew_oiii_5007"])
logews_mean = np.mean(logews)
logews_std = np.std(logews)

corr = np.corrcoef([
    logews, logcfs, loglbols, logbhs
])

def single_simulate(i):
    data = oiii[i]
    params_ = oiii_res[i]
    loglbol_norm = (loglbols[i] - loglbols_mean) / loglbols_std
    logbh_norm = (logbhs[i] - logbhs_mean) / logbhs_std

    logew_norm, logcf_norm = gen_by_covm(corr, [loglbol_norm, logbh_norm])
    logew = logew_norm * logews_std + logews_mean
    logcf = logcf_norm * logcfs_std + logcfs_mean

    return [*simulate(data, params_, logcf), logew]

if __name__ == "__main__":
    from multiprocessing import Pool

    p = Pool(40)
    res = p.map(single_simulate, range(len(oiii)))

    np.savetxt("../res/oiii.sim", np.array(res))
