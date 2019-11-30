from numba import jit, float64, int64
import numpy as np

@jit(int64(float64[:], float64, int64, int64), nopython=True)
def find(xs, x, l, u):
    if u - l == 1:
        return l
    m = (l + u) // 2
    if x < xs[m]:
        return find(xs, x, l, m)
    else:
        return find(xs, x, m, u)


@jit(float64(float64[:], float64[:], float64), nopython=True)
def interp(xs, ys, x):
    i = find(xs, x, 0, len(xs) - 1)
    return ys[i] + (x - xs[i]) * (ys[i+1] - ys[i]) / (xs[i+1] - xs[i])

    # for i in range(len(xs) - 1):
    #     if x >= xs[i] and x <= xs[i+1]:
    #         return ys[i] + (x - xs[i]) * (ys[i+1] - ys[i]) / (xs[i+1] - xs[i])
    
    # return 0

@jit(float64[:](float64[:], float64[:], float64[:]), nopython=True, parallel=True)
def interp_(xs, ys, x):
    res = np.zeros(len(x))
    for i in range(len(res)):
        res[i] = interp(xs, ys, x[i])
    
    return res