import numpy as np
import numba


@numba.jit(nopython=True)
def accumulate_events(out, xs, ys, ps):
    for i in range(xs.shape[0]):
        x, y, p = xs[i], ys[i], ps[i]
        out[y, x] += p


@numba.jit(nopython=True)
def accumulate_events_range(out, xs, ys, ps, ts, low, high):
    for i in range(xs.shape[0]):
        if low <= ts <= high:
            x, y, p = xs[i], ys[i], ps[i]
            out[y, x] += p
