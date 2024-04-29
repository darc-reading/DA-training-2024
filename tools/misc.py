import numpy as np

def createTime(t0, tf, deltat, discard):
    t = np.arange(t0 + discard * deltat, tf + deltat / 2, deltat)
    return t