import functools
import numpy as np
from tools.timestepping import rk4


def lorenz63(x0, tf, deltat, discard, param=None):
    """Evolution of the Lorenz 1963 3-variable model.
    Parameters
    ----------
    x0 : ndarray
        The initial state. The length of the 1D array
        is the number of L96 state variable
    tf : float
        The final model time
    deltat : float
        The timestep of the integration scheme
    discard : int
        The number of timesteps at the beginning to discard
    param : tuple or list or array
        The parameters used for L63

    Returns
    -------
    xf : ndarray
        The forecast time sequence. shape: (3, nt)
    """
    # Number of time steps (+1 as t=0 included also)
    nt = int(tf / deltat) + 1
    # Now, define the vectors for space vars. 
    # They're organized in an
    # array of 3 columns [x,y,z].
    xf = np.empty((3, nt), order='F')
    xf.fill(np.nan)

    # giving model parameters
    model = functools.partial(f, param=param)
    # Run a number of timesteps to spin-up the model
    # (these are discarded)
    for time in range(discard):
        x0[:] = x0[:] + rk4(x0[:], deltat, model)

    # Initial conditions for part of integration which is kept
    xf[:, 0] = x0[:]

    # The cycle containing time integration
    for time in range(nt-1): # for each time step
        xf[:, time + 1] = xf[:, time] + rk4(xf[:, time], deltat, model)
    return xf

def f(x, param=None):
    """time tendency of Lorenz-63 model d/dt

    Parameters
    ----------
    x : ndarray
        1D array of L96 state at one time step
    param : tuple or list or array
        The parameters used for L63
    Returns
    -------
    k : ndarray
        df/dt as described by the Lorenz-96 model
    """
    # The parameters
    if np.all(param)==None:
        sigma = 10.0
        b = 8/3.0
        r = 28.0
    else:
        sigma, b, r = param
     
    # Initialize
    k = np.empty_like(x)
    k.fill(np.nan)
    # The Lorenz equations
    k[0] = sigma*(x[1]-x[0])
    k[1] = x[0]*(r-x[2])-x[1]
    k[2] = x[0]*x[1]-b*x[2]
    return k

