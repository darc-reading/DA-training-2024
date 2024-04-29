import numpy as np
from tools.L96_model import lorenz96
from scipy.optimize import fsolve

def var3d(x0, t, tobs, y, H, B0sq, invR, F, opcini=1):
    """Data assimilation routine for Lorenz 1996 using 3D-Var

    The 3D-Var implementation and its cycling is similar to
    an iterative Kalman filter.

    This means that the 3DVar occurs at the end of each cycle

    Arguments
    ---------
    x0 : ndarray
        the real initial conditions. shape: nx
    t : ndarray
        time array of model time steps. shape: nt
    tobs : ndarray
        time array of the observations (must have a timestep a
        multiple of the model's),
        these are the times that 3D-Var is done. shape: nobt
    y : ndarray
        the observations. shape: ny, nobt
    H : ndarray
        observation matrix. shape: ny, nx
    B0sq : ndarray
        the square root of background error covariance matrix. shape: nx, nx
    invR : ndarray
        the inverse of observation error covariance matrix. shape: ny, ny
    F : float
        the forcing used in the L96 model when cycling
    opcini : int
        set to method for initial guess condition

    Returns
    -------
    xb_traj : ndarray
        the background trajectory. shape: nx, nt
    xa_traj : ndarray
        the analysis trajectory. shape: nx, nt
    """
    # General settings
    nt = len(t)  # Number of timesteps in total
    nx = len(x0)  # Number of state variables

    # Model's timestep
    deltat = t[1] - t[0]

    # Analysis timestep
    anal_dt = tobs[1] - tobs[0]

    # The ratio (number of timesteps between obs)
    o2t = np.rint(anal_dt / deltat).astype(int)

    # Check that ob timestep is multiple of model timestep
    assert np.isclose(o2t, anal_dt / deltat), \
        "observation timestep must be a multiple of model timestep"

    # Make blank arrays for background and analysis trajectories
    xb_traj = np.empty([nx, nt], order='F')
    xb_traj.fill(np.nan)
    xa_traj = np.empty([nx, nt], order='F')
    xa_traj.fill(np.nan)

    # Initial Condition for First Guess of First Window
    if opcini == 0:
        xold = x0 + B0sq@np.random.randn(nx)
    if opcini == 1:
        xold = x0

    # This is the initial condition for the experiment period
    xb_traj[:, 0] = xold[:]
    xa_traj[:, 0] = xb_traj[:, 0]

    for j in range(len(tobs) - 1):
        # Do a background forecast between now and the start of the next cycle
        x_traj = lorenz96(xold, anal_dt, deltat, 0, F)
        # store the trajectory
        xb_traj[:, j * o2t + 1 : (j + 1) * o2t + 1] = x_traj[:, 1:]
        xa_traj[:, j * o2t + 1 : (j + 1) * o2t + 1] = x_traj[:, 1:]
        # Find the 3D-Var analysis for the next cycle
        xa_aux = one3dvarPC(xb_traj[:, (j + 1) * o2t], y[:, j + 1], H, B0sq, invR)
        # Store in the analysis array
        xa_traj[:, (j + 1) * o2t] = xa_aux[:]
        # set the initial condition for the next cycle
        xold = xa_traj[:, (j + 1) * o2t]
    return xb_traj, xa_traj


def one3dvarPC(xold, yaux, H, sqrtB, invR):
    """Solving analysis for a single 3DVar window
    """
    nx = np.size(xold)
    d = yaux - np.dot(H,xold)

    # The Cost function
    def CostFun(v):
        v = np.reshape(v,(nx,1))
        # The background term
        Jb = 0.5*(v.T@v)
        # The observation error term
        dd = d - H@sqrtB@v
        Jok = 0.5*(dd.T@invR@dd)
        return Jb + Jok

    # The gradient
    def gradJ(v):
        # The background term
        gJb = v
        # The observation error term
        gJok = -sqrtB.T@(H.T@(invR@(d - H@sqrtB@v)))
        return gJb + gJok

    vold = np.zeros((nx,))
    va = fsolve(gradJ,vold)
    if np.sum(np.isfinite(va))==nx:
        xa = xold + np.dot(sqrtB,va)
    else:
        xa = xold

    return xa
