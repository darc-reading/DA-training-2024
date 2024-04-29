import numpy as np
from scipy.optimize import fsolve
from tools.L96_model import Lorenz96_and_TLM, lorenz96


def var4d(x0, t, period_obs, anawin, y_traj, H, B0sq, invR, F):
    """4D-Var cycling data assimilation routine for Lorenz 1996 using

    Arguments
    ---------
    x0 : ndarray
        the real initial conditions (truth). shape: nx, nt
    t : ndarray
        time array of model time steps (all cycles). shape: nt
    period_obs : int
        the number of model timesteps between observations.
    anawin : int
        the number of observation periods between 4D-Var cycles
    y_traj : ndarray
        the observations (all cycles). shape: ny, nt
    H : ndarray
        observation matrix. shape: ny, nx
    B0sq : ndarray
        the square root of background error covariance matrix. shape: nx, nx
    invR : ndarray
        the inverse of observation error covariance matrix. shape: ny, ny
    F : float
        the forcing used in the L96 model when cycling

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

    # Number of 4D-Var cycles that will be done here
    anal_nt = period_obs*anawin
    anal_dt = anal_nt * deltat
    ncycles = int(float(nt) / float(anal_nt))

    # Make blank arrays for background and analysis trajectories
    xb_traj = np.empty([nx, nt])
    xb_traj.fill(np.nan)
    xa_traj = np.empty([nx, nt])
    xa_traj.fill(np.nan)

    # This is the background for the first instant
    xb_traj[:, 0] = x0[:]
    xa_traj[:, 0] = x0[:]

    # pre-compute certain matrix for efficiency
    HTinvR = H.T@invR
    nouterloops = 1

    # The following cycle contains evolution and assimilation
    # Loop around 4D-Var cycles
    for cycle in range(ncycles):
        # This cycle corresponds to the following start and end times
        start = cycle * anal_nt
        end = start + anal_nt

        # Extract observations for this cycle
        yaux_traj = y_traj[:, cycle * anawin + 1 : (cycle + 1) * anawin + 1]

        # Find the 4D-Var analysis
        xbaux_traj, xaaux_traj = one4dvarPC(xa_traj[:, start], deltat, anawin,
                            yaux_traj, H, B0sq, invR, period_obs, HTinvR, F, nouterloops)

        # Store this for reference
        xa_traj[:, start : end + 1] = xaaux_traj[:, 0 : anal_nt + 1]
        xb_traj[:, start + 1 : end + 1] = xbaux_traj[:, 1 : anal_nt + 1]

    return xb_traj, xa_traj


def one4dvarPC(xb0, deltat, anawin, yaux, H, Bsq, invR, period_obs, HTinvR, F, nouterloops):
    """Solving analysis for a single 4DVar window
    """
    xg0 = xb0
    nx = len(xg0[:])
    anal_nt = period_obs*anawin
    anal_dt = anal_nt * deltat

    # initial guess of the control variable
    vold = np.zeros(nx)
    # innovation vector
    d = np.zeros_like(yaux, order='F')

    for jouter in range(nouterloops):
        # get tangent linear model
        xb, tm = Lorenz96_and_TLM(np.eye(nx), anal_nt, xg0, F, deltat)
        for j in range(anawin):
            d[:, j] = yaux[:, j] - H@xb[:, (j+1)*period_obs]

        # The gradient
        def gradJ(v):
            # The background term
            gJ = v.copy()
            # The observation error term, evaluated at different times
            for j in range(anawin):
                aux = tm[...,(j+1)*period_obs]@(Bsq@v)
                gJ += - Bsq.T@(tm[..., (j+1)*period_obs].T@(HTinvR@(d[:, j] - H@aux)))
            return gJ.ravel()

        va = fsolve(gradJ, vold, maxfev=10)
        xa0 = xg0 + Bsq@va
        xa = lorenz96(xa0, anal_dt, deltat, 0, F)
        xg0 = xa0
            
    return xb, xa
