import numpy as np
from tools.L96_model import lorenz96, Lorenz96_TL_propogation
from numpy import linalg as LA


def compute_lin_error(dx, x0, F, tf, deltat):
    """compute error between TLM and nonlinear model

    Parameters
    ----------
    dx : ndarray
        1D array of model perturbations
    x0 : ndarry
        initial condition of the model
    F : float
        model parameter
    tf: : float
        total model time
    deltat : float
        time step

    Returns
    -------
    NLdiff : ndarray
        1D array of ||NL(x+Dx)-NL(x)||
    TLdiff : ndarry
        1D array of ||TL(Dx)||
    lin_error : int
        Difference between NLdiff and TLdiff
    """
    # Number of time steps (+1 as t=0 included also)
    nt = int(tf / deltat) + 1
    dx_norm = LA.norm(dx, axis=0)

    NLdiff = lorenz96(x0 + dx, tf, deltat, 0, F) - lorenz96(x0, tf, deltat, 0, F)
    TLdiff = Lorenz96_TL_propogation(dx, nt - 1, x0, F, deltat)

    NLdiff_norm = LA.norm(NLdiff, axis=0)
    TLdiff_norm = LA.norm(TLdiff, axis=0)
    lin_error = LA.norm(NLdiff - TLdiff, axis=0)

    return lin_error, NLdiff_norm, TLdiff_norm


def rmse_spread(xt, xmean, Xens, anawin):
    """Compute RMSE and spread.

    This function computes the RMSE of the background (or analysis) 
    mean with respect to the true run, as well as the spread of
    the background (or analysis) ensemble.

    Parameters
    ----------
    xt : ndarray
        the true run of the model [nx, nt]
    xmean : ndarray
        the background or analysis mean [nx, nt]
    Xens : ndarray
        the background or analysis ensemble [nx, n, nt] or None if
        no ensemble
    anawin : int
        the analysis window length.  When assimilation
        occurs every time we observe then anawin = period_obs.

    Returns
    -------
    rmse : ndarray
        root mean square error of xmean relative to xt. shape: nt
    spread : ndarray
        spread of Xens. shape: nt
        Only returned if Xens != None.
    """

    nx, nt = np.shape(xt)

    # Select only the values at the time of assimilation
    ind = range(0, nt, anawin)
    mse = np.mean((xt[:, ind] - xmean[:, ind])**2, axis=0)
    rmse = np.sqrt(mse)

    if np.any(Xens) != None:
        spread = np.var(Xens[..., ind], ddof=1, axis=1)
        spread = np.mean(spread, axis=0)
        spread = np.sqrt(spread)
        return rmse, spread
    else:
        return rmse


if __name__ == '__main__':
    from plots import plotL96_Linerr
    import matplotlib.pyplot as plt
    deltat = 0.01
    lin_t = 10*deltat
    nx = 40
    np.random.seed(100)
    x0 = np.random.random(nx)
    pert   = -np.ones(nx)*np.sqrt(5/np.pi)
    errors = compute_lin_error(pert, x0, 8.0, lin_t, deltat)
    plotL96_Linerr(*errors)
    plt.show()
