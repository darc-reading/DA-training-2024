import numpy as np
from tools.cov import msq, minv
from tools.L96_model import lorenz96
from tools.enkf import enkfs, evolvemembers, getlocmat, getObsForLocalDomain
from tools.var4d_cvt import one4dvarPC


def En4DVar(x0, t, period_obs, anawin, ne, y_traj, H, B, R, beta, F,
    rho, lam=None, loctype=None):                
    """4DVar-ETKF cycling data assimilation routine for Lorenz 1996 using

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
    ne : int
        number of ensemble memebrs
    y_traj : ndarray
        the observations (all cycles). shape: ny, nt
    H : ndarray
        observation matrix. shape: ny, nx
    B : ndarray
        the background error covariance matrix for 4DVar. shape: nx, nx
    R : ndarray
        the observation error covariance matrix. shape: ny, ny
    beta : list
        a two element list that contains the weighting for each covariance matrix 
    F : float
        the forcing used in the L96 model when cycling
    rho : ndarray
        inflation for P.  Notice we multiply (1+rho)*Xpert
        or P*(1+rho)^2.
    lam : int
        the localization radius in gridpoint units.  If None,
        it means no localization.
    loctype : str
        a string indicating the type of localization: 'GC'
        to use the Gaspari-Cohn function, 'cutoff' for a sharp cutoff

    Returns
    -------
    xb_traj : ndarray
        the background trajectory. shape: nx, nt
    xa_traj : ndarray
        the analysis trajectory. shape: nx, nt

    Xa_kf : ndarray
        the analysis ensemble anomaly trajectory from LETKF. shape: nx, nt
    xa_kf : ndarray
        the analysis mean trajectory from LETKF. shape: nx, nt
    """
    # General settings
    nt = len(t)  # Number of timesteps in total
    nx = len(x0)  # Number of state variables
    ny = len(H)   # number of observations
    if type(F) is float:
        # this is used to acommodate the evolvemembers function
        # because it assumes an ensemble of parameters
        Fp = F*np.ones((1, ne))
    # Model's timestep
    deltat = t[1] - t[0]

    # Number of 4D-Var cycles that will be done here
    anal_nt = period_obs*anawin
    ncycles = int(float(nt) / float(anal_nt))

    # random seed
    np.random.seed(0)

    # ETKF related variables
    # domain localisation weight
    # Getting the R-localization weights
    if lam != None:
        locmatrix = getlocmat(nx, ny, H, lam, loctype)
        Blocmatrix = getlocmat(nx, nx, np.eye(nx), lam, loctype)
        localDomainObsMask = getObsForLocalDomain(nx, lam, H)
    else:
        locmatrix = None
        Blocmatrix = np.ones((nx, nx))
        localDomainObsMask = None

    B0sq = msq(B)
    # initial guess for ensemble
    xboldens = np.empty((nx, ne))
    xaoldens = np.empty((nx, ne))
    for m in range(ne):
        xboldens[:,m] = x0 + B0sq@np.random.randn(nx)
    xaoldens[:] = xboldens[:]

    # ensemble mean
    xa_kf = np.empty([nx, nt])
    xa_kf.fill(np.nan)
    # ensemble anomaly trajectory
    Xa_kf = np.empty([nx, ne, nt])
    Xa_kf.fill(np.nan)

    # VAR method related variables
    # Make blank arrays for background and analysis trajectories
    xb_traj = np.empty([nx, nt])
    xb_traj.fill(np.nan)
    xa_traj = np.empty([nx, nt])
    xa_traj.fill(np.nan)
    # This is the background for the first instant
    xb_traj[:, 0] = x0[:]
    xa_traj[:, 0] = x0[:]

    # pre-compute certain matrix for efficiency
    invR = np.linalg.inv(R)
    HTinvR = H.T@invR
    nouterloops = 1

    # The following cycle contains evolution and assimilation
    # Loop around 4D-Var cycles
    for cycle in range(ncycles):
        
        # Extract observations for this cycle
        yaux = y_traj[:, cycle * anawin + 1 : (cycle + 1) * anawin + 1]
        # compute the Pb from the ensemble:
        Pb = computePb(xboldens, nx, ne)

        # do LETKF
        for it in range(anawin):
            start = cycle*anal_nt + it*period_obs
            end = cycle*anal_nt + (it+1)*period_obs
            xnew = evolvemembers(xaoldens, deltat, period_obs, lorenz96, Fp)
            xa_kf[..., start:end] = np.mean(xnew[..., :period_obs], axis=1)
            Xa_kf[..., start:end] = xnew[..., :period_obs] - xa_kf[..., None, start:end]

            Xa, rho = enkfs(xnew[..., period_obs], yaux[:, it],
                            H, R, rho, 'ETKF',
                            localDomainObsMask, locmatrix, False)
            
            xaoldens = Xa.copy()
            xa_kf[..., end] = np.mean(Xa, axis=1)
            Xa_kf[..., end] = Xa - xa_kf[..., None, end]
        xboldens = xnew[..., period_obs]

        # Compute the hybrid matrix
        Ph = beta[0]*B + beta[1]*Pb*Blocmatrix
        Phsq = msq(Ph)

        # do 4DVar
        # This cycle corresponds to the following start and end times
        start = cycle * anal_nt
        end = start + anal_nt
        xbaux_traj, xaaux_traj = one4dvarPC(xa_traj[:, start], deltat, anawin, yaux, H, Phsq,
                                            invR, period_obs, HTinvR, F, nouterloops)

        # Store this for reference
        xa_traj[:, start : end + 1] = xaaux_traj[:, :anal_nt + 1]
        xb_traj[:, start : end + 1] = xbaux_traj[:, :anal_nt + 1]
                
    return xb_traj, xa_traj, Xa_kf, xa_kf
    

def computePb(X, Nx, ne):
    Xpert = 1/np.sqrt(ne) * (X - np.mean(X, axis=1, keepdims=True))
    Pb = Xpert@Xpert.T
    return Pb


def msq(B):
    U, s, Vh = np.linalg.svd(B)
    B_sq = (U * np.sqrt(s[..., None, :])) @ Vh
    return B_sq
