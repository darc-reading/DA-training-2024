import warnings
import numpy as np
from tools.cov import msq, minv


def kfs(x0_t, param, f, t, tobs, y, H, R, rho, ne, met,
        lam=None, loctype=None, back0='random', desv=2.0,
        param_estimate=False, alpha=None, seed=None, adaptive=False):
    """Data assimilation for Lorenz 1996 using Ensemble Kalman Filters.

    Parameters
    ----------
    x0_t : ndarray
        the real initial position
    param : int or list or tuple
        the initial model parameter, in L96 it is F, in L63 it is (sigma, b, r)
    f : func
        model propagation function. arguments: x0, tf, deltat, discard
    t : ndarray
        time array of the model (should be evenly spaced)
    tobs : ndarray
        time array of the observations (should be evenly spaced
        with a timestep that is a multiple of the model timestep)
    y : ndarray
        the observations. shape: (ny, nt)
    H : ndarray
        observation matrix
    R : ndarray
        the observational error covariance matrix
    rho : ndarray
        inflation for P.  Notice we multiply (1+rho)*Xpert
        or P*(1+rho)^2.
    ne : int
        number of ensemble member
    met : str
        a string containing the method: 'SEnKF', 'ETKF'
    lam : int
        the localization radius in gridpoint units.  If None,
        it means no localization.
    loctype : str
        a string indicating the type of localization: 'GC'
        to use the Gaspari-Cohn function, 'cutoff' for a sharp cutoff
    back0 : str
        Sampling the initial condition based on two methods.
        'random': initialise ensemble from sampled normal distribution
        'fixed': initialised ensemble from uniform intervals
    desv : float
        The range of the state ensemble.
        'random' : the standard deviation of normal
        'fixed': the range of uniform intervals
    param_estimate : bool
        Whether parameter estimation is used.
    alpha : float
        The range of the parameter ensemble.
        'random' : the standard deviation of normal
        'fixed': the range of uniform intervals
    seed : int
        Random seed for initial ensemble
    adaptive : bool
        switch for adaptive inflation (not implemented)

    Returns
    -------
    Xb: ndarray
        the background ensemble 3D array. shape: [nx, ne, nt]
    xb: ndarray
        background mean. shape: [nx, nt]
    Xa: ndarray
        the analysis ensemble 3D array. shape: [nx, ne, nt]
    xa: ndarray
        analysis mean. shape: [nx, nt]
    locmatrix: ndarray
        localization matrix (or None if lam is None)
    """

    # General settings
    # Number of observations and variables
    nt = len(t)
    ny, nx = H.shape
    if type(param) is float:
        param = np.array([param])
    Nparam = len(param)
    # For the true time
    tstep_truth = t[1] - t[0]
    # For the analysis (we assimilate everytime we get observations)
    tstep_obs = tobs[1] - tobs[0]
    # The ratio
    o2t = np.rint(tstep_obs / tstep_truth).astype(int)

    # Precreate the arrays for background and analysis
    Xb = np.empty((nx, ne, nt), order='F')
    Xb.fill(np.nan)
    Xa = np.empty((nx, ne, nt), order='F')
    Xa.fill(np.nan)
    Param_a = np.zeros((Nparam, ne, len(tobs)), order='F')
    # the state vector into the EnKF
    X = np.empty((nx + Nparam, ne), order='F') if param_estimate \
        else np.empty((nx, ne), order='F')
    Hall = np.hstack((H,np.zeros((ny, Nparam)))) if param_estimate \
        else H
    X.fill(np.nan)

    np.random.seed(seed)
    # For the original background ensemble
    # Fixed initial conditions for our ensemble (created ad hoc)
    if back0 == "fixed":
        Xb[..., 0] = np.linspace(-desv, desv, ne, endpoint=True)[
            None, :
        ]
        if param_estimate:
            Param_a[..., 0] = np.linspace(-alpha, alpha, ne, endpoint=True)[
                None, :
            ]
    elif back0 == "random":
        Xb[..., 0] = np.random.normal(size=(nx, ne), scale=desv)
        Xb[..., 0] = Xb[..., 0] - np.mean(Xb[..., 0], axis=-1, keepdims=True)
        if param_estimate:
            Param_a[..., 0] = np.random.normal(size=(nx, ne), scale=desv)
            Param_a[..., 0] = Param_a[..., 0] - \
                              np.mean(Param_a[..., 0], axis=-1, keepdims=True)
    else:
        Xb[..., 0] = 0.
        Param_a[..., 0] = 0.

    Xb[..., 0] = Xb[..., 0] + x0_t[:, None]
    Param_a[:] = Param_a[:] + param[:, None, None]

    # Since we don't have obs at t=0 the first analysis is the same as
    # background
    Xa[..., 0] = Xb[..., 0]

    localDomainObsMask = None
    # Getting the R-localization weights
    if lam != None:
        locmatrix = getlocmat(nx, ny, H, lam, loctype)
        Blocmatrix = getlocmat(nx, nx, np.eye(nx), lam, loctype)

        if met == 'ETKF':
            localDomainObsMask = getObsForLocalDomain(nx, lam, H)
    else:
        locmatrix = None
        Blocmatrix = np.ones((nx, nx))


    # if met != 'ETKF' or lam == None:
    #     adaptive = False

    # # deal with adaptive inflation
    # if adaptive:
    #     rho = np.empty(nx, order='F')
    #     rho[:] = 0.05


    # The following cycle evolve and assimilate for all time steps
    for j in range(len(tobs) - 1):
        # Evolve from analysis!
        xold = Xa[..., j * o2t]
        paramold = Param_a[..., j] # [Nparam,M]

        # Time goes forward
        xnew = evolvemembers(xold, tstep_truth, o2t, f, paramold)
        # The new background
        Xb[..., j * o2t + 1 : (j + 1) * o2t + 1] = xnew[..., 1:]
        Xa[..., j * o2t + 1 : (j + 1) * o2t + 1] = xnew[..., 1:]

        X[:nx] = Xb[..., (j + 1) * o2t]
        if param_estimate:
            X[nx:] = paramold + alpha*np.random.randn(Nparam,ne)

        # The assimilation
        X, rho = enkfs(X, y[:, j + 1], Hall, R, rho, met,
                       localDomainObsMask, locmatrix, adaptive)
        Xa[..., (j + 1) * o2t] = X[:nx]
        if param_estimate:
            Param_a[..., j+1] = X[nx:]

    # The background and analysis mean
    x_b = np.mean(Xb, axis=1)  # [N,M,t] -> [N, t]
    x_a = np.mean(Xa, axis=1)  # [N,M,t] -> [N, t]
    param_a = np.mean(Param_a, axis=1)

    if param_estimate:
        return Xb, x_b, Xa, x_a, Param_a, param_a, locmatrix, Blocmatrix
    else:
        return Xb, x_b, Xa, x_a, locmatrix, Blocmatrix


def evolvemembers(xold, deltat, o2t, f, param):
    """Evolving the members.

    Parameters
    ----------
    xold : ndarray
        an [nx, n] array of initial conditions for the
        ne members and nx variables
    tstep_truth : float
        the time step used in the nature run
    o2t : int
        frequency of observations in time steps
    f : func
        model propagation function. arguments: x0, tf, deltat, discard
    Returns
    -------
    xnew : ndarray
        a [nx, n, o2t+1] array with the evolved members
    """

    tf = o2t*deltat
    nx, ne = xold.shape
    xnew = np.empty((nx, ne, o2t+1))
    xnew.fill(np.nan)

    for j in range(ne):
        xnew[:, j] = f(xold[:, j], tf,
                       deltat, 0, param[:, j])
    return xnew


##############################################################################
## The EnKF algorithms
def enkfs(Xb, y, H, R, rho, met, localDomainObsMask, locmatrix, adaptive):
    """Performs the analysis using different EnKF methods.

    Parameters
    ----------
    Xb : ndarray
        the ensemble background (nx, ne)
    y : ndarray
        the observations (ny)
    H : ndarray
        the observation matrix (ny, nx)
    R : ndarray
        the obs error covariance matrix (ny, ny)
    rho : float
        inflation for P.  Notice we multiply (1+rho)*Xpert
        or P*(1+rho)^2.
    met : str
        a string that indicated what method to use
    localDomainObsMask : list 
        observation mask used for each local analysis domain
        each element of the list contains observation mask for each domain
    locmatrix : ndarray
        localization matrix (nx, ny)
    adaptive : bool
        switch for adaptive inflation

    Returns
    -------
    Xa : ndarray
        the full analysis ensemble (nx, ne)
    """
    # General settings
    # The background information
    y = y.reshape(len(y), 1) # array -> column vector
    sqR = msq(R)
 
    # Number of state variables, ensemble members and observations
    nx, ne = Xb.shape
    ny, _ = H.shape

    # Auxiliary matrices that will ease the computation of averages and
    # covariances
    U = np.ones((ne, ne))/ne
    I = np.eye(ne)

    # Create the ensemble in Y-space
    Yb = np.empty((ny, ne))
    Yb.fill(np.nan)

    inv_ne = 1.0/(ne-1)
    # Now, we choose from one of three methods
    # Stochastic Ensemble Kalman Filter
    if met=='SEnKF':
        # The ensemble is inflated (rho can be zero)
        Xb_pert = (1+rho)*(Xb@(I-U))
        Xb = Xb_pert + Xb@U

        # Map every ensemble member into observation space
        for jm in range(ne):
            Yb[:,jm] = H@Xb[:,jm]
        # The matrix of perturbations
        Yb_pert = Yb@(I-U)

        if np.all(locmatrix) == None:
            # The Kalman gain matrix without localization
            Khat = inv_ne*Xb_pert@Yb_pert.T @ minv(inv_ne*Yb_pert@Yb_pert.T+R)
        else:
            # The Kalman gain with localization
            Caux = locmatrix * (Xb_pert@Yb_pert.T)
            Khat = inv_ne*Caux@minv(inv_ne*H@Caux+R)

        # Fill Xa (the analysis matrix) member by member using perturbed observations
        Xa = np.empty((nx, ne), order='F')
        Xa.fill(np.nan)
        for jm in range(ne):
            yaux = y + sqR@np.random.randn(ny, 1)
            yaux = np.squeeze(yaux)
            Xa[:, jm] = Xb[:,jm] + Khat@(yaux - Yb[:,jm])

    # Ensemble Transform Kalman Filter
    elif met=='ETKF':
        if np.all(locmatrix) == None:
            # The ensemble is inflated (rho can be zero)
            Xb_pert = (1+rho)*(Xb@(I-U))
            Xb = Xb_pert + Xb@U

            # Map every ensemble member into observation space
            for jm in range(ne):
                Yb[:,jm] = H@Xb[:,jm]
            # The matrix of perturbations
            Yb_pert = Yb@(I-U)

            # Means
            xb_bar = Xb@np.ones((ne,1))/ne
            yb_bar = Yb@np.ones((ne,1))/ne

            # The method without localization (ETKF)
            Pa_ens = minv((ne-1)*np.eye(ne)+Yb_pert.T@minv(R)@Yb_pert)
            Wa = msq((ne-1)*Pa_ens) # matrix square root (symmetric)
            wa = Pa_ens@(Yb_pert.T@(minv(R)@(y-yb_bar)))
            # get analysis perturbations
            Xa_pert = Xb_pert@Wa
            xa_bar = xb_bar + Xb_pert@wa
            Xa = Xa_pert + xa_bar@np.ones((1,ne))
        else:
            Xb_pert = Xb@(I-U)
            # Map every ensemble member into observation space
            for jm in range(ne):
                Yb[:,jm] = H@Xb[:,jm]
            # The matrix of perturbations
            Yb_pert = Yb@(I-U)
            # Means
            xb_bar = Xb@np.ones((ne,1))/ne
            yb_bar = Yb@np.ones((ne,1))/ne
            Xa, rho = letkf(Xb_pert, xb_bar, Yb_pert, yb_bar, y, H, rho,
                       localDomainObsMask, locmatrix, R, adaptive)
    else:
        raise NotImplementedError(f'{met} is not implemented, try ETKF or SEnKF')
  
    return Xa, rho


def letkf(Xb_pert, xb_bar, Yb_pert, yb_bar, y, H, rho, mask, locmatrix, R, adaptive):
    """local ensemble transform Kalman filter
    """
    nx, ne = Xb_pert.shape
    Xa_pert = np.zeros_like(Xb_pert)
    Xa = np.zeros_like(Xb_pert)
    xa_bar = np.zeros_like(xb_bar)
    for i in range(nx):
        if np.all(~mask[i]): continue
        # get observations in i-th local domain
        Yb_pert_aux = Yb_pert[mask[i]]
        locmatrix_aux = np.diag(locmatrix[i, mask[i]])
        Rinv = locmatrix_aux*minv(R[mask[i]][:, mask[i]])
        d_aux = y[mask[i]]-yb_bar[mask[i]]
        # if adaptive:
        #     rho[i] = adaptive_inflation(locmatrix_aux, Yb_pert_aux, ne, Rinv, rho[i], d_aux)
        #     if rho[i]<0.01 or rho[i]>0.4:
        #         rho[i] = 0.05
        #     Yb_pert_aux = (1.0 + rho[i]) * Yb_pert_aux
        #     Xb_pert[i] = (1.0 + rho[i]) * Xb_pert[i]
        # else:
        Yb_pert_aux = (1.0 + rho) * Yb_pert_aux
        Xb_pert[i] = (1.0 + rho) * Xb_pert[i]
        # do ETKF as usual
        Pa_ens = minv((ne-1)*np.eye(ne)+Yb_pert_aux.T@Rinv@Yb_pert_aux)
        # get w vector
        Wa = msq((ne-1)*Pa_ens) # matrix square root (symmetric)
        wa = Pa_ens@(Yb_pert_aux.T@(Rinv@d_aux))
        # get analysis perturbations
        Xa_pert[i] = Xb_pert[i]@Wa
        # get analysis mean
        xa_bar[i] = xb_bar[i] + Xb_pert[i]@wa
        Xa[i] = Xa_pert[i] + xa_bar[i]@np.ones((1,ne))
    return Xa, rho


def adaptive_inflation(locmatrix_aux, Yb_pert_aux, M, loc_invR, rhob, d_aux):
    loc_tr = np.trace(locmatrix_aux)          
    vb = 0.05**2.0 # This is something prescribed and tuned!
    den = np.trace(np.dot(Yb_pert_aux,Yb_pert_aux.T)/(M-1.0)* loc_invR)
    alphab = (1+rhob)**2
    alphao = (np.trace( np.dot(d_aux,d_aux.T)* np.mat(loc_invR)) - loc_tr)/den
    vo = 2/np.trace(locmatrix_aux)*((alphab*den + loc_tr)/den)**2
    alphaa = (alphab*vo + alphao*vb)/(vo+vb)
    return np.sqrt(alphaa)-1


def getObsForLocalDomain(nx, lam, H):
    """ obtain a list of masks which gives
    the observations used for each local domain.
    
    This function assumes that the coordinate is periodic
    1D integers such as the index of Lorenz96 model.

    Parameters
    ----------
    nx : int
        dimension of the state vector
    H : np.ndarray
        observation operator
    lam : float
        localisation radius
    Returns
    -------
    : list
        observation mask used for each local analysis domain
        each element of the list contains observation mask for each domain
    """
    # this is a rule of thumb, could be refined
    influence_dist = np.ceil(nx/6.0)
    # get the left and right end of the domain of influence
    if lam >= influence_dist:
        lim1 = np.zeros(nx, dtype=int)
        lim2 = (nx-1)*np.ones(nx, dtype=int)
    else:
        x = np.arange(nx)
        lim1 = np.roll(x, 3*int(np.ceil(lam)))
        lim2 = np.roll(x, -3*int(np.ceil(lam)))

    # get the number of grid points in the domain of influence
    n = lim2[0] - lim1[0] + 1
    # when the domain cross the boundary
    if n <= 0: n += nx 
    # the special case when we should include entire domain
    if lam > 0. and n == 1: n = nx

    # get the grid points in the domain of influence
    radius = np.zeros((nx, n), dtype=int)
    for i, (l1, l2) in enumerate(zip(lim1, lim2)):
        if l1>l2:
            radius[i] = np.append([np.arange(0,l2+1)], [np.arange(l1,nx)])
        elif l1<l2:   
            radius[i] = np.arange(l1,l2+1,1)
        else:
            radius[i] = np.arange(nx) if lam > 0. else l1

    # get observations mask within the radius of local domain
    return [np.sum(H[:, radius[i]] > 0, axis=1, dtype=bool) for i in range(nx)]


def getlocmat(nx, ny, H, lam, loctype):
    """Obtain localisation matrix

    Parameters
    ----------
    nx : int
        number of model state variables
    ny : int
        number of observations
    H : ndarray
        time array of the observations (should be evenly spaced
        with a timestep that is a multiple of the model timestep)
    lam : int
        the localization radius in gridpoint units.  If None,
        it means no localization.
    loctype : str
        a string indicating the type of localization: 'GC'
        to use the Gaspari-Cohn function, 'cutoff' for a sharp cutoff

    Returns
    -------
    locmatrix : ndarray
        localization matrix
    """
    assert loctype is not None, f'lam={lam}, loctype cannot be None'
    # To get the localization weights.
    indx = np.array(range(nx)).reshape(nx, 1)
    indy = H@indx
    dist = np.empty((nx, ny))
    dist.fill(np.nan)

    # First obtain a matrix that indicates the distance (in grid points)
    # between state variables and observations
    for jrow in range(nx):
        for jcol in range(ny):
            dist[jrow,jcol] = np.amin([abs(indx[jrow]-indy[jcol]),\
                                       nx-abs(indx[jrow]-indy[jcol])])

    # Now we create the localization matrix
    # If we want a sharp cuttof
    if loctype=='cutoff':
        locmatrix = 1.0*(dist<=lam)

    # If we want something smooth, we use the Gaspari-Cohn function
    elif loctype=='GC':
        locmatrix = np.empty_like(dist, order='F')
        locmatrix.fill(np.nan)
        for j in range(ny):
            locmatrix[:,j] = gasparicohn(dist[:,j],lam)

    return locmatrix


def gasparicohn(z, lam):
    """The Gaspari-Cohn function.

    Parameters
    ----------
    z : ndarray
        distance between model state
    lam : int
        the localization radius in gridpoint units.  If None,
        it means no localization.

    Returns
    -------
    C0 : ndarray
        localization matrix
    """
    c = lam/np.sqrt(3.0/10)
    zn = abs(z)/c
    C0 = np.zeros_like(zn)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'divide by zero encountered in reciprocal')
        C0 = np.where(zn <= 1, - 1.0/4*zn**5 + 1.0/2*zn**4 + \
                               5.0/8*zn**3 - 5.0/3*zn**2 + 1,
                      np.where((zn > 1) & (zn <= 2), 
                               1.0/12*zn**5 - 1.0/2*zn**4 + \
                               5.0/8*zn**3 + 5.0/3*zn**2 - \
                               5*zn + 4 - 2.0/3*zn**(-1),
                               0
                               )
                      )

    return C0


if __name__ == '__main__':
    from tools.obs import gen_obs, createH
    from tools.L96_model import lorenz96
    from tools.misc import createTime
    np.random.seed(10)
    nx = 12
    F = 8
    x0 = np.random.random(nx)
    tf = 4
    deltat = 0.025
    #Create truth.
    xt = lorenz96(x0, tf, deltat, 0, 8.0)
    t = createTime(0., tf, deltat, 0)
    _, H = createH('1010', nx)
    tobs, y, R = gen_obs(t, xt, 2, H, 2.0, 1)
    Xb, xb, Xa, xa, L_obs, L_x = kfs(x0, 8.0, lorenz96, t, tobs, y,
                                     H, R, 0., 24,
                                     'ETKF', lam=2, loctype='GC',
                                     desv=1.0)