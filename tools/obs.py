"""
"""
import numpy as np


def createH(obsgrid, nx, footprint=None):
    """observation operator for L96 model

    Parameters
    ----------
    obsgrid : str
        Observation grid type. With following choices:
        `all`: observe all grid points
        `1010`: observe every other grid points
        `landsea`: observe left half grid points
        `foot_cent`: only one observation used, which is a weigted
                     average of grid points within a radius of
                     the centre of the domain. The weigting is given
                     by a Gaussian structure function.
        `foot_6`: 6 observations and each observation is a weighted
                  average of gridpoints within a radius of
                  one central grid point. The radius is determined by
                  half of the footprint, and the 6 central grid points
                  are equally distributed on the model domain.
    nx : float
       The size of the state space
    footprint : float
       Number of gridpoints observed by one variable
       only for the obsgrid='foot_6'option

    Returns
    -------
    ny : int
        Number of observations
    H : ndarray
        Linear observation operator
    """
    _ny = {"all": nx, "1010": nx // 2, "landsea": nx // 2, "foot_cent": 1, "foot_6": 6}

    if obsgrid not in _ny:
        raise NotImplementedError(f'observation network option {obsgrid} not supported')

    ny = _ny[obsgrid]
    H = np.zeros((ny, nx), order='F')

    if obsgrid == "all":
        # Observe all
        np.fill_diagonal(H, 1.0)
    elif obsgrid == "1010":
        # Observe every other variable
        for i in range(ny):
            H[i, 2 * i ] = 1.0
    elif obsgrid == "landsea":
        # Observe left half ("land/sea" configuration)
        np.fill_diagonal(H, 1.0)
    elif obsgrid == "foot_cent":
        # Observe footprint of half of the domain (in the centre)
        # A Gaussian-shaped structure function
        L = nx / 4.0  # Lengthscale of the footprint
        factor = -1.0 / (2.0 * L * L)
        centre = nx // 2
        observed = np.arange(centre - nx // 4, centre + nx // 4)
        distance = centre - observed
        H[0, centre-nx//4:centre+nx//4] = np.exp(factor * distance * distance)
        # normalise
        H[0] /= np.sum(H[0])
    elif obsgrid == "foot_6":
        # Six observations footprint of half of the domain
        # (throughout domain)
        # A Gaussian-shaped structure function
        # footprint=5
        assert footprint is not None, "footprint is not given with foot_6 obsgrid"
        L = footprint / 2  # Lengthscale of the footprints
        factor = -1.0 / (2.0 * L * L)
        for i in range(ny):
            centre = int(((i + 0.5) * nx) / ny)
            position = centre - int(L) + np.arange(int(footprint))
            distance = position - centre
            position = np.mod(position, nx)
            H[i, position] = np.exp(factor * distance * distance)
            # normalise
            H[i] /= np.sum(H[i])

    return ny, H


def gen_obs(t, x, period_obs, H, var_obs, seed=None, skip0=False):
    """This function generates (linear) observations from a state.

    Parameters
    ----------
    t : ndarray
        one dimensional array of model times
    x : ndarray
        2D array of the model state. shape: (nx, nobt)
    period_obs : int
        number of time steps between observations
        ? todo: change name?
    H : ndarray
        Observation operator matrix. shape: (ny, nx)
    var_obs : ndarray or int
        the observation error variance.
        It can be a scalar or a 1D array with size of ny
    seed : int
        the random number generator seed
    skip0 : bool
        whether we skip the first time step

    Returns
    -------
    tobs : ndarray
        The one dimensional time array of the observations
    y : ndarray
        The observations, shape: (ny, nobt)
    R : ndarray
        The observational error covariance matrix
    """
    # Extract number of observations (per time) and size of state space
    ny, nx = H.shape

    # Determine the observation times
    tobs = t[::period_obs]

    # Initialise observations array
    y = np.zeros((ny, len(tobs)), order="F")
    y.fill(np.nan)

    # Make the (diagonal) obs error covariance matrix
    R = var_obs * np.eye(ny)

    # The cycle that generates the observations
    np.random.seed(seed)
    std_obs = np.sqrt(var_obs)

    t0 = 1 if skip0 else 0
    for time in range(t0, len(tobs)):
        # Let's do the matrix multiplication explicitly
        # for ob in range(ny):
            # noise = std_obs * np.random.randn()
            # y[ob, time] = np.sum(H[ob, :] * x[:, period_obs * time]) + noise

        # not so sure why it is explicit, more like a fortran way...
        # I think using H@x as written in the equations is straightforward
        # noise = np.random.normal(scale=std_obs, size=ny)
        noise = np.squeeze(std_obs*np.random.randn(ny,1))
        y[:, time] = H @ x[:, period_obs * time] + noise

    return tobs, y, R


def createH_L63(obsgrid, nx):
    """observation operator for L96 model

    Parameters
    ----------
    obsgrid : str
        Observation grid type. With following choices:
        'x' : observe L63 grid
        'y' : observe L63 grid
        'xy' : observe L63 grid
        'xz' : observe L63 grid
        'yz' : observe L63 grid
        'xyz' : observe L63 grid
    nx : float
       The size of the state space

    Returns
    -------
    ny : int
        Number of observations
    H : ndarray
        Linear observation operator
    """
    _obsvar = {"x": [1, 0, 0], "y": [0, 1, 0],
           "z": [0, 0, 1], "xy": [1, 1, 0],
           "xz": [1, 0, 1], "yz": [0, 1, 1],
           "xyz": [1, 1, 1]}
    H = np.diag(_obsvar[obsgrid])
    H = H[~np.all(H == 0, axis=1)]
    ny = len(H)

    return ny, H


if __name__ == "__main__":

    def createTime(t0, tf, deltat, discard):
        t = np.arange(t0 + discard * deltat, tf + deltat / 2, deltat)
        return t

    from L96_model import lorenz96

    nx = 40
    period_obs = 10

    x0 = np.random.random(nx)
    x = lorenz96(x0, 8.0, tf=5.0, deltat=0.01, discard=0)
    t = createTime(0.0, tf=5.0, deltat=0.01, discard=0)
    ny, H = createH("foot_6", nx, 3)
    print("Number of obs per output time", ny)
    print("Size of state                ", nx)
    tobs, y, R = gen_obs(t, x, period_obs, H, 1.0, 1000)
    print("There are " + str(len(tobs)) + " observation times")
    print(tobs)
    print("R-matrix")
    print(R)
    import matplotlib.pyplot as plt

    fig = plt.figure()
    fig.clf()
    fig = plt.figure()
    theta = np.linspace(0, 2 * np.pi, nx + 1)
    theta = H @ theta[:-1]
    x = x[:, ::period_obs]
    for xt, yt in zip(x.T, y.T):
        fig.clf()
        ax = fig.add_subplot(projection="polar")
        truth = H @ xt
        ax.plot(theta, truth, color="b")
        ax.plot(theta, yt, color="r")
        plt.pause(0.01)
    plt.show()
