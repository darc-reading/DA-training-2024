"""
"""
import functools

import numpy as np

from tools.timestepping import rk4, rk4TLM, rk4ADJ, euler


def lorenz96(x0, tf, deltat, discard, param):
    """time integration of L96

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
    param : float
       The forcing parameter
    Returns
    -------
    xf : ndarray
        The forecast time sequence. shape: (nx, nt)
    """
    # Size of state space
    nx = len(x0)
    # Number of time steps (+1 as t=0 included also)
    nt = int(tf / deltat) + 1
    # Set-up the output matrix
    xf = np.empty((nx, nt), order="F")
    xf.fill(np.nan)

    # giving model parameters
    model = functools.partial(f, F=param)
    # Run a number of timesteps to spin-up the model
    # (these are discarded)
    for time in range(discard):
        x0[:] = x0[:] + rk4(x0[:], deltat, model)

    # Initial conditions for part of integration which is kept
    xf[:, 0] = x0[:]

    # Timesteps solved via RK4
    for time in range(nt - 1):
        xf[:, time + 1] = xf[:, time] + rk4(xf[:, time], deltat, model)
    return xf


def Lorenz96_and_TLM(dx0, nt, x0, F, deltat):
    """Propogate the perturbation `dx` using the the TLM by nt
        timesteps. x0 is the full state at the initial time.

    Parameters
    ----------
    dx0 : ndarray
        Perturbation to the model
    nt : int
        total number of time steps
    x0 : ndarray
        1D initial condition of the state vector
    deltat : float
        time step

    Returns
    -------
    xf : ndarray
        model trajectory
    dx_traj : ndarray
        tangent linear model. shape: (nx, nx, nt+1)
    """
    nx = len(dx0)
    dx_traj = np.zeros([nx, nx, nt + 1], order='F')
    dx_traj[..., 0] = dx0
    xf = np.zeros((nx, nt + 1), order="F")
    xf[..., 0] = x0
    model = functools.partial(f, F=F)
    for it in range(nt):
        delta, delta_dx = rk4TLM(xf[..., it], dx_traj[..., it], deltat, tlm_matrix, model)
        xf[..., it + 1] = xf[..., it] + delta
        dx_traj[..., it + 1] = dx_traj[..., it] + delta_dx
    return xf, dx_traj


def Lorenz96_TL_propogation(dx0, nt, x0, F, deltat):
    """Propogate the perturbation `dx` using the the TLM by nt
        timesteps. x0 is the full state at the initial time.

    Parameters
    ----------
    dx0 : ndarray
        Perturbation to the model
    nt : int
        total number of time steps
    x0 : ndarray
        1D initial condition of the state vector
    deltat : float
        time step

    Returns
    -------
    X_p_traj : ndarray
        trajectory of model state perturbations. shape: (nx, nt+1)
    """
    nx = len(dx0)
    dx_traj = np.empty([nx, nt + 1], order='F')
    dx_traj[:, 0] = dx0
    xf = x0.copy()
    model = functools.partial(f, F=F)
    for it in range(nt):
        delta, delta_dx = rk4TLM(xf, dx_traj[:, it], deltat, tlm_matrix, model)
        dx_traj[:, it + 1] = dx_traj[:, it] + delta_dx
        xf += delta
    return dx_traj


def Lorenz96_TL_propogation_adj(dx, nt, x0, F, deltat):
    """Propogate backwards the perturbation dx using the the
       ADJOINT of the model.

    Parameters
    ----------
    dx : ndarray
        Perturbation to the model
    nt : int
        total number of time steps
    x0 : ndarray
        1D initial condition of the state vector
    deltat : float
        time step

    Returns
    -------
    dx0 : ndarray
        initial model perturbations. shape: nx
    """
    # Do a non-linear forward run in order to obtain the reference points
    x_traj = lorenz96(x0, nt * deltat, deltat, 0, F)
    dx0 = dx.copy()
    model = functools.partial(f, F=F)
    for it in range(nt-1, -1, -1):
        dx0 += rk4ADJ(x_traj[:, it], dx0, deltat, fadj_matrix, model)
    return dx0


def Lorenz96_TL1 (dx0, x0, F, deltat):
    """Propogate the perturbation `dx` using the the TLM by one
        timesteps. x0 is the full state at the initial time.

    Parameters
    ----------
    dx0 : ndarray
        Perturbation to the model
    nt : int
        total number of time steps
    x0 : ndarray
        1D initial condition of the state vector
    deltat : float
        time step

    Returns
    -------
     : ndarray
        model state
     : ndarray
        model perturbation
    """
    model = functools.partial(f, F=F)
    delta, delta_dx = rk4TLM(x0, dx0, deltat, tlm_matrix, model)
    return dx0 + delta_dx, x0 + delta


def Lorenz96_TL1_adj (dx, x0, F, deltat):
    """Propogate backwards the perturbation dx using the the
       ADJOINT of the model by one step.

    Parameters
    ----------
    dx : ndarray
        Perturbation to the model
    nt : int
        total number of time steps
    x0 : ndarray
        1D initial condition of the state vector
    deltat : float
        time step

    Returns
    -------
    dx0 : ndarray
        initial model perturbations. shape: nx
    """
    model = functools.partial(f, F=F)
    dx0 = dx + rk4ADJ(x0, dx, deltat, fadj_matrix, model)
    return dx0


def Lorenz96_TL_simple(x_traj, dx0, deltat, integration_type):
    """Propogate the perturbation `dx` using the the TLM by nt
        timesteps. Based on given model trajectory

    Parameters
    ----------
    x_traj : ndarray
        model trajectory. shape: (nx, nt)
    dx0 : ndarray
        Perturbation to the model at initial time step
    deltat : float
        time step
    integration_type : str
        numerical integration scheme

    Returns
    -------
    X_p_traj : ndarray
        trajectory of model state perturbations. shape: (nx, nt+1)
    """
    nx, nt = x_traj.shape
    dx_traj = np.empty([nx, nt], order='F')
    dx_traj[:, 0] = dx0
    integrator = eulerTLM if integration_type == 'Euler' else rk4TLM
    for it in range(nt-1):
        _, delta_dx = integrator(x_traj[:, it], dx_traj[:, it], deltat, tlm_matrix)
        dx_traj[:, it + 1] = dx_traj[:, it] + delta_dx
    return dx_traj


def Lorenz96_TL_simple_adj(x_traj, dx, deltat, integration_type):
    """Propogate backwards the perturbation dx using the the
       ADJOINT of the model.

    Parameters
    ----------
    x_traj : ndarray
        model trajectory. shape: (nx, nt)
    dx : ndarray
        Perturbation to the model at final step
    deltat : float
        time step
    integration_type : str
        numerical integration scheme

    Returns
    -------
    dx0 : ndarray
        initial model perturbations. shape: nx
    """
    # Do a non-linear forward run in order to obtain the reference points
    nx, nt = x_traj.shape
    dx0 = dx.copy()
    integrator = eulerADJ if integration_type == 'Euler' else rk4ADJ
    for it in range(nt-2, -1, -1):
        dx0 += integrator (x_traj[:,it], dx0, deltat, fadj_matrix)
    return dx0


def Lorenz96_TL1_simple (x, dx0, deltat, integration_type):
    """Propogate the perturbation `dx` using the the TLM by one
        timesteps. Based on given model trajectory

    Parameters
    ----------
    x : ndarray
        model trajectory. shape: (nx, nt)
    dx0 : ndarray
        Perturbation to the model at initial time step
    deltat : float
        time step
    integration_type : str
        numerical integration scheme

    Returns
    -------
     : ndarray
        model perturbation
    """
    integrator = eulerTLM if integration_type == 'Euler' else rk4TLM
    _, delta_dx = integrator(x, dx0, deltat, tlm_matrix)
    return dx0 + delta_dx


def Lorenz96_TL1_simple_adj (x, dx, deltat, integration_type):
    """Propogate the perturbation `dx` backward by one step
    using adjoint model

    Parameters
    ----------
    x : ndarray
        model trajectory. shape: (nx, nt)
    dx : ndarray
        Perturbation to the model at initial time step
    deltat : float
        time step
    integration_type : str
        numerical integration scheme

    Returns
    -------
    dx : ndarray
        model perturbation
    """
    integrator = eulerADJ if integration_type == 'Euler' else rk4ADJ
    dx0 = dx + integrator (x, dx, deltat, fadj_matrix)
    return dx0


def shift(x, n):
    """help function for L96

    This function shifts the entire array periodically

    Parameters
    ----------
    x : ndarray
        1D array of L96 state at one time step
    n : int
        The number of shifts of the array.
        When n > 0, it shift to the left (equal to using j + n)
        When n < 0, it shift to the right (equal to j - n)

    Returns
    -------
     : ndarray
        The shifted array
    """
    return np.roll(x, -n, axis=0)


def f(x, F):
    """time tendency of Lorenz-96 model d/dt

    Parameters
    ----------
    x : ndarray
        1D array of L96 state at one time step
    F : float
        The forcing parameter
    Returns
    -------
    k : ndarray
        df/dt as described by the Lorenz-96 model
    """

    # --- legacy code ---
    # n = len(x)
    # k = np.empty_like(x)
    # k.fill(np.nan)
    # # Remember it is a cyclical model, hence we need modular algebra
    # for j in range(n):
    #   k[j] = (x[(j+1)%n]-x[(j-2)%n]) * x[(j-1)%n] - x[j] + F

    # for efficiency reasons change this line of code
    k = (shift(x, 1) - shift(x, -2)) * shift(x, -1) - x + F
    return k


def tlm_matrix(x):
    """The tangent linear model matrix of L96

    Parameters
    ----------
    x : ndarray
        1D array of L96 state at one time step
    nx : int
        number of model state
    Returns
    -------
    TLM : ndarray
        TLM matrix
    """
    nx = len(x)
    TLM = np.zeros((nx,nx))
    for i in range(nx):
        TLM[i,i-2] = -x[i-1] 
        TLM[i,i-1] = -x[i-2] + x[(i+1)%nx]
        TLM[i,i] = -1
        TLM[i,(i+1)%nx] = x[i-1]
    return TLM


def tlm(x, dx):
    """The tangent linear model matrix of L96

    Parameters
    ----------
    x : ndarray
        1D array of L96 state at one time step
    dx : ndarray
        perturbation
    Returns
    -------
    TLM : ndarray
        TLM matrix
    """
    nx = len(x)
    TLM = np.zeros((nx))
    for j in range(nx):
        TLM[j] = ( -x[j-1] * dx[j-2] +
                   (x[(j+1)%nx]-x[j-2])*dx[j-1] -
                   dx[j] +
                   x[j-1] * dx[(j+1)%nx]
                  )
    return TLM


def fadj_matrix(x):
    """The adjoint model matrix

    Parameters
    ----------
    x : ndarray
        1D array of L96 state at one time step
    dx : ndarray
        the input perturbation
    Returns
    -------
    dx_new : ndarray
        adjoint model tendency
    """
    nx = len(x)
    adj = np.zeros((nx, nx))
    for i in range(nx):
        adj[i, i] = -1
        adj[i, i-1] = x[i-2]
        adj[i, (i+1)%nx] = x[(i+2)%nx] - x[i-1]
        adj[i, (i+2)%nx] = -x[(i+1)%nx]

    return adj


def fadj(x, dx):
    """The adjoint model matrix

    Parameters
    ----------
    x : ndarray
        1D array of L96 state at one time step
    dx : ndarray
        the input perturbation
    Returns
    -------
    dx_new : ndarray
        adjoint model tendency
    """
    nx = len(x)
    adj = np.zeros((nx))
    for j in range(nx):
        adj[j-2] -= x[j-1] * dx[j]
        adj[j-1] += (x[(j+1)%nx]-x[j-2]) * dx[j]
        adj[j]   -= dx[j]
        adj[(j+1)%nx] += x[j-1] * dx[j]
    return adj


if __name__ == "__main__":
    nx = 40
    F = 8.0
    deltat = 0.01
    tf = 0.1
    x0 = np.random.random(nx)
    dx0 = np.random.random(nx)
    x = lorenz96(x0, tf, deltat, 0, F)
    _, nt = x.shape
    print (nt)
    dx = Lorenz96_TL_propogation(dx0, nt, x0, F, deltat)
    dx_adj = Lorenz96_TL_propogation_adj(dx[:,-1], nt, x0, F, deltat)
    print (np.sum(dx[:, -1]*dx[:, -1]) - np.sum(dx_adj*dx0))

    dx_t = Lorenz96_TL_simple(x, dx0, deltat, 'rk4')
    dx_adj_t = Lorenz96_TL_simple_adj(x, dx_t[:, -1], deltat, 'rk4')
    print (dx_t[:, -1]@dx_t[:, -1] - dx_adj_t@dx0)