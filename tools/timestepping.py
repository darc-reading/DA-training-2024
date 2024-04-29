import numpy as np


def euler(Xold, deltat, f):
    """Forward Euler

    Parameters
    ----------
    Xold : ndarray
        Model state at current time step
    deltat : float
       Time step interval
    f : func
       d/dx (time tendency) of the model.
       A function of the form f = f(Xold)

    Returns
    -------
     : ndarray
        increment to model state due to model equations
    """
    return deltat * f(Xold)


def eulerTLM(Xold, dx, deltat, ftlm):
    """Forward Euler for TLM 

    Parameters
    ----------
    Xold : ndarray
        Model state at current time step
    dx : ndarray
        Model perturbation at current time step
    deltat : float
       Time step interval
    ftlm : func
       d/dx (time tendency) of the model.
       A function of the form f = f(Xold)

    Returns
    -------
     : ndarray
        time increment to model perturbation
    """
    return deltat * ftlm(Xold, dx)


def eulerADJ(Xold, dx, deltat, fadj):
    """Forward Euler for adjoint

    Parameters
    ----------
    Xold : ndarray
        Model state at current time step
    dx : ndarray
        Model perturbation at current time step
    deltat : float
       Time step interval
    ftlm : func
       d/dx (time tendency) of the model.
       A function of the form f = f(Xold)

    Returns
    -------
     : ndarray
        time increment to model perturbation
    """
    return deltat * fadj(Xold, dx)


def rk4(Xold, deltat, f):
    """Fourth order Runge-Kutta solution

    Parameters
    ----------
    Xold : ndarray
        Model state at current time step
    deltat : float
       Time step interval
    f : func
       d/dx (time tendency) of the model.
       A function of the form f = f(Xold)

    Returns
    -------
    delta : ndarray
        increment to model state due to model equations
    """
    k1 = f(Xold)
    k2 = f(Xold + 1 / 2.0 * deltat * k1)
    k3 = f(Xold + 1 / 2.0 * deltat * k2)
    k4 = f(Xold + deltat * k3)
    delta = deltat * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return delta


def rk4TLM(Xold, dx, deltat, ftlm, f=None):
    """Fourth order Runge-Kutta solution
       along with tagent linear model

    Parameters
    ----------
    Xold : ndarray
        Model state at current time step
    dx : ndarray
        Model perturbation at current time step
    deltat : float
       Time step interval
    ftlm : func
       d/dx (time tendency) of the model.
       A function of the form f = f(Xold)
    f : func
       d/dx (time tendency) of the model.
       A function of the form f = f(Xold)

    Returns
    -------
    delta : ndarray
        time increment to model state
    delta_dx : ndarray
        time increment to model perturbation
    """
    if f is not None:
        k1 = f(Xold)*deltat
        k2 = f(Xold + 0.5*k1)*deltat
        k3 = f(Xold + 0.5*k2)*deltat
        k4 = f(Xold + k3)*deltat
    else:
        k1, k2, k3, k4 = 0, 0, 0, 0

    # k1_tlm = ftlm(Xold, dx)*deltat
    # k2_tlm = ftlm(Xold + 0.5*k1, dx + 0.5*k1_tlm)*deltat
    # k3_tlm = ftlm(Xold + 0.5*k2, dx + 0.5*k2_tlm)*deltat
    # k4_tlm = ftlm(Xold + k3, dx + k3_tlm)*deltat

    # for long time integration, this differs from the above formulation
    k1_tlm = (ftlm(Xold)@dx)*deltat
    k2_tlm = (ftlm(Xold + 0.5*k1)@\
                     (dx + 0.5*k1_tlm))*deltat
    k3_tlm = (ftlm(Xold + 0.5*k2)@\
                     (dx + 0.5*k2_tlm))*deltat
    k4_tlm = (ftlm(Xold + k3)@(dx + k3_tlm))*deltat
    delta_dx = (k1_tlm + 2.0 * k2_tlm + 2.0 * k3_tlm + k4_tlm) / 6.0
    delta = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return delta, delta_dx


def rk4ADJ(Xold, dx, deltat, fadj, f=None):
    """Fourth order Runge-Kutta solution
       for the adjoint model

    Parameters
    ----------
    Xold : ndarray
        Model state at current time step
    dx : ndarray
        Model perturbation at current time step
    deltat : float
       Time step interval
    f : func
       d/dx (time tendency) of the model.
       A function of the form f = f(Xold)

    fadj : func
       Adjoint model

    Returns
    -------
    delta : ndarray
        time increment to model state due to model perturbations
    """
    if f is not None:
        k1 = f(Xold)*deltat
        k2 = f(Xold + 0.5*k1)*deltat
        k3 = f(Xold + 0.5*k2)*deltat
        k4 = f(Xold + k3)*deltat
    else:
        k1, k2, k3, k4 = 0, 0, 0, 0

    # k4_adj = fadj(Xold + k3, dx)*deltat
    # k3_adj = fadj(Xold + 0.5*k2, dx + 0.5*k4_adj)*deltat
    # k2_adj = fadj(Xold + 0.5*k1, dx + 0.5*k3_adj)*deltat
    # k1_adj = fadj(Xold, dx + k2_adj)*deltat

    k4_adj = fadj(Xold + k3)@(dx)*deltat
    k3_adj = fadj(Xold + 0.5*k2)@(dx + 0.5*k4_adj)*deltat
    k2_adj = fadj(Xold + 0.5*k1)@(dx + 0.5*k3_adj)*deltat
    k1_adj = fadj(Xold)@(dx + k2_adj)*deltat

    return (k1_adj + 2.*(k2_adj + k3_adj) + k4_adj)/6.
