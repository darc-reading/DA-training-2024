import numpy as np
import scipy
import scipy.optimize
from tools.L96_model import lorenz96, Lorenz96_TL_simple, Lorenz96_TL1_simple_adj
from tools.plots import plot_log_test


def var3dL96(x0, t, tobs, y, H, B, R, F, gradtest=False):
    """Data assimilation routine for Lorenz 1996 using 3D-Var

    3D-Var cycling for L96 only (and initial gradient test)

    The function assumes that tobs[0] == t[0]

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
    B : ndarray
        the background error covariance matrix. shape: nx, nx
    R : ndarray
        the observation error covariance matrix. shape: ny, ny
    F : float
        the forcing used in the L96 model when cycling
    gradtest : bool
        set to True to perform a gradient test

    Returns
    -------
    x_b : ndarray
        the background trajectory. shape: nx, nt
    x_a : ndarray
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
    o2t = int(anal_dt / deltat)

    # Check that ob timestep is multiple of model timestep
    assert np.isclose(o2t, anal_dt / deltat), \
        "observation timestep must be a multiple of model timestep"

    # Make blank arrays for background and analysis trajectories
    xb_traj = np.empty([nx, nt], order='F')
    xb_traj.fill(np.nan)
    xa_traj = np.empty([nx, nt], order='F')
    xa_traj.fill(np.nan)

    # Find the inverse of B and R
    invB = scipy.linalg.pinv(B)
    invR = scipy.linalg.pinv(R)

    # The first background (initial guess) will be truth + random pert
    # The variance of the random perts is based on the average variance in B
    # This is not the best way of doing this in practice as the added error
    # doesn't respect the correlation structure of backgrond errors
    # But this is quick and easy
    # np.random.seed(10)
    xb1 = x0 + np.sqrt(np.mean(np.diag(B))) * np.random.randn(nx)

    # This is the background for the first instant
    xb_traj[:, 0] = xb1[:]

    # -------------------------------------------------------
    # Do a gradient test
    if gradtest:
        # Calculate a sample gradient at a fixed sample point (choose truth for convenience)
        grad = one3dvar(xb1, y[:, 0], H, invB, invR, purpose="grad", state=x0)
        # Find the norm of this gradient
        norm = np.sqrt(np.sum(grad[:] * grad[:]))
        # unit is the unit vector that points in the direction of the gradient
        unit = grad / norm
        # Calculate value of cost function at fixed sample point
        Jb0Jo0J0 = one3dvar(xb1, y[:, 0], H, invB, invR, purpose="cost", state=x0)
        # alpha is the multiple of the unit vector
        logalpha = np.arange(-10., 1.)
        alpha_store = 10.0 ** logalpha
        # Calculate value of cost function at perturbed sample point
        JbJoJ = np.array([
            one3dvar(
                xb1, y[:, 0], H, invB, invR, purpose="cost", state=x0 + alpha * unit
            ) for alpha in alpha_store
            ])
        PhiMinusOne = np.abs((JbJoJ[:, 2] - Jb0Jo0J0[2]) / (alpha_store * norm) - 1.0)
        plot_log_test(
            alpha_store, PhiMinusOne, "alpha", "Phi - 1", "L96 3D-Var Gradient Test"
        )

    # -------------------------------------------------------
    # The following cycle contains evolution and assimilation
    for cycle in range(len(tobs)):
        print(" > Running 3D-Var cycle number ", cycle + 1, "of ", len(tobs))

        # This cycle corresponds to the following start and end times
        start = np.where(np.isclose(t, tobs[cycle], rtol=0., atol=1e-7))[0][0]
        # start = tindex = find_in_array(t, tobs[cycle], 0.0000001)
        end = start + o2t

        # Do a background forecast between now and the next cycle
        # (This is just as a reference)
        x_traj = lorenz96(xb_traj[:, start], anal_dt, deltat, 0, F)
        # Store this for reference
        if cycle < len(tobs) - 1:
            xb_traj[:, start:end] = x_traj[:, 0:o2t]

        # Extract observations for this cycle
        yaux = y[:, cycle]

        # Find the 3D-Var analysis
        xa_aux = one3dvar(xb_traj[:, start], yaux, H, invB, invR)

        # Do an analysis forecast between now and the next cycle
        # (This is needed to make the background for the next cycle)
        x_traj = lorenz96(xa_aux, anal_dt, deltat, 0, F)
        # Store in the analysis array
        if cycle < len(tobs) - 1:
            xa_traj[:, start : end + 1] = x_traj[:, 0 : o2t + 1]

        # The last analysis timestep from this cycle is the background for the next cycle
        if cycle < len(tobs) - 1:
            xb_traj[:, end] = xa_traj[:, end]

    return xb_traj, xa_traj


def one3dvar(xb, y, H, invB, invR, purpose="DA", state=0):
    """The main 3D-Var algorithm (one cycle)

    Arguments
    ---------
    xb : ndarray
        background state at this time. shape: nx
    y : ndarray
        the observations at this time. shape: ny
    H : ndarray
        the observation operator. shape: ny, nx
    invB : ndarray
        the inverse of the background error cov matrix. shape: nx, nx
    invR : ndarray
        the inverse of the observation error cov matrix. shape: ny, ny
    purpose : str
        'DA' run one cycle of 3D-Var
        'grad' : compute the gradient of the cost function at state
        'cost : compute the cost function at state
    state : int 
        the state (only used for 'grad' and 'cost')

    Returns
    -------
    result : ndarray or list
        'DA' returns the analysis state at this time (ndarray)
        'grad' returns the gradient of the cost function (ndarray)
        'cost' returns Jb, Jo, J as a list
    """
    y_vec = np.array([y]).T  # array -> column vector
    xb_vec = np.array([xb]).T  # array -> column vector

    def var3dCostfn(x):
        """The 3D-Var cost function
        """
        x_vec = np.array([x]).T
        Jback = (x_vec - xb_vec).T @ invB @ (x_vec - xb_vec) / 2.0
        Jback = Jback[0, 0]
        mod_y_vec = H @ x_vec
        Jobs = (y_vec - mod_y_vec).T @ invR @ (y_vec - mod_y_vec) / 2.0
        Jobs = Jobs[0, 0]
        J = Jback + Jobs
        return [Jback, Jobs, J]

    def gradJ(x):
        """The gradient of the 3D-Var cost function with respect to x
        """
        # This is simple for 3D-Var, x is the state vector
        # Note other variables inside one3dvar are visible here
        x_vec = np.array([x]).T
        # The background term
        gJb = invB @ (x_vec - xb_vec)
        # The observation term
        gJo = -H.T @ invR @ (y_vec - H @ x_vec)
        gJ = gJb + gJo
        return gJ.flatten()

    # ----- The executed code of this subroutine continues here -----

    if purpose == "DA":
        # Do data assimilation
        # Call the descent algorithm.
        # gradJ is the name of the function used to find the gradient of the cost fn
        # result is the analysis
        result = scipy.optimize.fsolve(gradJ, xb, xtol=1e-4)
    elif purpose == "grad":
        # Return the gradient of the cost function
        result = gradJ(state)
    elif purpose == "cost":
        # Return the components of the cost function
        result = var3dCostfn(state)

    return result


def var4dL96(x0, t, tobs, period_obs, anawin, y_traj, H, B, R, F, TL_type, gradtest=False):
    """4D-Var cycling data assimilation routine for Lorenz 1996

    Arguments
    ---------
    x0 : ndarray
        the real initial conditions (truth). shape: nx, nt
    t : ndarray
        time array of model time steps (all cycles). shape: nt
    tobs : ndarray
        the time array of observation times (all cycles). shape: nobt
    period_obs : int
        the number of model timesteps between observations.
    anawin : int
        the number of observation periods between 4D-Var cycles
    y_traj : ndarray
        the observations (all cycles). shape: ny, nt
    H : ndarray
        observation matrix. shape: ny, nx
    B : ndarray
        the background error covariance matrix. shape: nx, nx
    R : ndarray
        the observation error covariance matrix. shape: ny, ny
    F : float
        the forcing used in the L96 model when cycling
    TL_type : str
        the tangent model solution type ('Euler' or 'RK4')
    gradtest : bool
        set to True to perform a gradient test

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

    # Number of timesteps between analyses
    anal_nt = period_obs * anawin
    anal_dt = anal_nt * deltat

    # Number of 4D-Var cycles that will be done here
    ncycles = int(float(nt) / float(anal_nt))

    # Make blank arrays for background and analysis trajectories
    xb_traj = np.empty([nx, nt])
    xb_traj.fill(np.nan)
    xa_traj = np.empty([nx, nt])
    xa_traj.fill(np.nan)

    # Find the inverse of B and R
    invB = scipy.linalg.pinv(B)
    invR = scipy.linalg.pinv(R)

    # The first background will be truth + random pert
    # Base the variance of the random perts on the average variance in B
    # This is not the best way of doing this in practice as the added error
    # doesn't respect the correlation structure of background errors
    # But this is quick and easy
    xb1 = x0 + np.sqrt(np.mean(np.diag(B))) * np.random.randn(nx)

    # This is the background for the first instant
    xb_traj[:, 0] = xb1[:]

    # -------------------------------------------------------
    # Do a gradient test
    if gradtest:
        # Calculate a sample gradient at a fixed sample point (choose truth for convenience)

        # Do a background forecast between now and the next cycle
        xaux_traj = lorenz96(xb_traj[:, 0], anal_dt, deltat, 0, F)

        # Extract the timesteps for this cycle
        t_aux = t[0 : anal_nt + 1]

        # Extract observations for this cycle
        yaux_traj = y_traj[:, 0:anawin]

        # Extract observation times for this cycle
        tobs_aux = tobs[0:anawin]

        # Compute the gradient
        grad = one4dvar(
            t_aux,
            xaux_traj,
            tobs_aux,
            yaux_traj,
            H,
            invB,
            invR,
            TL_type,
            purpose="grad",
            state=x0,
        )
        # Find the norm of this gradient
        norm = np.sqrt(np.sum(grad[:] * grad[:]))
        # unit is the unit vector that points in the direction of the gradient
        unit = grad / norm
        # Calculate value of cost function at fixed sample point
        Jb0Jo0J0 = one4dvar(
            t_aux,
            xaux_traj,
            tobs_aux,
            yaux_traj,
            H,
            invB,
            invR,
            TL_type,
            purpose="cost",
            state=x0,
        )
        # alpha is the multiple of the unit vector
        logalpha = np.arange(-10., 1.)
        alpha_store = 10.0 ** logalpha
        # Calculate value of cost function at perturbed sample point
        JbJoJ = np.array([
                            one4dvar(
                                t_aux,
                                xaux_traj,
                                tobs_aux,
                                yaux_traj,
                                H,
                                invB,
                                invR,
                                TL_type,
                                purpose="cost",
                                state=x0 + alpha * unit,
                            ) for alpha in alpha_store
                          ])
        PhiMinusOne = np.abs((JbJoJ[:, 2] - Jb0Jo0J0[2]) / (alpha_store * norm) - 1.0)
        plot_log_test(
            alpha_store, PhiMinusOne, "alpha", "Phi - 1", "L96 4D-Var Gradient Test"
        )

    # -------------------------------------------------------
    # The following cycle contains evolution and assimilation
    # Loop around 4D-Var cycles
    for cycle in range(ncycles):
        print(" > Running 4D-Var cycle number ", cycle + 1, "of ", ncycles)

        # This cycle corresponds to the following start and end times
        start = cycle * anal_nt
        end = start + anal_nt

        # Do a background forecast between now and the next cycle
        xaux_traj = lorenz96(xb_traj[:, start], anal_dt, deltat, 0, F)
        # Store this for reference
        xb_traj[:, start:end] = xaux_traj[:, 0:anal_nt]

        # Extract the timesteps for this cycle
        t_aux = t[start : end + 1]

        # Extract observations for this cycle
        yaux_traj = y_traj[:, cycle * anawin : (cycle + 1) * anawin]

        # Extract observation times for this cycle
        tobs_aux = tobs[cycle * anawin : (cycle + 1) * anawin]

        # Find the 4D-Var analysis
        xa = one4dvar(t_aux, xaux_traj, tobs_aux, yaux_traj, H, invB, invR, TL_type)

        # Do an analysis forecast between now and the next cycle
        # (This is needed to make the background for the next cycle)
        xaux_traj = lorenz96(xa, anal_dt, deltat, 0, F)
        # Store this for reference
        xa_traj[:, start : end + 1] = xaux_traj[:, 0 : anal_nt + 1]

        # The last analysis timestep from this cycle is the background for the next cycle
        xb_traj[:, end] = xa_traj[:, end]

    return xb_traj, xa_traj


def one4dvar(t, xb_traj, tobs, y_traj, H, invB, invR, TL_type, purpose="DA", state=0):
    """The main 4D-Var algorithm (one cycle, linearised about xb_traj)

    Arguments
    ---------
    t : ndarray
        the array of model time steps. shape: nt
    xb_traj : ndarray
        the background state at the above times. shape: nx, nt
    tobs : ndarray
        the array of observation times. shape: nobt
    y_traj : ndarray
        the observations at the above times. shape: ny, nobt
    H : ndarray
        the observation operator. shape: ny, nx
    invB : ndarray
        the inverse of the background error cov matrix. shape: nx, nx
    invR : ndarray
        the inverse of the observation error cov matrix. shape: ny, ny
    TL_type : 
        the tangent model solution type ('Euler' or 'RK4')
    purpose : str
        'DA' run one cycle of 3D-Var
        'grad' : compute the gradient of the cost function at state
        'cost : compute the cost function at state
    state : int 
        the state (only used for 'grad' and 'cost')

    Returns
    -------
    result : ndarray or list
        'DA' returns the analysis state at this time (ndarray)
        'grad' returns the gradient of the cost function (ndarray)
        'cost' returns Jb, Jo, J as a list
    """

    # All the above pertain to this cycle only
    nx = xb_traj.shape[0]  # Number of state variables
    nt = xb_traj.shape[1] - 1  # Number of model time steps
    ny = y_traj.shape[0]  # Number of observations
    nobt = y_traj.shape[1]  # Number of observation times
    ym_traj = np.zeros([ny, nobt])  # Set-up of model observations
    deltat = t[1] - t[0]  # Model timestep

    # ----- The 4D-Var cost function -----
    def var4dCostfn(xp0):
        # This is incremental 4D-Var, xp0 is the increment at t=0
        # Note other variables inside one4dvar are visible here

        # The background term
        # Act with Binv on xp0
        invBxp0 = invB@xp0
        Jb = xp0@invBxp0 / 2.0

        # The observation term
        # Array to hold observation space data at a particular time
        diffRm1 = np.zeros(ny)
        Jo = 0.0
        # Run the forward trajectory of the incremental state (linearised about xb_traj)
        xp_traj = Lorenz96_TL_simple(xb_traj, xp0, deltat, TL_type)
        # Compute the model observations and their increments
        ymp_traj = np.zeros([ny, nobt])
        for time in range(nobt):
            # This observation is at tobs[time]; what index in t does this correspond to?
            tindex = np.isclose(t, tobs[time], rtol=0., atol=1e-7)
            # tindex = find_in_array(t, tobs[time], 0.0000001)
            # Compute the perturbation in model observations at this time
            ymp_traj[:, time] = H@xp_traj[:, tindex]
            # Compute the observation differences at this time
            diff = y_traj[:, time] - ym_traj[:, time] - ymp_traj[:, time]
            # Act with inverse of ob error cov matrix
            Rm1diff = invR@diff
            # Compute the contribution to the observation term
            Jo += diff@Rm1diff / 2.0

        # The total cost function
        J = Jb + Jo

        return [Jb, Jo, J]

    # ----- The gradient function - this uses the adjoint method -----
    def gradJ(xp0):
        # This is incremental 4D-Var, xp0 is the increment at t=0
        # Note other variables inside one4dvar are visible here

        # Run the forward trajectory of the incremental state (linearised about xb_traj)
        xp_traj = Lorenz96_TL_simple(xb_traj, xp0, deltat, TL_type)

        # Compute the model observations and their increments
        ymp_traj = np.zeros([ny, nobt])
        for time in range(nobt):
            # This observation is at tobs[time]; what index in t does this correspond to?
            tindex = np.isclose(t, tobs[time], rtol=0., atol=1e-7)
            # tindex = find_in_array(t, tobs[time], 0.0000001)
            # Compute the perturbation in model observations at this time
            ymp_traj[:, time] = H@xp_traj[:, tindex]

        # Integrate the adjoint equations backwards in time
        # Initialise adjoint variable to 0
        lam = np.zeros([nx])
        for time in range(nt - 1, -1, -1):
            # Are there observations at this time?
            obindex = np.where(np.isclose(tobs, t[time], rtol=0., atol=1e-7))[0]
            obindex = obindex[0] if len(obindex) > 0 else -1
            # obindex = find_in_array(tobs, t[time], 0.0000001)
            if obindex > -1:
                # Compute difference between obs and current model estimate of these obs
                diff = y_traj[:, obindex] - ym_traj[:, obindex] - ymp_traj[:, obindex]
                # Act with inverse of ob error cov matrix
                diffRm1 = invR@diff
                # Act with adjoint of observation operator
                diffRm1HT = H.T@diffRm1
            else:
                diffRm1HT = np.zeros([nx])

            # Act with adjoint of model, and add on the contribution from obs at this time
            inc = Lorenz96_TL1_simple_adj(xb_traj[:, time], lam, deltat, TL_type)
            lam = diffRm1HT + inc

        # Find the gradient at t=0
        gJo = -1.0 * lam
        gJb = invB@xp0
        gJ = gJb + gJo

        # print ('Norm of tot, bg, ob gradient ', np.sqrt(np.sum( gJ[:] * gJ[:])),    \
        #                                        np.sqrt(np.sum( gJb[:] * gJb[:] )), \
        #                                        np.sqrt(np.sum( gJo[:] * gJo[:] )))
        return gJ

    # ----- The executed code of this subroutine continues here -----

    # The forward trajectory of the reference state has already been done (xb_traj)
    # Compute the model observations of the reference trajectory
    for time in range(nobt):
        # This observation is at tobs[time]; what index in t does this correspond to?
        tindex = np.isclose(t, tobs[time], rtol=0., atol=1e-7)
        # tindex = find_in_array(t, tobs[time], 0.0000001)
        ym_traj[:, time] = H @ xb_traj[:, tindex]

    if purpose == "DA":
        # Do data assimilation
        # Call the descent algorithm.
        # gradJ is the name of the function used to find the gradient of the cost fn
        # result is the analysis
        result = xb_traj[:, 0] + scipy.optimize.fsolve(gradJ, np.zeros([nx]), xtol=1e-4)
    elif purpose == "grad":
        # Return the gradient of the cost function
        result = gradJ(state)
    elif purpose == "cost":
        # Return the components of the cost function
        result = var4dCostfn(state)

    return result
