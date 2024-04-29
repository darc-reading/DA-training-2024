import numpy as np
from scipy.optimize import fsolve

from tools.L96_model import lorenz96
from tools.cov import minv, msq
from tools.enkf import evolvemembers, enkfs, getlocmat, getObsForLocalDomain


def L4DEnVar(x0, t, period_obs, anawin, ne, y_traj, H, B, R, F,
    rho, doLoc, lam=None, loctype=None, maxmodes_in=None):
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
    F : float
        the forcing used in the L96 model when cycling
    rho : ndarray
        inflation for P.  Notice we multiply (1+rho)*Xpert
        or P*(1+rho)^2.
    doLoc : bool
        switch for localisation
    lam : int
        the localization radius in gridpoint units.  If None,
        it means no localization.
    loctype : str
        a string indicating the type of localization: 'GC'
        to use the Gaspari-Cohn function, 'cutoff' for a sharp cutoff
    maxmodes_in : int
        number of modes in the localisation matrix for 4DEnVar localisation

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
    ny = len(H)   # number of observations
    if type(F) is float:
        # this is used to acommodate the evolvemembers function
        # because it assumes an ensemble of parameters
        Fp = F*np.ones((1, ne))
    # Model's timestep
    deltat = t[1] - t[0]

    # Number of 4D-Var cycles that will be done here
    anal_nt = period_obs*anawin
    anal_dt = anal_nt*deltat
    ncycles = int(float(nt) / float(anal_nt))

    # Precreate the arrays for background and analysis
    np.random.seed(0)

    FreeEns = np.zeros((nx, ne, nt))
    VarEns = np.zeros((nx, ne, nt))
    VarEns_ = np.zeros((nx, nt))

    x_b = np.empty((nx, nt))
    x_b.fill(np.nan)
    x_a = np.empty((nx, nt))
    x_a.fill(np.nan)

    # Initial Condition for First Guess of First Window
    # initial guess for the variational method
    Bsq = msq(B)
    xold = x0 + Bsq@np.random.randn(nx)
    # initial guess for the ensemble
    xoldens = np.zeros((nx, ne))
    for m in range(ne):
        xoldens[:,m] = x0 + Bsq@np.random.randn(nx)
    xaoldens = xoldens.copy()

    if doLoc:
        Lxx = getlocmat(nx, nx, np.eye(nx), lam, loctype)
        locmatrix = getlocmat(nx, ny, H, lam, loctype)
        localDomainObsMask = getObsForLocalDomain(nx, lam, H)
        # number of modes in localisation matrix
        maxmodes = 9 if maxmodes_in == None else maxmodes_in
        # truncate the localisation matrix
        U, s, Vh = np.linalg.svd(Lxx)
        s = s[:maxmodes]
        U = U[:,:maxmodes]

        # get the square root of the localisation matrix
        s_sq = np.sqrt(s)
        Lxsq = (U*s_sq[..., None, :])@Vh[:maxmodes]
        # get the square root of the localisation matrix in obs. space
        Lysq = H@Lxsq
    else:
        localDomainObsMask = None
        locmatrix = None


    # pre-compute certain matrix for efficiency
    invR = minv(R)
    nouterloops = 1

    for cycle in range(ncycles):
        # Extract observations for this cycle
        yaux = y_traj[:, cycle * anawin + 1 : (cycle + 1) * anawin + 1]

        # do LETKF
        for it in range(anawin):
            start = cycle*anal_nt + it*period_obs
            end = cycle*anal_nt + (it+1)*period_obs
            xnew = evolvemembers(xaoldens, deltat, period_obs, lorenz96, Fp)
            VarEns_[..., start:end] = np.mean(xnew[..., :period_obs], axis=1)
            VarEns[..., start:end] = xnew[..., :period_obs]

            Xa, rho = enkfs(xnew[..., period_obs], yaux[:, it],
                            H, R, rho, 'ETKF',
                            localDomainObsMask, locmatrix, False)
            
            xaoldens = Xa.copy()
            VarEns_[..., end] = np.mean(Xa, axis=1)
            VarEns[..., end] = Xa
        

        start = anal_nt*cycle
        end = anal_nt*(cycle+1)
        # Free ensemble
        FreeEns[..., start:end+1] = evolvemembers(xoldens, deltat, anal_nt, lorenz96, Fp)
        X = FreeEns[..., start:end+1] - np.mean(FreeEns[..., start:end+1], axis=1, keepdims=True)
        X = X/np.sqrt(ne - 1)
        # 4DENVAR
        x_b[:, start:end+1] = lorenz96(xold, anal_dt, deltat, 0, F)
        if doLoc:
            xa0 = one4denvarLoc(x_b[:, start:end+1],
                                anawin, yaux, H, X, invR, Lxsq, Lysq, period_obs, nouterloops)
        else:
            xa0 = one4denvar(x_b[:, start:end+1],
                            anawin, yaux, H, X, invR, period_obs, nouterloops)

        x_a[:, start:end+1] = lorenz96(xa0, anal_dt, deltat, 0, F)

        # create new initial conditions for 4denvar
        xold = x_a[:, end]
        # create new initial conditions for ETKF and free ensemble run 
        # the ensemble mean uses the 4DEnVar analysis and the perturbations are generated from the ETKF
        xoldens = xold[:, None] + (xnew[..., period_obs] - np.mean(xnew[..., period_obs], axis=1, keepdims=True))

    return x_a, x_b, VarEns, VarEns_, FreeEns


def one4denvarLoc(xb, anawin, yaux, H, X, invR, Lxsq, Lysq, period_obs, nouterloops):
    """The 4DEnVar algorithm with localisation for one assimilation window.
    todo
    ----
    Need to add outer vars if needed
    """
    d = np.zeros_like(yaux)
    ny = len(yaux)
    nx, ne, _ = X.shape
    maxmodes = Lysq.shape[-1]
    Y = np.zeros((ny, ne, anawin), order='F')

    xg0 = np.squeeze(xb[:, 0])

    outerloops = nouterloops

    # if outerloops>1:
    #     P = X[0,:,:]@X[0,:,:].T
    #     P = P * Lxx
    #     gamma, U = np.linalg.eig(P)
    #     gamma = np.real(gamma)
    #     gamma = gamma.clip(min=0)
    #     ind = gamma>0;
    #     gamma[ind] = gamma[ind]**(-1)
    #     gamma = np.diag(gamma)
    #     invP = U@(gamma@U.T)

    for jotl in range(outerloops):
        # initial guess
        xg0 = np.squeeze(xb[:, 0])
        # get innovation vector and HX
        for j in range(anawin):
            d[:, j] = yaux[:, j] - H@xb[:, period_obs*(j+1)]
            # just because H is linear we can do the following
            Y[..., j] = H@X[..., period_obs*(j+1)]

        # The gradient
        def gradJ(deltav0):
            if deltav0.ndim == 1:
                deltav = deltav0.reshape((ne, maxmodes))
            else:
                deltav = deltav0
            # The background term
            gJ = deltav.ravel().copy()

            # if jotl>=1:
            #     # this is the gradient of cost function
            #     # (X*(D^\frac{1}{2}\delta x)) ^ T (L \circ B)^{-1} (X*(D^\frac{1}{2}\delta x))
            #     # this seems quite unnessary? perhaps just gJ = xg0 - xb[:, -1] is fine?
            #     # why do we want increment to be xg0 - xb[:, -1]?
            #     # Or, is xb[:, -1] a bug that it was supposed to be xb[:, 0]?
            #     z_aux = invP@(xg0 - xb[:, -1])
            #     for m in range(ne):
            #         gJ[m*maxmodes:(m+1)*maxmodes] += Lxsq.T@(Xfree_pert[:, m, 0]*z_aux)

            # The observation error term, evaluated at different times
            for i in range(anawin):
                z = np.zeros(ny)
                for m in range(ne):
                    z += Y[:, m, i] * (Lysq@deltav[m])

                z = invR@(d[:, i] - z)
                gJo = np.zeros((ne, maxmodes))
                for m in range(ne):
                    gJo[m] = -Lysq.T@(Y[:, m, i] * z)

                gJ += gJo.ravel()

            return gJ.ravel()

        v0 = np.zeros((ne, maxmodes))
        v = fsolve(gradJ, v0, xtol=1e-6, maxfev=20)
        v = v.reshape((ne, maxmodes))
        deltax = np.zeros(nx)
        for m in range(ne):
            deltax += X[:, m, 0] * (Lxsq@v[m])
        xa = xg0 + deltax

        # if jotl<outerloops-1:
        #     xb,seed_b = l96num(x,taux,xa,noiseswitch,Qsq,seed_b)

    return xa


def one4denvar(xb, anawin, yaux, H, X, invR, period_obs, nouterloops):
    """The 4DEnVar algorithm without localisation for one assimilation window.
    todo
    ----
    Need to add outer vars if needed
    """
    d = np.zeros_like(yaux)
    ny = len(yaux)
    nx, ne, _ = X.shape
    Y = np.zeros((ny, ne, anawin), order='F')

    xg0 = np.squeeze(xb[:, 0])

    outerloops = 1

    # if outerloops>1:
    #     Ux,sx,VTx = np.linalg.svd(np.squeeze(Xfree_pert[0,:,:]),full_matrices=False)

    for jotl in range(outerloops):
        # initial guess
        xg0 = np.squeeze(xb[0,:])
        # innovation vector and HX
        for j in range(anawin):
            d[:, j] = yaux[:, j] - H@xb[:, period_obs*(j+1)]
            # just because H is linear we can do the following
            Y[..., j] = H@X[..., period_obs*(j+1)]
        # The gradient
        def gradJ(deltav):
            # The background term
            gJ = deltav.copy()
            # if jotl >= 1:
            #     aux = np.dot(Ux.T,(xg0-xb[-1,:]))
            #     aux = np.dot(np.diag(sx**(-1)), aux)
            #     aux = np.dot(VTx.T,aux)
            #     gJ += aux

            # The observation error term, evaluated at different times
            for i in range(anawin):
                gJ += - Y[..., i].T@invR@(d[:, i] - (Y[..., i]@deltav))
            return gJ.flatten()

        v0 = np.zeros(ne)
        v = fsolve(gradJ, v0, xtol=1e-6, maxfev=20)
        xa = xb[:, 0] + X[..., 0]@v

        # if jotl<outerloops-1:
        #     xb,seed_b = l96num(x,taux,xa,noiseswitch,Qsq,seed_b)

    return xa


def backupCodeForWC4DVar():
    pass
    # if scwc=='wc' and compute_qt==1:
    #     ensrun_i = np.empty((winlen+1,nx,M)); ensrun_i.fill(np.nan)
    #     ensrun_i[0,:,:] = ensrun[0,:,:]
    #     xoldens_i = ensrun_i[0,:,:]

    # xa0 = one4denvar(taux,x,period_obs,obsperwin,yaux,loc_obs,H,invR,M,nx,nx_obs,\
    #       noiseswitch,invQ,Qsq,xb_new,seed_b,Xfree_pert,Xfreei_pert,locenvar,Lxsq,Lysq,\
    #       Lxx,scwc,maxmodes,compute_qt)
    # if scwc=='sc':
    #     xa0 = np.reshape(xa0,(nx,1))
    #     xa_new = l96num(x,taux,xa0,noiseswitch,Qsq,seed_b)
    # if scwc=='wc':
    #     xa_new = xa0
    # for m in range(M):
    #     aux = Xa_e[-1,:,m] - xa_e[-1,:] + xa_new[-1,:]
    #     xoldens[:,m] = aux
    #     if scwc=='wc' and compute_qt==1:
    #         xoldens_i[:,m] = xoldens[:,m]
    # def gradJ(deltav):
    #         if scwc=='wc':
    #             if  locenvar==0:
    #                 v = np.reshape(deltav,(M*(1+obsperwin),)) # Fixing Annoyances aka fsolve
    #                 gJ = np.empty((M*(1+obsperwin),))
    #                 # the background and model error
    #                 dyt = np.empty((obsperwin,M))
    #                 for j in range(obsperwin):
    #                     aux = v[0*M:1*M] + v[(j+1)*M:(j+2)*M]
    #                     aux = d0[j,:] - np.dot(Ytt[j,:,:],aux)
    #                     aux = np.dot(invR,aux)
    #                     dyt[j,:] = np.dot(Ytt[j,:,:].T, aux)
    #                     del aux
    #                 del j

    #                 gJ[0*M:1*M] = v[0*M:1*M] - np.sum(dyt,0)

    #                 for j in range(obsperwin):
    #                     jobs = period_obs*(j+1)

    #                     if compute_qt==0:
    #                         invQt = invQ#/jobs
    #                         aux = np.dot(Xfree_pert[jobs,:,:], v[(j+1)*M:(j+2)*M])
    #                         aux = np.dot(Xfree_pert[jobs,:,:].T, np.dot(invQt,aux))

    #                     if compute_qt==1:
    #                         Deltaip = Xfreei_pert[jobs,:,:] - Xfree_pert[jobs,:,:]
    #                         auxinv = np.linalg.pinv(np.dot(Deltaip,Deltaip.T),rcond=1e-8)
    #                         aux = np.dot(Xfree_pert[jobs,:,:] , v[(j+1)*M:(j+2)*M])
    #                         aux = np.dot(auxinv,aux)
    #                         aux = np.dot(Xfree_pert[jobs,:,:].T,aux)

    #                     gJ[(j+1)*M:(j+2)*M] = aux - dyt[j,:]
    #                     del aux
    #                 del j


    #             # With localisation
    #             if  locenvar==1:
    #                 v = np.reshape(deltav,(maxmodes*M*(1+obsperwin),)) # Fixing Annoyances aka fsolve
    #                 gJ = np.empty((maxmodes*M*(1+obsperwin),))

    #                 # the background
    #                 gJ[0*M*maxmodes:1*M*maxmodes] = v[0*maxmodes*M:1*maxmodes*M]

    #                 # model error
    #                 for j in range(obsperwin):
    #                     jobs = period_obs*(j+1)
    #                     mod_error = np.zeros((maxmodes*M))
    #                     z = np.zeros((nx))
    #                     for m in range(M):
    #                         aux = v[(j+1)*maxmodes*M:(j+2)*maxmodes*M]
    #                         aux = np.dot(Lxsq,aux[m*maxmodes:(m+1)*maxmodes])
    #                         aux = Xfree_pert[jobs,:,m] * aux
    #                         z = z + aux

    #                         del aux
    #                     del m
    #                     z = np.dot(invQ,z)

    #                     for m in range(M):
    #                         aux = Xfree_pert[jobs,:,m] * z
    #                         mod_error[maxmodes*m:maxmodes*(m+1)] = np.dot(Lxsq.T,aux)
    #                         del aux
    #                     del m
    #                     gJ[(j+1)*maxmodes*M:(j+2)*maxmodes*M] = mod_error
    #                     del mod_error
    #                 del j, z

    #                 # observations
    #                 for j in range(obsperwin):
    #                     z = np.zeros((nx_obs))

    #                     for m in range(M):
    #                         aux0 = v[0*maxmodes*M:1*maxmodes*M]
    #                         aux0 = aux0[m*maxmodes:(m+1)*maxmodes]
    #                         auxt = v[(j+1)*maxmodes*M:(j+2)*maxmodes*M]
    #                         auxt = auxt[m*maxmodes:(m+1)*maxmodes]
    #                         aux = aux0+auxt
    #                         del aux0, auxt
    #                         aux = np.dot(Lysq,aux)
    #                         aux = Ytt[j,:,m]*aux
    #                         z = z + aux
    #                         del aux
    #                     del m

    #                     z = np.dot(invR,d0[j,:]-z)
    #                     incr = np.zeros((maxmodes*M))

    #                     for m in range(M):
    #                         aux = Ytt[j,:,m] * z
    #                         obs_error = np.dot(Lysq.T,aux)
    #                         incr[m*maxmodes:(m+1)*maxmodes] = -obs_error
    #                         del obs_error
    #                     del m

    #                     gJ[0*maxmodes*M:1*maxmodes*M] = gJ[0*maxmodes*M:1*maxmodes*M] + incr
    #                     gJ[(j+1)*maxmodes*M:(j+2)*maxmodes*M] = gJ[(j+1)*maxmodes*M:(j+2)*maxmodes*M] + incr

    #                     del z, incr
    # #-------------------------------------
    #     if scwc=='wc':
    #         if  locenvar==0:
    #             v0 = np.zeros((M*(1+obsperwin),))
    #             v = fsolve(gradJ,v0,xtol=1e-6,maxfev=10)
    #             xa0 = xb[0,:] + np.dot(Xfree_pert[0,:,:],v[0*M:1*M])
    #             xa,seed_a = l96num(x,taux,xa0,0,Qsq)
    #             for jsteps in range(obsperwin):
    #                 jobs = period_obs * (jsteps+1)
    #                 xa[jobs,:] = xb[jobs,:] + np.dot(Xfree_pert[0,:,:],\
    #                              v[(jsteps+1)*M:(jsteps+2)*M])
    #             del jsteps, jobs

    #         if  locenvar==1:
    #             v0 = np.zeros((maxmodes*M*(1+obsperwin),))
    #             v = fsolve(gradJ,v0,xtol=1e-6,maxfev=10)
    #             deltax0 = np.zeros((nx,))
    #             auxv0 = v[0*maxmodes*M:1*maxmodes*M]
    #             for m in range(M):
    #                 aux = np.dot(Lxsq,auxv0[m*maxmodes:(m+1)*maxmodes])
    #                 aux = Xfree_pert[0,:,m] * aux
    #                 deltax0 = deltax0 + aux
    #                 del aux
    #             del m, auxv0
    #             xa0 = xb[0,:] + deltax0
    #             xa,seed_a = l96num(x,taux,xa0,0,Qsq)

    #             for jsteps in range(obsperwin):
    #                 deltaxt = np.zeros((nx,))
    #                 auxvt = v[(jsteps+1)*maxmodes*M:(jsteps+2)*maxmodes*M]
    #                 jobs = (1+jsteps) * period_obs

    #                 for m in range(M):
    #                     aux = np.dot(Lxsq,auxvt[m*maxmodes:(m+1)*maxmodes])
    #                     aux = Xfree_pert[jobs,:,m] * aux
    #                     deltaxt = deltaxt + aux
    #                     del aux
    #                 del m, auxvt
    #                 xa[jsteps,:] = xb[jsteps,:] + deltaxt
    #                 del jobs

    #             del jsteps

    #         xb = xa