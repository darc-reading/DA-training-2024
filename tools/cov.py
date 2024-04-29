import numpy as np
import scipy.linalg

from tools.L96_model import lorenz96

def getBcanadian(xt, diff_period, max_var, sample_size):
    """Canadian quick method to obtain the background error covariance
       from model run

    Parameters
    ----------
    xt : ndarry
        model trajectory. shape: (nx, nt)
    diff_period : int
      number of time steps to offset forecast differences
    max_var : float
      The background error variance
    sample_size : int
      Total time steps used for getting covariance matrix

    Returns
    -------
    B : ndarray
        The background covariance matrix
    Bcorr : ndarray
        The correlation matrix
    """
    _, total_steps = xt.shape
    assert total_steps >= sample_size, \
        f'model trajectory length {total_steps} must >= sample_size {sample_size}'

    sam_period = 1
    ind_sample_0 = np.arange(0, sample_size - diff_period, sam_period)
    sam_size = len(ind_sample_0)
    ind_sample_plus = ind_sample_0 + diff_period
    x_sample = xt[:, ind_sample_0] - xt[:, ind_sample_plus]

    Bcorr = np.corrcoef(x_sample)
    B = np.cov(x_sample)

    if max_var:
        alpha = max_var / np.amax(np.diag(B))
        B = alpha * B
    return B, Bcorr


def getBsimple(xt, nt, samfreq=2):
    """A very simple method to obtain the background error covariance.

    Obtained from a long run of a model.

    Parameters
    ----------
    xt : ndarry
        model trajectory. shape: (nx, nt)
    nt : int
        total time steps
    samfreq : int
        sampling frequency of the trajectory. Default: 2

    Returns
    -------
    B : ndarray
        The background covariance matrix
    Bcorr : ndarray
        The correlation matrix
    """
    err2 = 2
    # Precreate the matrix
    ind_sample = range(0, total_steps, samfreq)
    x_sample = xt[:, ind_sample]
    Bcorr = np.corrcoef(x_sample)

    B = np.cov(x_sample)
    alpha = err2/np.amax(np.diag(B))
    B = alpha*B
    return B, Bcorr


def getBClimate(Nx):
    Bc_row = np.zeros(Nx)
    Bc_row[0] = 3.9338
    Bc_row[1] = 1.3789; Bc_row[-1] = 1.3789
    Bc_row[2] = 0.4646; Bc_row[-2] = 0.4646
    return scipy.linalg.circulant(Bc_row)


def getPbs(Lxx, Ubkf, Nx, nsample, period_obs):
    ind = np.arange(period_obs, (nsample + 1) * period_obs, period_obs)
    Pbs_kf = np.empty((Nx, Nx, nsample))
    LPbs_kf = np.empty((Nx, Nx, nsample))
    for j, idx in enumerate(ind):
        aux = np.cov(np.squeeze(Ubkf[:, :, idx]), ddof=1)
        Pbs_kf[:, :, j] = aux
        LPbs_kf[:, :, j] = Lxx * aux
    return Pbs_kf, LPbs_kf


def msq(B):
    if np.allclose(np.diag(np.diag(B)), B):
        return np.sqrt(B)
    U, s, Vh = np.linalg.svd(B)
    B_sq = (U * np.sqrt(s[..., None, :])) @ Vh
    return B_sq

def minv(B):
    if np.allclose(np.diag(np.diag(B)), B):
        return np.diag(1./np.diag(B))
    U, s, Vh = np.linalg.svd(B)
    B_sq = (U /s[..., None, :]) @ Vh
    return B_sq


def evolve_cov(Bc, tmat, Nx, lags):
    Bt = np.empty((Nx,Nx,lags))
    B0t = np.empty((Nx,Nx,lags))
    for j in range(lags):
        B0t[:,:,j] = Bc@tmat[:,:,j].T    
        Bt[:,:,j] = tmat[:,:,j]@B0t[:,:,j]
    return Bt, B0t


def evolve_ensemble_cov(x0, nx, ne, lags, deltat, F, Bc_sq):
    t = deltat*np.arange(0, lags, 1)
    nt = len(t)
    uref = lorenz96(x0, (lags - 1)*deltat, deltat, 0, F)
    # evolve ensemble
    X = np.zeros((nx, ne, nt))
    for m in range(ne):
        csi_gen = np.random.RandomState(m)
        pert = Bc_sq@csi_gen.normal(0,1,nx)
        X[:,m] = lorenz96(x0 + pert, (lags - 1)*deltat, deltat, 0, F)

    Pbt = np.empty((nx,nx,nt))
    Pb0t = np.empty((nx,nx,nt))
    # initial time
    Xpert_0 = X[..., 0] - uref[:, None, 0]

    for j in range(nt):
        Xpert_t = X[..., j] - uref[:, None, j]
        Pbt[:,:,j] = Xpert_t@Xpert_t.T/(ne - 1) # cov of t with t
        Pb0t[:,:,j] = Xpert_0@Xpert_t.T/(ne - 1) # cov of 0 with t

    return Pbt,Pb0t


if __name__ == "__main__":
    nx = 5
    np.random.seed(100)
    x0 = np.random.random(nx)
    B, Bcorr = getBcanadian(x0, 0.01, 10, None, 8.0, 1000)
    print(B)
    print(Bcorr)
