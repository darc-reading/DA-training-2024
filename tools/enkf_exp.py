import dataclasses
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from tools.obs import createH, gen_obs
from tools.diag import rmse_spread
from tools.enkf import kfs
from tools.plots import plotRMSP, plotDA_kf
from tools.L96_model import lorenz96

@dataclasses.dataclass
class Experiment:
    """
    Class that holds all settings for a single Lorenz-96 ensemble Kalman filter experiment. 
    
    Any setting can be overwritten by passing the new setting as key,value-pair to the constructor. 
    
    Methods 
    -------
    create_observations
        Sample observations from truth. 
    run
        Run the DA model and store the output. 
    calculate_metrics
        Calculate performance metrics.
    plot_metrics
        Plot the metrics produced by self.calculate_metrics as function of time. 
    plot_state
        Plot ensemble mean for forecast and analysis ensemble together with truth and observations 
        as function of time. 
    plot_localisation
         Plot the localisation matrix.
        
    Attributes
    ----------
    x0 : np.ndarray 
        Initial model state. Default is true initial condition.
    seed : int
        Seed for random number generator used to create observational errors
    period_obs : int>=1
        Number of time steps between observations
    obs_grid: 'all' | '0101' | 'landsea' | 'foot_cent' | 'foot_6'
        Observation operator to be used. 
    var_obs: float>0
        Observational error variance. 
    footprint : int 
        If obs_grid='foot_6' the length-scale of the observation foot print. 
    n_ens: int>=1
        Number of ensemble members.
    da_method: 'SEnKF' | 'ETKF'
        Ensemble Kalman method to be used. 
    inflation: float
        Ensemble inflation. If inflation=0 no ensemble inflation is applied. 
    loc_method: 'CG' | 'cutoff'
        Localisation method.
    loc_radius: NoneType | float>0
        Half-width localisation radius in grid points. If None, no localisation is applied. 
        
    Xb: 3D np.ndarray   
        Ensemble of background states with time along the 0th axis, grid position along the 1st axis and 
        ensemble member along the 2nd axis. 
    xb: 2D np.ndarray 
        Background ensemble mean with time along the 0th axis and grid position along the 1st axis. 
    Xa: 3D np.ndarray   
        Ensemble of analysis states with time along the 0th axis, grid position along the 1st axis and 
        ensemble member along the 2nd axis. 
    xa: 2D np.ndarray 
        Analysis ensemble mean with time along the 0th axis and grid position along the 1st axis. 
    L_obs: 2D np.ndarray 
        Array with localisation coefficients between elements in the state and observations.
    L_x: 2D np.ndarray 
        Array with localisation coefficients between elements in the state.
    tobs: 1D np.ndarray 
        Array with observation times. 
    y: 2D np.ndarray 
        Array with ith column containing observed values at time self.tobs[i]
    R: 2D np.ndarray 
        Observation error covariance matrix. 
    n_obs: int 
        Number of observations per observation time step. 
    observed_vars: list of int
        Indices of the state variables that are at one time or another are observed. 
    
    """
    
    #Default model settings. 
    x0: np.ndarray = dataclasses.field(default_factory=lambda:np.array(x0))
    Nx : int = 12
    t : np.ndarray = dataclasses.field(default_factory=lambda:np.array(t))
    xt : np.ndarray = dataclasses.field(default_factory=lambda:np.array(xt))
    F : float = 8.

    #Default observation operator settings.
    seed: int = 1 
    period_obs: int = 1
    obs_grid: str = 'all' 
    var_obs: float = 2.0 
    footprint: int = None
        
    #Default data assimilation system settings.
    n_ens: int = 24
    da_method: str = 'SEnKF' 
    inflation: float = 0.0
        
    #Localization settings
    loc_method: str = 'GC' 
    loc_radius: float = None
        
    def create_observations(self):
        """ Sample observations from truth. """
        self.n_obs, self.H = createH(self.obs_grid, self.Nx, self.footprint)
        self.observed_vars = [ivar for ivar,is_observed in enumerate(np.any(self.H, axis=0)) if is_observed]
        self.tobs, self.y, self.R = gen_obs(self.t, self.xt, self.period_obs, self.H, self.var_obs, self.seed)
    
    def run(self):
        """ Run the DA model and store the output. """
        #Create observations.
        self.create_observations()
        
        #State background/analysis
        self.Xb, self.xb, self.Xa, self.xa, self.L_obs, self.L_x = kfs(self.x0, float(self.F), lorenz96, self.t,
                                                                       self.tobs, self.y,
                                                                       self.H, self.R, self.inflation, 
                                                                       self.n_ens, self.da_method, 
                                                                       lam=self.loc_radius,
                                                                       loctype=self.loc_method,
                                                                       back0='random', desv=1.0,
                                                                       seed=self.seed)
        
        #cast 
        self.L_obs, self.L_x = np.array(self.L_obs), np.array(self.L_x)
        
    def calculate_metrics(self, step):
        """ 
        Calculate performance metrics.
        
        Parameters
        ----------
        step : int 
            Number of time steps between states for which metrics will be calculated. 
            
        Returns
        -------
        m : xr.Dataset object
            Dataset containing time series of RMSE and ensemble spread.
        
        """
        m = xr.Dataset(coords = {'DA':(['DA'],['background','analysis']), 
                                 'time':(['time'], self.t[::step])}
                      )
        
        #Initialise
        m['rmse'] = (['DA','time'], np.zeros((2,len(self.t[::step]))) )
        m['spread'] = (['DA','time'], np.zeros((2,len(self.t[::step]))) )
        
        #Background metrics
        m['rmse'][0], m['spread'][0] = rmse_spread(self.xt, self.xb, self.Xb, step)
        
        #Analysis metrics
        m['rmse'][1], m['spread'][1] = rmse_spread(self.xt, self.xa, self.Xa, step)

        return m
    
    def plot_metrics(self, step):
        """
        Plot the metrics produced by self.calculate_metrics as function of time. 
        
        Parameters
        ----------
        step : int 
            Number of time steps between states for which metrics will be calculated. 
            
        """
        m = self.calculate_metrics(step)
        plotRMSP(str(self), self.t, m['rmse'].sel(DA='background').data, m['rmse'].sel(DA='analysis').data,
                 m['spread'].sel(DA='background').data, m['spread'].sel(DA='analysis').data)
        
        
    def plot_state(self):
        """
        Plot ensemble mean for forecast and analysis ensemble together with truth and observations 
        as function of time. 
        """
        plotDA_kf(self.t, self.xt, self.tobs, self.H, self.y, self.Xb, self.xb, self.Xa, self.xa, self.__str__())
   
    def __str__(self):
        """ Name of the experiment. 
        
        Returns
        -------
        String with name of experiment. 
        
        """
        return ('ob freq:'+str(self.period_obs)+', density:'+str(self.obs_grid)+
                 ', err var:'+str(self.var_obs)+', M='+str(self.n_ens)+', loc_radius='+str(self.loc_radius)
                +', inflation='+str(self.inflation))