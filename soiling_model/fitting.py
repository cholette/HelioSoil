import soiling_model.base_models as smb
from soiling_model.utilities import _print_if,_check_keys, _parse_dust_str
import numpy as np
from numpy import radians as rad
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize_scalar, minimize
import numdifftools as ndt
import pickle

class common_fitting_methods:
    def compute_soiling_factor(self,rho0=None):
        # Converts helios.delta_soiled_area into an accumulated area loss and 
        # populates helios.soiling_factor.

        files = list(self.helios.tilt.keys())
        helios = self.helios
        helios.soiling_factor = {f: None for f in files} # clear the soiling factor
        
        for f in files:
            if rho0 is None:
                N_helios = helios.tilt[f].shape[0]
                cumulative_soil0 = np.zeros(N_helios)  # start from clean
            else:
                inc_factor = self.helios.inc_ref_factor[f]
                cumulative_soil0 = (1 - rho0[f]/self.helios.nominal_reflectance)/inc_factor # back-calculated soiled area from measurement

            cumulative_soil = np.c_[cumulative_soil0, helios.delta_soiled_area[f]]
            cumulative_soil = np.cumsum(cumulative_soil,axis=1)   # accumulate soiling 
            helios.soiling_factor[f] = 1 - cumulative_soil[:,1::]*helios.inc_ref_factor[f]  # soiling factor, still to be multiplied by rho0
        
        self.helios = helios

    def _compute_variance_of_measurements(self,sigma_dep,simulation_inputs,reflectance_data=None):
        
        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs,reflectance_data)

        sim_in = simulation_inputs
        files = list(sim_in.time.keys())

        s2_dep = sigma_dep**2
        s2total = dict.fromkeys(files)
        for f in files:

            if reflectance_data == None:
                pif = range(0,len(sim_in.time[f]))
            else:
                pif = reflectance_data.prediction_indices[f]

            b = self.helios.nominal_reflectance*self.helios.inc_ref_factor[f] # fixed for fitting experiments at reflectometer incidence angle
            try:
                attr = _parse_dust_str(sim_in.dust_type[f])
                den = getattr(sim_in.dust,attr) # dust.(sim_in.dust_type[f])
            except:
                raise ValueError("Dust measurement "+sim_in.dust_type[f]+\
                    " not present in dust class. Use dust_type="+sim_in.dust_type[f]+\
                        " option when initializing the model")
            
            alpha = sim_in.dust_concentration[f]/den[f]
            
            if reflectance_data == None:
                meas_sig = np.zeros(self.helios.tilt[f].shape).transpose()
            else:
                meas_sig = reflectance_data.sigma_of_the_mean[f]

            c2t = np.cumsum(alpha**2*np.cos(rad(self.helios.tilt[f]))**2,axis=1).transpose()
            ind1 = pif[0:-1]
            ind2 = [x-1 for x in pif[1::]]
            s2total[f] = s2_dep*b**2*(c2t[ind2,:]-c2t[ind1,:]) + meas_sig[0:-1,:]**2 + meas_sig[1::,:]**2
        
        return s2total

    def _sse(self,params,simulation_inputs,reflectance_data):
        # Computes the sum of squared errors between a soiling model and
        # the reflectance measurements. 

        pi = reflectance_data.prediction_indices
        meas = reflectance_data.average
        r0 = self.helios.nominal_reflectance # nominal clean reflectance

        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs,reflectance_data)

        sse = 0
        self.update_model_parameters(params)
        self.predict_soiling_factor(simulation_inputs,rho0=reflectance_data.rho0,verbose=False)
        sf = self.helios.soiling_factor
        files = list(sf.keys())
        for f in files:
            rho_prediction = r0*sf[f][:,pi[f]].transpose()
            sse += np.sum( (rho_prediction -meas[f] )**2 )
        return sse  

    def _negative_log_likelihood(self,params,simulation_inputs,reflectance_data):
        
        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs,reflectance_data)

        sim_in = simulation_inputs
        files = list(reflectance_data.times.keys())
        pi = reflectance_data.prediction_indices
        meas = reflectance_data.average
        r0 = self.helios.nominal_reflectance # nominal clean reflectance
        NL = [reflectance_data.average[f].shape[0] for f in files]
        
        # define optimization objective function (negative log likelihood)
        sigma_dep = params[1]    
        loglike = -0.5*np.sum(NL)*np.log(2*np.pi)
        self.update_model_parameters(params)
        self.predict_soiling_factor(simulation_inputs,rho0=reflectance_data.rho0,verbose=False)
        sf = self.helios.soiling_factor # soiling factor to be multiplied by clean reflectance

        # Compute variance in reflectance, not soiling factor
        s2total = self._compute_variance_of_measurements(sigma_dep,sim_in,reflectance_data=reflectance_data) 

        for f in files:
            delta_r = np.diff(meas[f],axis=0)
            rho_prediction =r0*sf[f][:,pi[f]].transpose()
            mu_delta_r = np.diff(rho_prediction,axis=0)
            loglike += np.sum( -0.5*np.log(s2total[f]) - (delta_r-mu_delta_r)**2/(2*s2total[f]) ) 
            
        return -loglike

    def _logpost(self,y,simulation_inputs,reflectance_data,priors):
        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs,reflectance_data)
        
        x = self.transform_scale(y)
        likelihood = -self._negative_log_likelihood(x,simulation_inputs,reflectance_data)

        unnormalized_posterior = likelihood
        names = list(priors.keys())
        for ii in range(len(names)):
            k = names[ii]
            unnormalized_posterior += priors[k].logpdf(y[ii])

        return unnormalized_posterior

    def fit_least_squares(self,simulation_inputs,reflectance_data,verbose=True):

        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs,reflectance_data)

        fun = lambda x: self._sse(x,simulation_inputs,reflectance_data)
        _print_if("Fitting parameters with least squares ...",verbose)
        xL = 1e-6 + 1.0
        xU = 1000 
        res = minimize_scalar(fun,bounds=(xL,xU),method="Bounded") # use bounded to prevent evaluation at values <=1
        _print_if("... done! \n estimated parameter is = "+str(res.x),verbose)
        return res.x, res.fun

    def fit_mle(self,simulation_inputs,reflectance_data,verbose=True,x0=None,
                transform_to_original_scale=False,**optim_kwargs):
        
        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(simulation_inputs,reflectance_data)

        ref_dat = reflectance_data
        sim_in = simulation_inputs
        
        if np.all(x0 == None): # intialize using least squares and 1D MLE
            _print_if("Getting initial deposition parameter guess via least squares",verbose)
            p0,sse = self.fit_least_squares(simulation_inputs,reflectance_data,verbose=False) 

            _print_if("Getting initial sigma_dep guess via MLE (at least-squares value for deposition parameters)",verbose)
            nloglike1D = lambda y: self._negative_log_likelihood([p0,y],sim_in,ref_dat)
            s0 = minimize_scalar(nloglike1D,bounds=(smb.tol,sse),method="Bounded") # use bounded to prevent evaluation at values <=1
            x0 = np.array([p0,s0.x])
            _print_if("x0 = ["+str(x0[0])+", "+str(x0[1])+"]",verbose) 
        
        # MLE. Transform to logs to ensure parameters are positive
        _print_if("Maximizing likelihood ...",verbose)
        y0 = self.transform_scale(x0,direction="forward")

        nloglike = lambda y: self._negative_log_likelihood(self.transform_scale(y),sim_in,ref_dat)
            
        res = minimize(nloglike,y0,**optim_kwargs)
        y = res.x 
        _print_if("  "+res.message,verbose)
        
        _print_if("Estimating parameter covariance using numerical approximation of Hessian ... ",verbose)
        H_log = 1e-6 + ndt.Hessian(nloglike)(y) # Hessian is in the log transformed space

        if transform_to_original_scale:
            
            # Get standard errors using observed information
            x_hat,H = self.transform_scale(y,likelihood_hessian=H_log)
            x_cov = np.linalg.inv(H)
            
            p_hat = x_hat
            p_cov = x_cov

        else:
            y_hat = y
            y_cov = np.linalg.inv(H_log) # Parameter covariance in the log space
            p_hat = y_hat
            p_cov = y_cov

        _print_if("... done!\n",verbose)
        return p_hat, p_cov

    def save_data(  self,log_p_hat=None,log_p_hat_cov=None,
                        training_simulation_data=None,training_reflectance_data=None):
            save_data = {'model':self,
                            'type':None}
            if log_p_hat is not None:
                save_data['transformed_parameters'] = log_p_hat
            if log_p_hat_cov is not None:
                save_data['transformed_parameter_covariance'] = log_p_hat_cov
            if training_simulation_data is not None:
                save_data['simulation_data'] = training_simulation_data
            if training_reflectance_data is not None:
                save_data['reflectance_data'] = training_reflectance_data
            
            return save_data

    def plot_soiling_factor(self,simulation_inputs,posterior_predictive_distribution_samples=None,
                        reflectance_data=None,figsize=None,reflectance_std='measurements',
                        save_path=None,fig_title=None,return_handles=False,
                        repeat_y_labels=True,orientation_strings=None):
    
        if reflectance_data is not None:
            self.predict_soiling_factor(simulation_inputs,rho0=reflectance_data.rho0)
        else:
            self.predict_soiling_factor(simulation_inputs)

        sim_in = simulation_inputs
        samples = posterior_predictive_distribution_samples
        files = list(sim_in.time.keys())
        N_mirrors = np.array([self.helios.tilt[f].shape[0] for f in files])
        if np.all(N_mirrors==N_mirrors[0]):
            N_mirrors = N_mirrors[0]
        else:
            raise ValueError("Number of mirrors must be the same for each experiment to use this function.")

        if reflectance_data != None:
            # check to ensure that reflectance_data and simulation_input keys correspond to the same files
            _check_keys(sim_in,reflectance_data)
        
        N_experiments = sim_in.N_simulations
        ws_max = max([max(sim_in.wind_speed[f]) for f in files]) # max wind speed for setting y-axes
        mean_predictions        =  {f: np.array([]) for f in files} 
        CI_upper_predictions    =  {f: np.array([]) for f in files}
        CI_lower_predictions    =  {f: np.array([]) for f in files}
        
        fig,ax = plt.subplots(N_mirrors+1,N_experiments,figsize = figsize,sharex="col")
        ax_wind = []
        for ii in range(N_experiments):
            f = files[ii]
            N_times = self.helios.tilt[f].shape[1]
            mean_predictions[f]     =  np.zeros(shape=(N_mirrors,N_times)) 
            CI_upper_predictions[f] =  np.zeros(shape=(N_mirrors,N_times))
            CI_lower_predictions[f] =  np.zeros(shape=(N_mirrors,N_times))

            dust_conc = sim_in.dust_concentration[f]
            ws = sim_in.wind_speed[f]
            dust_type = sim_in.dust_type[f]
            ts = sim_in.time[f]
            if reflectance_data != None:
                tr = reflectance_data.times[f]

            for jj in range(0,N_mirrors):
                if jj == 0:
                    tilt_str = r"Experiment "+str(ii)+ r", tilt = ${0:.0f}^{{\circ}}$"
                else:
                    tilt_str = r"tilt = ${0:.0f}^{{\circ}}$" # tilt only
                
                if orientation_strings is not None:
                    tilt_str += ", Orientation: "+orientation_strings[ii][jj]

                # get the axis handles
                if N_experiments == 1:
                    a = ax[jj] # experiment ii, mirror jj plot
                    a2 = ax[-1] # weather plot
                    am = ax[0] # plot to put legend on
                else:
                    a = ax[jj,ii]
                    a2 = ax[-1,ii]
                    am = ax[0,0]

                if reflectance_data is not None: # plot predictions and reflectance data
                    m = reflectance_data.average[f][:,jj]
                    r0 = self.helios.nominal_reflectance

                    if reflectance_std == 'measurements':
                        s = reflectance_data.sigma[f][:,jj]
                    elif reflectance_std == 'mean':
                        s = reflectance_data.sigma_of_the_mean[f][:,jj]
                    else:
                        raise ValueError("reflectance_std="+reflectance_std+" not recognized. Must be either \"measurements\" or \"mean\" ")

                    # measurement plots
                    error_two_sigma = 1.96*s
                    a.errorbar(tr,m,yerr=error_two_sigma,label="Measurement mean")

                    # mean prediction plot
                    if samples == None: # use soiling factor in helios
                        ym = r0*self.helios.soiling_factor[f][jj,:]
                        a.plot(sim_in.time[f],ym,label='Reflectance Prediction',color='black')
                    else:
                        y = r0*samples[f][jj,:,:]
                        ym = y.mean(axis=1)
                        a.plot(sim_in.time[f],ym,label='Reflectance Prediction (Bayesian)',color='red')

                    tilt = reflectance_data.tilts[f][jj]
                    if all(tilt==tilt[0]):
                        a.set_title(tilt_str.format(tilt[0]),fontsize=20)
                    else:
                        a.set_title(tilt_str.format(tilt.mean())+" (average)",fontsize=20)
                else: # predictions are from clean
                    r0 = self.helios.nominal_reflectance
                    if samples == None: 
                        ym = r0*self.helios.soiling_factor[f][jj,:] 
                        a.plot(sim_in.time[f],ym,label='Prediction',color='black')
                    else:
                        y = samples[f][jj,:,:]
                        ym = y.mean(y,axis=1)
                        a.plot(sim_in.time[f],ym,label='Prediction (Bayesian)',color='red')
                    
                    tilt = self.helios.tilt[f][jj,:]
                    if all(tilt==tilt[0]):
                        a.set_title(tilt_str.format(tilt[0]),fontsize=20)            
                    else:
                        a.set_title(tilt_str.format(tilt.mean())+" (average)",fontsize=20)

                if samples==None and len(self.helios.soiling_factor_prediction_variance)>0: # add +/- 2 sigma limits to the predictions, if sigma_dep is set
                    # var_predict = self.helios.delta_soiled_area_variance[f][jj,:]
                    var_predict = self.helios.soiling_factor_prediction_variance[f][jj,:]
                    sigma_predict = r0*np.sqrt( var_predict)
                    Lp = ym - 1.96*sigma_predict
                    Up = ym + 1.96*sigma_predict
                    a.fill_between(ts,Lp,Up,color='black',alpha=0.1,label=r'$\pm 2\sigma$ CI')
                elif samples != None: # use percentiles of posterior predictive samples for confidence intervals
                    Lp = np.percentile(y,2.5,axis=1)
                    Up = np.percentile(y,97.5,axis=1)
                    a.fill_between(ts,Lp,Up,color='red',alpha=0.1,label=r'95% Bayesian CI')
                
                a.xaxis.set_major_locator(mdates.DayLocator(interval=1)) # sets x ticks to day interval               
                
                if reflectance_data!=None: # reflectance is computed at reflectometer incidence angle
                    if repeat_y_labels or (ii==0):
                        ang = reflectance_data.reflectometer_incidence_angle[f]
                        s = a.set_ylabel(r"$\rho(t)$ at "+str(ang)+"$^{{\circ}}$")
                    else:
                        a.set_yticklabels([])

                else: # reflectance is computed at heliostat incidence angle. Put average incidence angle on axis label
                    if repeat_y_labels or (ii>0):
                        ang = np.mean( self.helios.incidence_angle[f] )
                        s = a.set_ylabel(r"soiling factor at "+str(ang)+"$^{{\circ}}$ \n (average)")
                    else:
                        a.set_yticklabels([])

                # set mean and CIs for output
                try:
                    mean_predictions[f][jj,:] = ym
                    CI_upper_predictions[f][jj,:] = Up
                    CI_lower_predictions[f][jj,:] = Lp 
                except:
                    mean_predictions[f][jj,:] = ym
            
            am.legend(fontsize = 16)
            label_str = dust_type + r" (mean = {0:.2f} $\mu g$/$m^3$)" 
            a2.plot(ts,dust_conc,label=label_str.format(dust_conc.mean()), color='blue')
            a2.xaxis.set_major_locator(mdates.DayLocator(interval=1)) # sets x ticks to day interval
            myFmt = mdates.DateFormatter('%d-%m-%Y')
            a2.xaxis.set_major_formatter(myFmt)
            a2.tick_params(axis ='y', labelcolor = 'blue')

            a2a = a2.twinx()
            p = a2a.plot(ts,ws,color='green',label="Wind Speed (mean = {0:.2f} m/s)".format(ws.mean()))
            ax_wind.append(a2a)
            a2a.tick_params(axis ='y', labelcolor = 'green')
            a2a.set_ylim((0,ws_max))
            
            if ii == 0: # ylabel for TSP on leftmost plot only
                fs = r"{0:s} $\frac{{\mu g}}{{m^3}}$"
                a2.set_ylabel(fs.format(dust_type),color='blue')
            else:
                a2.set_yticklabels([])

            if ii == N_experiments-1: # ylabel for wind speed on rightmost plot only
                a2a.set_ylabel('Wind Speed (m/s)', color='green')
            else:
                a2a.set_yticklabels([]) 
            
            a2.set_title(label_str.format(dust_conc.mean())+", \n Wind Speed (mean = {0:.2f} m/s)".format(ws.mean()),fontsize=20)
        
        if N_experiments > 1:

            # share y axes for all reflectance measurments
            ymax = max([x.get_ylim()[1] for x in ax[0:-1,:].flatten()])
            ymin = min([x.get_ylim()[0] for x in ax[0:-1,:].flatten()])
            for a in ax[0:-1,:].flatten(): 
                a.set_ylim(ymin,1)

            # share y axes for weather variables of the same type
            ymax_dust = max([x.get_ylim()[1] for x in ax[-1,:]])
            ymax_wind = max([x.get_ylim()[1] for x in ax_wind])
            for a in ax[-1,:]:
                a.set_ylim(0,1.1*ymax_dust)
            for a in ax_wind:
                a.set_ylim(0,1.1*ymax_wind)
        else:
            ymax = max([x.get_ylim()[1] for x in ax[0:-1].flatten()])
            ymin = min([x.get_ylim()[0] for x in ax[0:-1].flatten()])
            for a in ax[0:-1]:
                a.set_ylim(ymin,ymax)

        fig.autofmt_xdate()
        if save_path != None:
            fig.savefig(save_path)

        fig.suptitle(fig_title, fontsize=16)
        fig.tight_layout()
        if return_handles:
            return fig,ax,mean_predictions,CI_lower_predictions,CI_upper_predictions
        else:
            return mean_predictions,CI_lower_predictions,CI_upper_predictions
    
class semi_physical(smb.physical_base,common_fitting_methods):
    def __init__(self,file_params):
        table = pd.read_excel(file_params,index_col="Parameter")
        super().__init__()
        self.import_site_data_and_constants(file_params)
        self.helios.hamaker = float(table.loc['hamaker_glass'].Value)
        self.helios.poisson = float(table.loc['poisson_glass'].Value)
        self.helios.youngs_modulus = float(table.loc['youngs_modulus_glass'].Value)
        self.helios.nominal_reflectance = float(table.loc['nominal_reflectance'].Value)
        if not(isinstance(self.helios.stow_tilt,float)) and not(isinstance(self.helios.stow_tilt,int)):
            self.helios.stow_tilt = None

    def helios_angles(self,simulation_inputs,reflectance_data,verbose=True,second_surface=True):

        sim_in = simulation_inputs
        ref_dat = reflectance_data
        files = list(sim_in.time.keys())
        N_experiments = len(files)

        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(sim_in,ref_dat)

        _print_if("Setting tilts for "+str(N_experiments)+" experiments",verbose)
        helios = self.helios
        helios.tilt = {f: None for f in files} # clear the existing tilts
        helios.acceptance_angles = {f: None for f in files} # clear the existing acceptance angles
        for ii in range(N_experiments):
            f = files[ii]
            # start_idx = ref_dat.prediction_indices[f][0]        # Define the start index
            # end_idx = ref_dat.prediction_indices[f][-1]         # Define the start last index
            # tilts = ref_dat.tilts[f][:, start_idx:end_idx + 1]  # Extract the subset of tilts (end_idx + 1 include the last index)
            tilts = ref_dat.tilts[f] # THIS CANNOT MANAGE THE TRANSFORMATION OF SIM_IN WITH DAILY AVERAGE WHEN CHANGING START TIME (IF APPLIED AT THE BEGINNING)
            N_times = len(sim_in.time[f])
            N_helios = tilts.shape[0]
            self.helios.acceptance_angles[f] = [ref_dat.reflectometer_acceptance_angle[ii]]*N_helios
            self.helios.extinction_weighting[f] = [] # reset extinction weighting since heliostats are "new" - WHY?? THIS DEPENDS ONLY ON DUST! NOT?
            
            helios.tilt[f] = np.zeros((0,N_times))
            for jj in range(N_helios):
                row_mask = np.ones((1,N_times))
                helios.tilt[f] = np.vstack((helios.tilt[f],tilts[jj]*row_mask))

            helios.elevation[f] = 90-helios.tilt[f]
            helios.incidence_angle[f] = reflectance_data.reflectometer_incidence_angle[f]

            if second_surface==False:
                helios.inc_ref_factor[f] = (1+np.sin(rad(helios.incidence_angle[f])))/np.cos(rad(helios.incidence_angle[f])) # first surface
                _print_if("First surface model",verbose)
            elif second_surface==True:
                helios.inc_ref_factor[f] = 2/np.cos(rad(helios.incidence_angle[f]))  # second surface model
                _print_if("Second surface model",verbose)
            else:
                _print_if("Choose either first or second surface model",verbose)
    
        self.helios = helios

    def predict_soiling_factor(self,simulation_inputs,rho0=None,hrz0=None,
                               sigma_dep=None,verbose=True):
        # Uses simulation inputs and fitted model to predict the soiling
        # factor and the prediction variance (stored in 
        # helios.soiling_factor and helios.soiling_factor_prediction_variance,
        # respectively).

        self.deposition_flux(simulation_inputs,hrz0=hrz0,verbose=verbose)
        self.adhesion_removal(simulation_inputs,verbose=verbose)
        self.calculate_delta_soiled_area(simulation_inputs,sigma_dep=sigma_dep,verbose=verbose)
        self.compute_soiling_factor(rho0=rho0)

        # prediction variance
        if self.sigma_dep is not None:
            for f in self.helios.soiling_factor.keys():
                inc_factor = self.helios.inc_ref_factor[f]
                dsav = self.helios.delta_soiled_area_variance[f]
                self.helios.soiling_factor_prediction_variance[f] = np.cumsum( inc_factor**2 * dsav,axis=1 )
        else:
            self.helios.soiling_factor_prediction_variance = {}

    def fit_mle(self,simulation_inputs,reflectance_data,verbose=True,x0=None,
                transform_to_original_scale=False,**optim_kwargs):
        
        p_hat,p_cov = super().fit_mle(   simulation_inputs,reflectance_data,verbose=True,x0=x0,
                                        transform_to_original_scale=transform_to_original_scale,**optim_kwargs)
        

        # print estimates and confidence intervals
        fmtCI = "95% confidence interval for {0:s}: [{1:.2e}, {2:.2e}]"
        if transform_to_original_scale:
            fmtE = "hrz0 = {0:.2e}, sigma_dep = {1:.2e}"
            _print_if(fmtE.format(p_hat[0],p_hat[1]),verbose)

            # print confidence intervals
            s = np.sqrt(np.diag(p_cov))
            x_ci = p_hat + 1.96*s*np.array([[-1],[1]])
            _print_if(fmtCI.format("hrz0",x_ci[0,0],x_ci[1,0]),verbose)
            _print_if(fmtCI.format("sigma_dep",x_ci[0,1],x_ci[1,1]),verbose)
        else:
            fmtE = "log(log(hrz0)) = {0:.2e}, sigma_dep = {1:.2e}"
            _print_if(fmtE.format(p_hat[0],p_hat[1]),verbose)
            s = np.sqrt(np.diag(p_cov))
            y_ci = p_hat + 1.96*s*np.array([[-1],[1]])
            _print_if(fmtCI.format("log(log(hrz0))",y_ci[0,0],y_ci[1,0]),verbose)
            _print_if(fmtCI.format("log(sigma_dep)",y_ci[0,1],y_ci[1,1]),verbose)
        
        return p_hat,p_cov
                
    def transform_scale(self,x,likelihood_hessian=None,direction="inverse"):
        # direction is either "forward" (to log-scaled space) or "inverse" (back to original scale)
        x = np.array(x)
        if direction == "inverse":
            z = np.array( [np.exp(np.exp(x[0])),np.exp(x[1])] )
        elif direction == "forward":
            z = np.array( [np.log(np.log(x[0])),np.log(x[1])] )
        else:
            raise ValueError("Transformation direction not recognized.")

        if not isinstance(likelihood_hessian,np.ndarray): # can't use likelihood_hessian == None because it is an array if supplied
            return z
        else:
            #Jacobian for transformation. See Reparameterization at https://en.wikipedia.org/wiki/Fisher_information 
            J = np.array([  [np.exp(x[0] + np.exp(x[0])),0],\
                            [0, np.exp(x[1])]   ])

            if direction == "inverse":
                Ji = inv(J)                            
                H = Ji.transpose() @ likelihood_hessian @ Ji 
            elif direction == "forward":
                H = J.transpose() @ likelihood_hessian @ J 

            return z,H 

    def update_model_parameters(self,x):
        if isinstance(x,list) or isinstance(x,np.ndarray) :
            self.hrz0 = x[0]
            if len(x)>1:
                self.sigma_dep = x[1]
        else:
            self.hrz0 = x
            self.sigma_dep = None
    
    def save(self,file_name,log_p_hat=None,log_p_hat_cov=None,
             training_simulation_data=None,training_reflectance_data=None):
        save_data = super().save_data(log_p_hat,log_p_hat_cov,
                                      training_simulation_data,
                                      training_reflectance_data)
        
        save_data['type'] = "semi-physical"
        with open(file_name,'wb') as f:
            pickle.dump(save_data,f)

        # with open(file_name, 'w') as f:
        #     f.write(str(save_data))


class constant_mean_deposition(smb.constant_mean_base,common_fitting_methods):
    
    def __init__(self,file_params):
        super().__init__()
        self.import_site_data_and_constants(file_params)
        table = pd.read_excel(file_params,index_col="Parameter")
        self.helios.nominal_reflectance = float(table.loc['nominal_reflectance'].Value)

    def helios_angles(self,simulation_inputs,reflectance_data,verbose=True,second_surface=True):

        sim_in = simulation_inputs
        ref_dat = reflectance_data
        files = list(sim_in.time.keys())
        N_experiments = len(files)

        # check to ensure that reflectance_data and simulation_input keys correspond to the same files
        _check_keys(sim_in,ref_dat)

        _print_if("Setting tilts for "+str(N_experiments)+" experiments",verbose)
        helios = self.helios
        helios.tilt = {f: None for f in files} # clear the existing tilts
        helios.acceptance_angles = {f: None for f in files} # clear the existing tilts
        for ii in range(N_experiments):
            f = files[ii]
            # start_idx = ref_dat.prediction_indices[f][0]        # Define the start index
            # end_idx = ref_dat.prediction_indices[f][-1]         # Define the start last index
            # tilts = ref_dat.tilts[f][:, start_idx:end_idx + 1]  # Extract the subset of tilts (end_idx + 1 include the last index)
            tilts = ref_dat.tilts[f] # THIS COULD NOT MANAGE THE TRANSFORMATION OF SIM_IN WITH DAILY AVERAGE WHEN CHANGING START TIME
            N_times = len(sim_in.time[f])
            N_helios = tilts.shape[0]
            self.helios.acceptance_angles[f] = [ref_dat.reflectometer_acceptance_angle[ii]]*N_helios
            self.helios.extinction_weighting[f] = [] # reset extinction weighting since heliostats are "new"
            
            helios.tilt[f] = np.zeros((0,N_times))
            for jj in range(N_helios):
                row_mask = np.ones((1,N_times))
                helios.tilt[f] = np.vstack((helios.tilt[f],tilts[jj]*row_mask))

            helios.elevation[f] = 90-helios.tilt[f]
            helios.incidence_angle[f] = reflectance_data.reflectometer_incidence_angle[f]

            if second_surface==False:
                helios.inc_ref_factor[f] = (1+np.sin(rad(helios.incidence_angle[f])))/np.cos(rad(helios.incidence_angle[f])) # first surface
                _print_if("First surface model",verbose)
            elif second_surface==True:
                helios.inc_ref_factor[f] = 2/np.cos(rad(helios.incidence_angle[f]))  # second surface model
                _print_if("Second surface model",verbose)
            else:
                _print_if("Choose either first or second surface model",verbose)
    
        self.helios = helios

    def predict_soiling_factor(self,simulation_inputs,rho0=None,mu_tilde=None,sigma_dep=None,verbose=True):

        sim_in = simulation_inputs
        self.calculate_delta_soiled_area(sim_in,mu_tilde=mu_tilde,sigma_dep=sigma_dep,verbose=verbose)       
        self.compute_soiling_factor(rho0=rho0)

    def fit_map(self, simulation_inputs, reflectance_data, priors, verbose=True, 
                x0=None, transform_to_original_scale=False,save_file=None):

        _print_if("Getting MAP estimates ... ",verbose)
        y,y_cov = super().fit_map(simulation_inputs, reflectance_data, priors, verbose=False, x0=x0, transform_to_original_scale=False)

        _print_if("========== MAP Estimates ======== ",verbose)
        if transform_to_original_scale:
            x_hat,x_hat_cov = self.transform_scale(y,y_cov)

            # print estimates
            fmt = "mu_tilde = {0:.2e}, sigma_dep = {1:.2e}"
            _print_if(fmt.format(x_hat[0],x_hat[1]),verbose)

            # print confidence intervals
            s = np.sqrt(np.diag(x_hat_cov))
            x_ci = x_hat + 1.96*s*np.array([[-1],[1]])
            fmt = "95% confidence interval for {0:s}: [{1:.2e}, {2:.2e}]"
            _print_if(fmt.format("mu_tilde",x_ci[0,0],x_ci[1,0]),verbose)
            _print_if(fmt.format("sigma_dep",x_ci[0,1],x_ci[1,1]),verbose)

        else:
            x_hat = y
            x_hat_cov = y_cov

            # print estimates
            fmt = "log(mu_tilde) = {0:.2e}, log(sigma_dep) = {1:.2e} "
            _print_if(fmt.format(x_hat[0],x_hat[1]),verbose)

            # print confidence intervals
            s = np.sqrt(np.diag(x_hat_cov))
            x_ci = x_hat + 1.96*s*np.array([[-1],[1]])
            fmt = "95% confidence interval for {0:s}: [{1:.2e}, {2:.2e}]"
            _print_if(fmt.format("log(mu_tilde)",x_ci[0,0],x_ci[1,0]),verbose)
            _print_if(fmt.format("log(sigma_dep)",x_ci[0,1],x_ci[1,1]),verbose)
        
        return x_hat, x_hat_cov

    def fit_mle(self, simulation_inputs, reflectance_data, verbose=True, x0=None, 
                transform_to_original_scale=False,save_file=None):

        _print_if("Getting MLE estimates ... ",verbose)
        y,y_cov = super().fit_mle(simulation_inputs, reflectance_data, verbose=False, x0=x0, transform_to_original_scale=False)
        H_log = np.linalg.inv(y_cov)

        _print_if("========== MLE Estimates ======== ",verbose)
        if transform_to_original_scale:
            x_hat,H = self.transform_scale(y,H_log)
            x_hat_cov = np.linalg.inv(H)

            # print estimates
            fmt = "mu_tilde = {0:.2e}, sigma_dep = {1:.2e}"
            _print_if(fmt.format(x_hat[0],x_hat[1]),verbose)

            # print confidence intervals
            s = np.sqrt(np.diag(x_hat_cov))
            x_ci = x_hat + 1.96*s*np.array([[-1],[1]])
            fmt = "95% confidence interval for {0:s}: [{1:.2e}, {2:.2e}]"
            _print_if(fmt.format("mu_tilde",x_ci[0,0],x_ci[1,0]),verbose)
            _print_if(fmt.format("sigma_dep",x_ci[0,1],x_ci[1,1]),verbose)

        else:
            x_hat = y
            x_hat_cov = y_cov

            # print estimates
            fmt = "log(mu_tilde) = {0:.2e}, log(sigma_dep) = {1:.2e} "
            _print_if(fmt.format(x_hat[0],x_hat[1]),verbose)

            # print confidence intervals
            s = np.sqrt(np.diag(x_hat_cov))
            x_ci = x_hat + 1.96*s*np.array([[-1],[1]])
            fmt = "95% confidence interval for {0:s}: [{1:.2e}, {2:.2e}]"
            _print_if(fmt.format("log(mu_tilde)",x_ci[0,0],x_ci[1,0]),verbose)
            _print_if(fmt.format("log(sigma_dep)",x_ci[0,1],x_ci[1,1]),verbose)
        
        return x_hat, x_hat_cov

    def transform_scale(self, x, likelihood_hessian=None,direction="inverse"):
        if isinstance(x,np.ndarray) or isinstance(x,list):
            x = np.array(x)
            if direction == "inverse":
                z = np.array( [np.exp(x[0]),np.exp(x[1])] )
            elif direction == "forward":
                z = np.array( [np.log(x[0]),np.log(x[1])] )
            else:
                raise ValueError("Transformation direction not recognized.")

            if not isinstance(likelihood_hessian,np.ndarray): # can't use likelihood_hessian == None because it is an array if supplied
                return z
            else:
                #Jacobian for transformation. See Reparameterization at https://en.wikipedia.org/wiki/Fisher_information 
                J = np.array([  [np.exp(x[0]),0],\
                                [0, np.exp(x[1])]   ])

                if direction == "inverse":
                    Ji = inv(J)                            
                    H = Ji.transpose() @ likelihood_hessian @ Ji
                elif direction == "forward":
                    H = J.transpose() @ likelihood_hessian @ J 

                return z,H  

        # elif isinstance(x,az.data.inference_data.InferenceData):
        #     if likelihood_hessian != None:
        #         print("Warning: You have supplied an Arviz infrenceData object. The supplied likelihood Hessian will be ignored.")
            
        #     p = x.posterior
        #     if direction == "inverse":
        #         p2 = {  'mu_tilde': np.exp(p.log_mu_tilde),\
        #                 'sigma_dep':np.exp(p.log_sigma_dep) 
        #             }
        #     elif direction == "forward":
        #         p2 = {  'log_mu_tilde': np.log(p.mu_tilde),\
        #                 'log_sigma_dep':np.log(p.sigma_dep) 
        #             }
        #     p2 = az.convert_to_inference_data(p2)
        #     return p2
    
    def update_model_parameters(self,x):
        if isinstance(x,list) or isinstance(x,np.ndarray) :
            self.mu_tilde = x[0]
            if len(x)>1:
                self.sigma_dep = x[1]
        else:
            self.mu_tilde = x

    def save(self,file_name,log_p_hat=None,log_p_hat_cov=None,
             training_simulation_data=None,training_reflectance_data=None):
        save_data = super().save_data(log_p_hat,log_p_hat_cov,
                                      training_simulation_data,
                                      training_reflectance_data)
        
        save_data['type'] = "constant mean"
        with open(file_name,'wb') as f:
            pickle.dump(save_data,f)
                