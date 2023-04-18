import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import pickle
import soiling_model.base_models as smb
import soiling_model.fitting as smf

# %% Plot for the paper
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=10)
plt.rc('axes',labelsize=18)

def plot_for_paper(mod,rdat,sdat,train_experiments,train_mirrors,orientation,
                   rows_with_legend=[3],num_legend_cols=6,legend_shift=(0,0)):
    exps = list(mod.helios.tilt.keys())
    tilts = np.unique(mod.helios.tilt[0])
    fig,ax = plt.subplots(nrows=len(tilts)+2,ncols=len(exps),figsize=(12,15),sharex='col')
    ws_max = max([max(sdat.wind_speed[f]) for f in exps]) # max wind speed for setting y-axes
    dust_max = max([max(sdat.dust_concentration[f]) for f in exps]) # max dust concentration for setting y-axes
    hum_max = max([max(sdat.relative_humidity[f]) for f in exps]) # max relative humidity for setting y-axes

    # Define color for each orientation 
    colors = {'N':'blue','S':'red','E':'green','W':'magenta'}
    for ii,e in enumerate(exps):
        for jj,t in enumerate(tilts):
            tr = rdat.times[e]
            tr = (tr-tr[0]).astype('timedelta64[s]').astype(np.float64)/3600/24
            idx, = np.where(rdat.tilts[e][:,0] == t)

            idxs, = np.where(mod.helios.tilt[e][:,0] == t)
            idxs = idxs[0] # take first since all predictions are the same
            ts = sdat.time[e]
            ts = (ts-ts[0]).astype('timedelta64[s]').astype(np.float64)/3600/24
            
            for kk in idx: # reflectance data
                m = rdat.average[e][:,kk].squeeze()
                s = rdat.sigma_of_the_mean[e][:,kk].squeeze()
                m0 = m[0]
                m += (1-m0)
                error_two_sigma = 1.96*s

                color = colors[orientation[ii][kk]]
                ax[jj,ii].errorbar(tr,m,yerr=error_two_sigma,label=f'Orientation {orientation[ii][kk]}',color=color)

                if (e in train_experiments) and \
                    (rdat.mirror_names[e][kk] in train_mirrors):
                    a = ax[jj,e]
                    a.axvline(x=tr[0],ls=':',color='red')
                    a.axvline(x=tr[-1],ls=':',color='red')
                    a.patch.set_facecolor(color='yellow')
                    a.patch.set_alpha(0.2)
            
            ym = mod.helios.soiling_factor[e][idxs,:]
            var_predict = mod.helios.soiling_factor_prediction_variance[e][idxs,:]
            sigma_predict = np.sqrt( var_predict)
            Lp = ym - m0*1.96*sigma_predict
            Up = ym + m0*1.96*sigma_predict
            ax[jj,ii].plot(ts,ym,label='Soiling Factor Prediction',color='black')
            ax[jj,ii].fill_between(ts,Lp,Up,color='black',alpha=0.1,label=r'Prediction Interval')
            ax[jj,ii].grid('on')

            if jj==0:
                ax[jj,ii].set_title(f"Campaign {e}, Tilt: {t:.0f}"+r"$^{\circ}$")
            else:
                ax[jj,ii].set_title(f"Tilt: {t:.0f}"+r"$^{\circ}$")
            
    
        dust_conc = sdat.dust_concentration[e]
        ws = sdat.wind_speed[e]
        dust_type = sdat.dust_type[e]
        a2 = ax[-2,ii]
        a2.plot(ts,dust_conc, color='black')
        a2.tick_params(axis ='y', labelcolor = 'black')
        a2.set_ylim((0,1.01*dust_max))

        a2a = a2.twinx()
        p = a2a.plot(ts,ws,color='green')
        a2a.tick_params(axis ='y', labelcolor = 'green')
        a2a.set_ylim((0,1.01*ws_max))
        a2a.set_yticks((0,ws_max/2,ws_max))
        a2a.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        rel_hum = sdat.relative_humidity[e]
        a3 = ax[-1,ii]
        a3.plot(ts,rel_hum, color='blue')
        a3.tick_params(axis ='y', labelcolor = 'blue')
        a3.set_ylim((0,1.01*hum_max))
        
        if ii==0:
            fs = r"{0:s} $\frac{{\mu g}}{{m^3}}$"
            a2.set_ylabel(fs.format(dust_type),color='black')
            a3.set_ylabel("Relative \nHumidity (%)",color='blue')
        else:
            a2.set_yticklabels([])

        if ii==len(exps)-1:
            fs = r"{0:s} $\frac{{\mu g}}{{m^3}}$"
            a2a.set_ylabel('Wind Speed (m/s)', color='green')

    for ii,row in enumerate(ax):
        for jj,a in enumerate(row):
            if ii < len(tilts):
                a.set_ylim((0.85,1.01))
                a.set_yticks((0.85,0.9,0.95,1.0))
                if jj > 0:
                    a.set_yticklabels([])
                if (ii in rows_with_legend) and (jj==0):
                    ang = rdat.reflectometer_incidence_angle[jj]
                    a.set_ylabel(r"Normalized $\rho(t)$ at "+str(ang)+"$^{{\circ}}$")
                if (ii in rows_with_legend) and (jj==0):
                    # a.legend(loc='center',ncol=2,bbox_to_anchor=(0.25,0.5))
                    h_legend,labels_legend = a.get_legend_handles_labels()
            elif ii == len(tilts):
                a.set_yticks((0,150,300))
            else:
                a.set_yticks((0,50,100))
                
            
            if ii == ax.shape[0]-1:
                a.set_xlabel('Days')
            a.grid('on')


    fig.legend(h_legend,labels_legend,ncol=num_legend_cols,
               bbox_to_anchor=(0.9025+legend_shift[0],1.025+legend_shift[1]),bbox_transform=fig.transFigure)
    fig.subplots_adjust(wspace=0.1, hspace=0.3)
    fig.tight_layout()
    return fig,ax

def soiling_rate(alphas: np.ndarray,
                 alphas2: np.ndarray,
                 save_file: str,
                 M: int =1000):

    # load in parameters
    with open(save_file,'rb') as f:
        data = pickle.load(f)
        sim_data_train = data['simulation_data']
        imodel = data['model']
        log_param_hat = data['transformed_parameters']
        log_param_hat_cov = data['transformed_parameter_covariance']
        mu_tilde,sigma_dep = np.exp(log_param_hat)
    
    assert isinstance(imodel,smf.constant_mean_deposition_velocity), "Model in saved file must be constant-mean type."

    # simulate 
    sims = np.zeros((M,len(alphas)))
    inc_factor = imodel.helios.inc_ref_factor[0]
    for m in range(M):

        log_param = np.random.multivariate_normal(mean=log_param_hat,cov=log_param_hat_cov)
        mut,sigt = imodel.transform_scale(log_param)

        mean_loss_rate = inc_factor*mut*alphas # loss rate in one timestep
        var_loss_rate = (inc_factor*sigt)**2 * alphas2 # loss variance in one timestep

        # sample and sum days
        s = np.sqrt(var_loss_rate)
        dt_loss = np.random.normal(loc=mean_loss_rate,scale=s)
        
        # add samples to flattened list
        sims[m,:] = dt_loss*100
    
    return sims

def daily_soiling_rate( sim_dat: smb.simulation_inputs,
                        model_save_file: str,
                        percents: list or np.ndarray = None,
                        M: int = 10000,
                        dust_type="TSP"):
    # this assumes a horizontal reflector

    # get daily sums for \alpha and \alpha^2
    df = [pd.read_excel(f,"Weather") for f in sim_dat.file_name.values()]
    df = pd.concat(df)
    df.sort_values(by="Time",inplace=True)

    prototype_pm = getattr(sim_dat.dust,dust_type)[0]
    df['alpha'] = df[dust_type]/prototype_pm
    df['date'] = (df['Time'].dt.date)
    df['alpha2'] = df['alpha']**2
    daily_sum_alpha = (df.groupby('date')['alpha'].sum()).values
    daily_sum_alpha2 = (df.groupby('date')['alpha2'].sum()).values

    # get daily sums corresponding to percentiles of interest
    idx = np.argsort(daily_sum_alpha)
    daily_sum_alpha = daily_sum_alpha[idx]
    daily_sum_alpha2 = daily_sum_alpha2[idx]
    p = np.percentile(daily_sum_alpha,percents,interpolation='lower')
    mask = np.isin(daily_sum_alpha,p)
    sa = daily_sum_alpha[mask]
    sa2 = daily_sum_alpha2[mask]

    # simulate
    sims = soiling_rate(sa,
                        sa2,
                        model_save_file,
                        M=M)

    
    return sims,sa,sa2






