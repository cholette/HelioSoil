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
                   rows_with_legend=[3],num_legend_cols=6,legend_shift=(0,0),plot_rh=True,
                   yticks=None):
    
    if any("augusta".lower() in value.lower() for value in sdat.file_name.values()):
        plot_rh = False  # the RH sensor is broken since the beginning of Port Augusta experiments

    mod.predict_soiling_factor(sdat,rho0=rdat.rho0) # ensure predictions are fresh
    r0 = mod.helios.nominal_reflectance

    exps = list(mod.helios.tilt.keys())
    tilts = np.unique(mod.helios.tilt[0])

    if plot_rh:
        fig,ax = plt.subplots(nrows=len(tilts)+2,ncols=len(exps),figsize=(12,15),sharex='col')
    else:
        fig,ax = plt.subplots(nrows=len(tilts)+1,ncols=len(exps),figsize=(12,15),sharex='col')
        
    ws_max = max([max(sdat.wind_speed[f]) for f in exps]) # max wind speed for setting y-axes
    dust_max = max([max(sdat.dust_concentration[f]) for f in exps]) # max dust concentration for setting y-axes

    if plot_rh:
        hum_max = max([max(sdat.relative_humidity[f]) for f in exps]) # max relative humidity for setting y-axes

    # Define color for each orientation 
    if any("augusta".lower() in value.lower() for value in sdat.file_name.values()):
        colors = {'NW':'blue','SE':'red'}
    elif any("yadnarie".lower() in value.lower() for value in sdat.file_name.values()):
        colors = {'NE':'blue','SE':'red','SW':'green','NW':'magenta','N/A':'blue'}
    else:
        colors = {'N':'blue','S':'red','E':'green','W':'magenta','N/A':'blue'}
    
    ref_output = {}
    for ii,e in enumerate(exps):
        for jj,t in enumerate(tilts):
            tr = rdat.times[e]
            tr = (tr-tr[0]).astype('timedelta64[s]').astype(np.float64)/3600/24
            
            idx, = np.where(rdat.tilts[e][:,0] == t)
            idxs, = np.where(mod.helios.tilt[e][:,0] == t)

            if not any(rdat.tilts[e][:,0] == t):
                print('Tilt Not Found')
                continue

            if t==0 and any("augusta".lower() in value.lower() for value in sdat.file_name.values()):
                idx = idx[1:]   # In the Port Augusta data the first mirror is cleaned every time and used as control reference
            if t==0 and any("augusta".lower() in value.lower() for value in sdat.file_name.values()):
                idxs = idxs[1:]   # In the Port Augusta data the first mirror is cleaned every time and used as control reference
                        
            if t==0 and any("mildura".lower() in value.lower() for value in sdat.file_name.values()):
                idx = idx[2:]   # In the Mildura data the first mirror is cleaned every time and used as control reference and the 2nd is used for Heliostat comparison
            if t==0 and any("mildura".lower() in value.lower() for value in sdat.file_name.values()):
                idxs = idxs[2:]   # In the Mildura data the first mirror is cleaned every time and used as control reference and the 2nd is used for Heliostat comparison
            
            idxs = idxs[0] # take first since all predictions are the same
            ts = sdat.time[e].values[0:rdat.prediction_indices[e][-1]+1]
            ts = (ts-ts[0]).astype('timedelta64[s]').astype(np.float64)/3600/24
            
            for kk in idx: # reflectance data
                m = rdat.average[e][:,kk].squeeze().copy()
                s = rdat.sigma_of_the_mean[e][:,kk].squeeze()
                m += (1-m[0]) # shift up so that all start at 1.0 for visual comparison
                error_two_sigma = 1.96*s

                color = colors[orientation[ii][kk]]
                if np.ndim(ax) == 1:
                    ax = np.vstack(ax)  # create a fictious 2D array with only one column
                ax[jj,ii].errorbar(tr,m,yerr=error_two_sigma,label=f'Orientation {orientation[ii][kk]}',color=color)

                if (e in train_experiments) and \
                    (rdat.mirror_names[e][kk] in train_mirrors):
                    a = ax[jj,e]
                    a.axvline(x=tr[0],ls=':',color='red')
                    a.axvline(x=tr[-1],ls=':',color='red')
                    a.patch.set_facecolor(color='yellow')
                    a.patch.set_alpha(0.2)
            
            ym = r0*mod.helios.soiling_factor[e][idxs,0:rdat.prediction_indices[e][-1]+1] # ensure columns are time index # +1 is required to include the last time point (slicing would exclude it)
            # ref_output[e][jj] = ym
            ref_output[(e, int(t))] = ym.copy()
            if ym.ndim == 1:
                ym += (1.0-ym[0])
            else:
                ym += (1.0-ym[:,0])
            var_predict = mod.helios.soiling_factor_prediction_variance[e][idxs,0:rdat.prediction_indices[e][-1]+1]
            sigma_predict = r0*np.sqrt(var_predict)
            Lp = ym - 1.96*sigma_predict
            Up = ym + 1.96*sigma_predict
            ax[jj,ii].plot(ts,ym,label='Prediction Mean',color='black')
            ax[jj,ii].fill_between(ts,Lp,Up,color='black',alpha=0.1,label=r'Prediction Interval')
            ax[jj,ii].grid('on')

            if jj==0:
                ax[jj,ii].set_title(f"Campaign {e+1}, Tilt: {t:.0f}"+r"$^{\circ}$")
            else:
                ax[jj,ii].set_title(f"Tilt: {t:.0f}"+r"$^{\circ}$")
            
    
        new_var = sdat.dust_concentration[e][0:rdat.prediction_indices[e][-1]+1]
        dust_conc = new_var
        ws = sdat.wind_speed[e][0:rdat.prediction_indices[e][-1]+1]
        dust_type = sdat.dust_type[e][0:rdat.prediction_indices[e][-1]+1]
        if plot_rh:
            a2 = ax[-2,ii]
        else:
            a2 = ax[-1,ii]

        a2.plot(ts,dust_conc, color='brown')
        a2.tick_params(axis ='y', labelcolor = 'brown')
        a2.set_ylim((0,1.01*dust_max))

        a2a = a2.twinx()
        p = a2a.plot(ts,ws,color='green')
        a2a.tick_params(axis ='y', labelcolor = 'green')
        a2a.set_ylim((0,1.01*ws_max))
        a2a.set_yticks((0,ws_max/2,ws_max))
        a2a.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        if plot_rh:
            # THE LINE BELOW AVOID THE LAST ELEMENT (SO IT HAS THE SAME DIMENSION)
            rel_hum = sdat.relative_humidity[e][0:rdat.prediction_indices[e][-1]+1]
            a3 = ax[-1,ii]
            a3.plot(ts,rel_hum, color='blue')
            a3.tick_params(axis ='y', labelcolor = 'blue')
            a3.set_ylim((0,1.01*hum_max))
        
        if ii==0:
            fs = r"{0:s} $\frac{{\mu g}}{{m^3}}$"
            a2.set_ylabel(fs.format(dust_type),color='brown')

            if plot_rh:
                a3.set_ylabel("Relative \nHumidity (%)",color='blue')
        else:
            a2.set_yticklabels([])

        if ii==len(exps)-1:
            fs = r"{0:s} $\frac{{\mu g}}{{m^3}}$"
            a2a.set_ylabel('Wind Speed (m/s)', color='green')

    # for ii,row in enumerate(ax):
        # for jj,a in enumerate(row):
        #     if ii < len(tilts):
        #         if yticks is None:
        #             a.set_ylim((0.85,1.01))
        #             a.set_yticks((0.85,0.90,0.95,1.0))
        #         else:
        #             a.set_ylim((min(yticks),max(yticks)))
        #             a.set_yticks(yticks)

        #         if jj > 0:
        #             a.set_yticklabels([])
        #         if (ii in rows_with_legend) and (jj==0):
        #             ang = rdat.reflectometer_incidence_angle[jj]
        #             a.set_ylabel(r"Normalized reflectance $\rho(0)-\rho(t)$ at "+str(ang)+"$^{{\circ}}$")
        #         if (ii in rows_with_legend) and (jj==0):
        #             # a.legend(loc='center',ncol=2,bbox_to_anchor=(0.25,0.5))
        #             h_legend,labels_legend = a.get_legend_handles_labels()
        #     elif ii == len(tilts):
        #         a.set_yticks((0,150,300))
        #     else:
        #         a.set_yticks((0,50,100))
                
            
        #     if ii == ax.shape[0]-1:
        #         a.set_xlabel('Days')
        #     a.grid('on')

    handles_leg = []
    labels_leg = []

    for ii, row in enumerate(ax):
        for jj, a in enumerate(row):
            if ii < len(tilts):
                if yticks is None:
                    a.set_ylim((0.85, 1.01))
                    a.set_yticks((0.85, 0.90, 0.95, 1.0))
                else:
                    a.set_ylim((min(yticks), max(yticks)))
                    a.set_yticks(yticks)

                if jj > 0:
                    a.set_yticklabels([])

                if ii in rows_with_legend and jj == 0:
                    ang = rdat.reflectometer_incidence_angle[jj]
                    a.set_ylabel(r"Normalized reflectance $\rho(0)-\rho(t)$ at " + str(ang) + "$^{{\circ}}$")

                # Get legend handles and labels
                hs, ls = a.get_legend_handles_labels()
                handles_leg.extend(hs)
                labels_leg.extend(ls)

                # # Remove duplicates from the legend
                # for handle, label in zip(h_legend, labels_legend):
                #     if label not in seen_labels:
                #         seen_labels.add(label)
                #     else:
                #         # Remove the duplicate handle from the list
                #         h_legend.remove(handle)
                #         labels_legend.remove(label)

            elif ii == len(tilts):
                a.set_yticks((0, 150, 300))
            else:
                a.set_yticks((0, 50, 100))
    
    # Remove duplicates by filtering unique labels and keeping corresponding handles
    unique_labels = []
    unique_handles = []

    for handle, label in zip(handles_leg, labels_leg):
        if label not in unique_labels:  # Check if the label is already in the unique list
            unique_labels.append(label)  # Add unique label
            unique_handles.append(handle)  # Add corresponding handle

    fig.legend(unique_handles,unique_labels,ncol=num_legend_cols,
               bbox_to_anchor=(0.9025+legend_shift[0],1.025+legend_shift[1]),bbox_transform=fig.transFigure)
    fig.subplots_adjust(wspace=0.1, hspace=0.3)
    fig.tight_layout()
    return fig,ax,ref_output

def plot_for_heliostats(mod,rdat,sdat,train_experiments,train_mirrors,orientation,
                   rows_with_legend=[3],num_legend_cols=6,legend_shift=(0,0),plot_rh=True,
                   yticks=None,figsize=None):
    
    mod.predict_soiling_factor(sdat,rho0=rdat.rho0) # ensure predictions are fresh
    r0 = mod.helios.nominal_reflectance

    exps = list(mod.helios.soiling_factor.keys())
    # tilts = np.unique(mod.helios.tilt[0]) # this is for the mirror rig with fixed tilt
    hels = rdat.mirror_names[0] # [0] assuming all campaigns uses the same heliostats, otherwise modify here!

    if plot_rh:
        fig,ax = plt.subplots(nrows=len(hels)+2,ncols=len(exps),figsize=(12,15),sharex='col')
    else:
        fig,ax = plt.subplots(nrows=len(hels)+1,ncols=len(exps),figsize=(12,15),sharex='col')
        
    ws_max = max([max(sdat.wind_speed[f]) for f in exps]) # max wind speed for setting y-axes
    dust_max = max([max(sdat.dust_concentration[f]) for f in exps]) # max dust concentration for setting y-axes

    if plot_rh:
        hum_max = max([max(sdat.relative_humidity[f]) for f in exps]) # max relative humidity for setting y-axes

    ref_output = []
    for ii,e in enumerate(exps):
        exp_output = {}
        for jj,t in enumerate(hels):
            tr = rdat.times[e]
            tr_day = (tr-tr[0]).astype('timedelta64[s]').astype(np.float64)/3600/24
            
            ts = sdat.time[e].values[0:rdat.prediction_indices[e][-1]+1]
            ts_day = (ts-ts[0]).astype('timedelta64[s]').astype(np.float64)/3600/24
            
            m = rdat.average[e][:,jj].squeeze().copy()
            s = rdat.sigma_of_the_mean[e][:,jj].squeeze()

            valid_idx = np.where(~np.isnan(m))[0]
            if np.isnan(m).any():
                tr = tr[valid_idx]
                m = m[valid_idx]
                s = s[valid_idx]  
                tr_day = tr_day[valid_idx]
                ts_day = ts_day[(ts>=tr[0]) & (ts<=tr[-1])]
                ts = ts[(ts>=tr[0]) & (ts<=tr[-1])]
                

            m += (1-m[0]) # shift up so that all start at 1.0 for visual comparison
            error_two_sigma = 1.96*s

            if np.ndim(ax) == 1:
                ax = np.vstack(ax)  # create a fictious 2D array with only one column
            ax[jj,ii].errorbar(tr_day,m,yerr=error_two_sigma)

            ym = r0*mod.helios.soiling_factor[e][jj,rdat.prediction_indices[e][valid_idx[0]]:rdat.prediction_indices[e][-1]+1] # simulate reflectance losses between valid prediction indices
            # THE BELOW SIMPLY RESETS TO 0 THE CONFIDENCE INTERVAL IF A MEASUREMENT IS PERFORMED LATER - CHECK IF IT WORKS AS INTENDED BY MIKE
            var_predict = mod.helios.soiling_factor_prediction_variance[e][jj,rdat.prediction_indices[e][valid_idx[0]]:rdat.prediction_indices[e][-1]+1]-mod.helios.soiling_factor_prediction_variance[e][jj,rdat.prediction_indices[e][valid_idx[0]]]
           
            
            exp_output[t] = {
                'Reflectance': ym.copy(),
                'Variance': var_predict.copy(),
                'Time': ts.copy()}

            if ym.ndim == 1:
                ym += (1.0-ym[0])
            else:
                ym += (1.0-ym[:,0])
            
            sigma_predict = r0*np.sqrt(var_predict)
            Lp = ym - 1.96*sigma_predict
            Up = ym + 1.96*sigma_predict
            ax[jj,ii].plot(ts_day,ym,label='Prediction Mean',color='black')
            ax[jj,ii].fill_between(ts_day,Lp,Up,color='black',alpha=0.1,label=r'Prediction Interval')
            ax[jj,ii].grid('on')

            if jj==0:
                ax[jj,ii].set_title(f"Campaign {e+1}, Heliostat {t}")
            else:
                ax[jj,ii].set_title(f"Heliostat {t}")
        ref_output.append(exp_output)
            
    
        new_var = sdat.dust_concentration[e][0:rdat.prediction_indices[e][-1]+1]
        dust_conc = new_var
        ws = sdat.wind_speed[e][0:rdat.prediction_indices[e][-1]+1]
        dust_type = sdat.dust_type[e][0:rdat.prediction_indices[e][-1]+1]
        if plot_rh:
            a2 = ax[-2,ii]
        else:
            a2 = ax[-1,ii]

        a2.plot(ts_day,dust_conc, color='black')
        a2.tick_params(axis ='y', labelcolor = 'black')
        a2.set_ylim((0,1.01*dust_max))

        a2a = a2.twinx()
        p = a2a.plot(ts_day,ws,color='green')
        a2a.tick_params(axis ='y', labelcolor = 'green')
        a2a.set_ylim((0,1.01*ws_max))
        a2a.set_yticks((0,ws_max/2,ws_max))
        a2a.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        if plot_rh:
            # THE LINE BELOW AVOID THE LAST ELEMENT (SO IT HAS THE SAME DIMENSION)
            rel_hum = sdat.relative_humidity[e][0:rdat.prediction_indices[e][-1]+1]
            a3 = ax[-1,ii]
            a3.plot(ts_day,rel_hum, color='blue')
            a3.tick_params(axis ='y', labelcolor = 'blue')
            a3.set_ylim((0,1.01*hum_max))
        
        if ii==0:
            fs = r"{0:s} $\frac{{\mu g}}{{m^3}}$"
            a2.set_ylabel(fs.format(dust_type),color='black')

            if plot_rh:
                a3.set_ylabel("Relative \nHumidity (%)",color='blue')
        else:
            a2.set_yticklabels([])

        if ii==len(exps)-1:
            fs = r"{0:s} $\frac{{\mu g}}{{m^3}}$"
            a2a.set_ylabel('Wind Speed (m/s)', color='green')

    for ii,row in enumerate(ax):
        for jj,a in enumerate(row):
            if ii < len(hels):
                if yticks is None:
                    a.set_ylim((0.95,1.01))
                    a.set_yticks((0.95,0.97,0.99, 1.01))
                else:
                    a.set_ylim((min(yticks),max(yticks)))
                    a.set_yticks(yticks)

                if jj > 0:
                    a.set_yticklabels([])
                if (ii in rows_with_legend) and (jj==0):
                    ang = rdat.reflectometer_incidence_angle[jj]
                    a.set_ylabel(r"Normalized reflectance $\rho(0)-\rho(t)$ at "+str(ang)+"$^{{\circ}}$")
                if (ii in rows_with_legend) and (jj==0):
                    # a.legend(loc='center',ncol=2,bbox_to_anchor=(0.25,0.5))
                    h_legend,labels_legend = a.get_legend_handles_labels()
            elif ii == len(hels):
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
    return fig,ax,ref_output

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
    
    assert isinstance(imodel,smf.constant_mean_deposition), "Model in saved file must be constant-mean type."

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
    # This assumes a horizontal reflector

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

def fit_quality_plots(mod,rdat,files,mirrors,ax=None,min_loss=None,max_loss=None,include_fits=True,data_ls='b.',data_label="Data",replot=True,vertical_adjust=0,cumulative=False):
    pi = rdat.prediction_indices
    meas = rdat.average
    r0 = rdat.rho0
    sf = mod.helios.soiling_factor
    y = []
    y_hat = []
    if min_loss is None:
        min_loss = 0
    if max_loss is None:
        max_loss = 0
    if ax is None:
        fig,ax = plt.subplots()

    for ii,f in enumerate(files):
        mm = meas[f][:,mirrors]
        sfm = sf[f][mirrors,:]
        if cumulative:
            print(f)
            cumulative_loss = 100*(r0[f][mirrors]-mm)
            cumulative_loss -= cumulative_loss[0,:]
            cumulative_loss_prediction = 100*r0[f][mirrors][:,np.newaxis]*(1-sfm[:,pi[f]])
            cumulative_loss_prediction -= cumulative_loss_prediction[:,0][:,np.newaxis]
            cumulative_loss_prediction = cumulative_loss_prediction.transpose()
            y += [cumulative_loss.flatten()]
            y_hat += [cumulative_loss_prediction.flatten()]
        else:
            delta_loss = -100*np.diff(mm,axis=0).flatten()
            delta_rho_prediction = 100*r0[f][mirrors]*sfm[:,pi[f]].transpose()
            mu_delta_loss = -np.diff(delta_rho_prediction,axis=0).flatten()
            y += [delta_loss]
            y_hat += [mu_delta_loss]

        min_loss = np.min([min_loss,np.min(y[-1]),np.min(y_hat[-1])])
        max_loss = np.max([max_loss,np.max(y[-1]),np.max(y_hat[-1])])

    y_flat = np.concatenate(y,axis=0)
    y_hat_flat = np.concatenate(y_hat,axis=0)
    rmse = np.sqrt( np.mean( (y_flat-y_hat_flat)**2 ) )
    ax.plot(y_flat,y_hat_flat,data_ls,label=data_label+f", RMSE={rmse:.3f}")
    if include_fits:
        R = np.corrcoef(y_flat,y_hat_flat)[0,1]
        p = np.polyfit(y_flat,y_hat_flat,deg=1)

    
    w = max_loss - min_loss
    print(f"min_loss={min_loss},max_loss={max_loss}")
    min_loss -= 0.1*w
    max_loss += 0.1*w
    if replot:
        ax.plot([min_loss,max_loss],[min_loss,max_loss],'k-',label="Ideal")
    ax.set_ylim((min_loss,max_loss))
    ax.set_xlim((min_loss,max_loss))
    ax.set_box_aspect(1)

    if include_fits:
        linear_fit_values = p[1] + p[0]*np.array([min_loss,max_loss])
        ax.plot([min_loss,max_loss],linear_fit_values ,'r:',label=data_label+"_fit")
        ax.annotate(fr'$R$={R:.2f}', xy=(0.05, 0.92+vertical_adjust), xycoords='axes fraction')
        ax.set_ylabel(f"Predicted={p[0]:.2f}*measured + {p[1]:.2f}",fontsize=12)
        ax.legend(loc='lower right')
    else:
        ax.set_ylabel(f"Predicted",fontsize=12)  
        ax.legend(loc='best')
    ax.set_xlabel(r'Measured',fontsize=12)
    plt.tight_layout()
    
    
    if ax is None:
        return fig,ax
       
def summarize_fit_quality(model,ref,train_experiments,train_mirrors,
                            test_mirrors,test_experiments,min_loss,max_loss,
                            save_file,figsize=(8,6),include_fits=True):

    fig,ax = plt.subplots(nrows=1,ncols=3,
                          sharex=True,sharey=True,
                          figsize=figsize)
    print('1')
    fit_quality_plots(model,
                    ref,
                    train_experiments,
                    train_mirrors,
                    ax=ax[0],
                    min_loss=min_loss,
                    max_loss=max_loss,
                    include_fits=include_fits)
    
    print('2')
    fit_quality_plots(model,
                    ref,
                    train_experiments,
                    test_mirrors,
                    ax=ax[1],
                    min_loss=min_loss,
                    max_loss=max_loss,
                    include_fits=include_fits)

    print('3')
    fit_quality_plots(model,
                        ref,
                        test_experiments,
                        train_mirrors+test_mirrors,
                        ax=ax[2],
                        min_loss=min_loss,
                        max_loss=max_loss,
                        include_fits=include_fits)

    ax[0].set_title('Training mirror(s)',fontsize=14)
    ax[1].set_title('Test mirror(s) \n (training interval)',fontsize=16)
    ax[2].set_title('Test experiments\n (all tilts)',fontsize=16)
    fig.tight_layout(pad=5)
    fig.savefig(save_file+"_fit_quality.pdf",bbox_inches='tight')

    return fig,ax

def daily_soiling_tilt_all_data( sim_dat: smb.simulation_inputs,
                                    model_save_file: str,
                                    M: int = 1000,
                                    dust_type="TSP",
                                    tilt:float = None,
                                    trim_percents = None):

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

    if trim_percents is not None:
        xl,xu = np.percentile(daily_sum_alpha,trim_percents,axis=None)
        mask = ((daily_sum_alpha>=xl) & (daily_sum_alpha<=xu))
        daily_sum_alpha = daily_sum_alpha[mask]
        daily_sum_alpha2 = daily_sum_alpha2[mask]

    if tilt is not None:
        daily_sum_alpha = daily_sum_alpha*np.cos(np.radians(tilt))
        daily_sum_alpha2 = daily_sum_alpha2*np.cos(np.radians(tilt))**2

    # simulate
    sims = soiling_rate(daily_sum_alpha,
                        daily_sum_alpha2,
                        model_save_file,
                        M=M)

    
    return sims,daily_sum_alpha,daily_sum_alpha2


