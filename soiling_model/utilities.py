import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps
import scipy.optimize as spo
import numpy as np
from copy import deepcopy
from openpyxl import load_workbook
import miepython

def _print_if(s,verbose):
    # Helper function to control level of output display.
    if verbose:
        print(s)

def _ensure_list(s):
    if not isinstance(s,list):
        s = [s]
    return s

def _check_keys(simulation_data,reflectance_data):
    for ii in range(len(simulation_data.time.keys())):
            if simulation_data.file_name[ii] != reflectance_data.file_name[ii]:
                raise ValueError("Filenames in simulation data and reflectance do not match. Please ensure you imported the same list of files for both.")

def simple_annual_cleaning_schedule(n_sectors,n_trucks,n_cleans,dt=1,n_sectors_per_truck=1):
    T_days = 365
    n_hours = int(T_days*(24/dt)) # number of hours between cleanings
    clean_interval = np.floor(T_days/n_cleans)
    min_clean_interval = np.ceil(n_sectors/n_trucks/n_sectors_per_truck)
    if clean_interval < min_clean_interval:
        clean_interval = min_clean_interval
        n_cleans = int(np.floor(T_days/clean_interval))
        print("Warning: Cannot clean that many times. Setting number of cleans = "+str(n_cleans))

    # evenly space cleaning ends
    clean_ends = np.linspace(0,n_hours-1,num=n_cleans+1,dtype=int)
    clean_ends = np.delete(clean_ends,-1) # remove the last clean since (clean at 0 takes care of this) 
    
    # shift schedule
    cleans = np.zeros((n_sectors,n_hours))
    for ii in range(n_trucks*n_sectors_per_truck,n_sectors,n_trucks*n_sectors_per_truck):
        idx0 = n_sectors-ii
        idx1 = n_sectors-(ii-n_trucks*n_sectors_per_truck)
        idx_col = clean_ends-(24/dt)*(int(ii/n_trucks/n_sectors_per_truck)-1)
        for jj in idx_col.astype(int):
            if jj<0:
                cc = jj + 365*24
                cleans[idx0:idx1,cc] = 1
            else:
                cleans[idx0:idx1,jj] = 1

    # take care of remainder (first day of a field clean)
    if idx0 != 0:
        idx_col = clean_ends-(24/dt)*int(ii/n_trucks/n_sectors_per_truck)
        for jj in idx_col.astype(int):
            if jj<0:
                cc = jj + 365*24
                cleans[0:idx0,cc] = 1
            else:
                cleans[0:idx0,jj] = 1
    return cleans

def plot_experiment_data(simulation_inputs,reflectance_data,experiment_index):
    sim_data = simulation_inputs
    reflect_data = reflectance_data
    f = experiment_index

    fig,ax = plt.subplots(nrows=4,sharex=True)
    fmt = r"${0:s}^\circ$"
    ave = reflect_data.average[f]
    t = reflect_data.times[f]
    std = reflect_data.sigma[f]
    names = ["M"+str(ii+1) for ii in range(ave.shape[1])]
    for ii in range(ave.shape[1]):
        ax[0].errorbar(t,ave[:,ii],yerr=1.96*std[:,ii],label=fmt.format(names[ii]),marker='o',capsize=4.0)

    ax[0].grid(True) 
    label_str = r"Reflectance at {0:.1f} $^{{\circ}}$".format(reflect_data.reflectometer_incidence_angle[f]) 
    ax[0].set_ylabel(label_str)
    ax[0].legend()

    ax[1].plot(sim_data.time[f],sim_data.dust_concentration[f],color='brown',label="measurements")
    ax[1].axhline(y=sim_data.dust_concentration[f].mean(),color='brown',ls='--',label = "Average")
    label_str = r'{0:s} [$\mu g\,/\,m^3$]'.format(sim_data.dust_type[0])
    ax[1].set_ylabel(label_str,color='brown')
    ax[1].tick_params(axis='y', labelcolor='brown')
    ax[1].grid(True)
    ax[1].legend()

    # Rain intensity, if available
    if len(sim_data.rain_intensity)>0: # rain intensity is not an empty dict
        ax[2].plot(sim_data.time[f],sim_data.rain_intensity[f])
    else:
        rain_nan = np.nan*np.ones(sim_data.time[f].shape)
        ax[2].plot(sim_data.time[f],rain_nan)
    
    ax[2].set_ylabel(r'Rain [mm/hour]',color='blue')
    ax[2].tick_params(axis='y', labelcolor='blue')
    YL = ax[2].get_ylim()
    ax[2].set_ylim((0,YL[1]))
    ax[2].grid(True)

    ax[3].plot(sim_data.time[f],sim_data.wind_speed[f],color='green',label="measurements")
    ax[3].axhline(y=sim_data.wind_speed[f].mean(),color='green',ls='--',label = "Average")
    label_str = r'Wind Speed [$m\,/\,s$]'
    ax[3].set_ylabel(label_str,color='green')
    ax[3].set_xlabel('Date')
    ax[3].tick_params(axis='y', labelcolor='green')
    ax[3].grid(True)
    ax[3].legend()

    return fig,ax

def trim_experiment_data(simulation_inputs,reflectance_data,trim_ranges):
    sim_dat = deepcopy(simulation_inputs)
    ref_dat = deepcopy(reflectance_data)
    files = sim_dat.time.keys()

    for f in files:
        if isinstance(trim_ranges,list):  
            assert isinstance(trim_ranges[f],list) or isinstance(trim_ranges[f],np.ndarray), "trim_ranges must be a list of lists or a list of 1D np.arrays"
            lb = trim_ranges[f][0]
            ub = trim_ranges[f][1]
        elif trim_ranges=="reflectance_data":
            assert ref_dat is not None, "Reflectance data must be supplied for trim_ranges==""reflectance_data"""
            lb = ref_dat.times[f][0]
            ub = ref_dat.times[f][-1]
        elif trim_ranges == "simulation_inputs":
            lb = sim_dat.times[f][0]
            ub = sim_dat.times[f][-1]
        else:
            raise ValueError("""Value of trim_ranges not recognized. Must be a list of lists/np.array [lb,ub], """+\
                """ "reflectance_data" or "simulation_inputs" """)

        # trim simulation data
        mask = (sim_dat.time[f]>=lb) & (sim_dat.time[f]<=ub)
        if all(mask==0):
            raise ValueError(f"Provided date range of {lb} to {ub} for file {sim_dat.file_name[f]} excludes all data.")
            
        sim_dat.time[f] = sim_dat.time[f][mask]
        sim_dat.time_diff[f] = sim_dat.time_diff[f][mask]
        sim_dat.air_temp[f] = sim_dat.air_temp[f][mask]
        sim_dat.wind_speed[f] = sim_dat.wind_speed[f][mask]
        sim_dat.dust_concentration[f] = sim_dat.dust_concentration[f][mask]
        if len(sim_dat.rain_intensity)>0:
            sim_dat.rain_intensity[f] = sim_dat.rain_intensity[f][mask]
        if len(sim_dat.dni)>0:
            sim_dat.dni[f] = sim_dat.dni[f][mask]
        if len(sim_dat.relative_humidity)>0:
            sim_dat.relative_humidity[f] = sim_dat.relative_humidity[f][mask]
        
        if reflectance_data is not None:
            # trim reflectance data
            if len(ref_dat.tilts)>0:
                ref_dat.tilts[f] = ref_dat.tilts[f][:,mask]
            mask = (ref_dat.times[f]>=lb) & (ref_dat.times[f]<=ub)
            ref_dat.times[f] = ref_dat.times[f][mask] 
            ref_dat.average[f] = ref_dat.average[f][mask,:]
            ref_dat.sigma[f] = ref_dat.sigma[f][mask,:]
            ref_dat.sigma_of_the_mean[f] = ref_dat.sigma_of_the_mean[f][mask,:]

            ref_dat.prediction_indices[f] = []
            ref_dat.prediction_times[f] = []
            time_grid = sim_dat.time[f]
            for m in ref_dat.times[f]:
                ref_dat.prediction_indices[f].append(np.argmin(np.abs(m-time_grid)))        
                ref_dat.prediction_times[f].append(time_grid[ref_dat.prediction_indices[f]])
                ref_dat.rho0[f] = ref_dat.average[f][0,:]
    
    return sim_dat,ref_dat

def sample_simulation_inputs(historical_files,window=np.timedelta64(30,"D"),N_sample_years=10,sheet_name=None,\
    output_file_format="sample_{0:d}.xlsx",dt=np.timedelta64(3600,'s'),verbose=True):

    # load in historical data files into a single pandas dataframe
    df = pd.DataFrame()
    for f in historical_files:
        fi = pd.read_excel(f,sheet_name=sheet_name)

        #check that time difference is equal to time grid
        if not np.all( fi['Time'].diff()[1::] == dt ): # omit first time, which is NaT
            raise ValueError("Time in file "+f+" is inconsistent with specified dt")

        fi['day'] = fi['Time'].apply(lambda x:x.day) 
        fi['month'] = fi['Time'].apply(lambda x:x.month)
        fi['year'] = fi['Time'].apply(lambda x:x.year)
        df = pd.concat((df,fi),ignore_index=True)

    # Create N_sample_years by sampling days from the historical dataset around a windwow
    t0 = pd.Timestamp( np.datetime64('now').astype('datetime64[Y]').astype('datetime64[m]') ) # t0 is the beginning of the current year
    tf = pd.Timestamp( t0 + np.timedelta64(365,'D') ) # t0 is the beginning of the current year

    dt_str = str(dt.astype('timedelta64[s]').astype('int'))+'s'
    time_grid = pd.date_range(start=t0,end=tf,freq=dt_str)
    day_grid = pd.date_range(start=t0,end=tf,freq="D")
    
    for n in range(N_sample_years):
        samples = pd.DataFrame(columns=df.columns)
        _print_if("Building sample {0:d} of {1:d}".format(n+1,N_sample_years),verbose)
        for ii in range(len(day_grid)-1):
            t = day_grid[ii]

            # samples days in the window
            sample_days = pd.date_range(start=t-window/2,end=t+window/2,freq="D")
            idx = np.random.randint(0,high=len(sample_days))
            sample_day = sample_days[idx]

            # select a random year of the selected day
            mask = (df.day==sample_day.day) & (df.month==sample_day.month)
            sample_years = np.unique(df.year[mask])
            idx_y = np.random.randint(0,high=len(sample_years))
            sample_year = sample_years[idx_y]

            # select 24-hour period that corresponds to the sampled dat in the historical database
            sample_mask = (df.year==sample_year) & mask

            samples = pd.concat((samples,df[sample_mask]),ignore_index=True)

        samples['Time'] = time_grid[0:-1]
        samples.set_index('Time',inplace=True)
        samples.drop(labels=['day','month','year'],axis=1,inplace=True)
        samples.to_excel(output_file_format.format(n),sheet_name=sheet_name)

def _extinction_function(diameters,lambdas,intensities,acceptance_angle,refractive_index,grid_size_mu=int(1e4),grid_size_x=1000,verbose=False):
    
    # theta_s = np.radians(np.linspace(-180,180,grid_size_theta_s)) # angle of scattering (\theta=0 is direction of radiation)
    m = refractive_index
    lam = lambdas/1000 # nm -> µm
    E = intensities*1000 #W/m^2/nm -> W/m^2/µm

    # set up grids
    # mu = np.sort(np.cos(theta_s)) 
    aa_cos = np.cos(acceptance_angle)
    mu = np.linspace(-1,aa_cos,num=grid_size_mu)
    _print_if(f"\t Acceptance angle cosine = {aa_cos:.6f}",verbose)

    # making lookup table in x
    min_x = np.pi*np.min(diameters)/np.max(lam)
    max_x = np.pi*np.max(diameters)/np.min(lam)
    xg = np.logspace(np.log10(min_x),np.log10(max_x),grid_size_x)
    Qxg = np.zeros(xg.shape)
    for ii,x in enumerate(xg):
        scat = miepython.i_unpolarized(m,x,mu,'qext')
        Qxg[ii] = np.trapz(scat,mu)
    
    # apply look up table to data
    Qx = np.zeros((len(diameters),len(lam)))
    for ii,d in enumerate(diameters):
        for jj,lamjj in enumerate(lam):
            x = np.pi*d/lamjj
            Qx[ii,jj] = np.interp(x,xg,Qxg)
    gamma = 2*np.pi*np.trapz(Qx*E,x=lam,axis=1) # for unit irradiance
    
    return gamma

def _same_ext_coeff(helios,simulation_data):
    
    sim_dat = simulation_data
    dust = sim_dat.dust
    D = dust.D
    refractive_index = dust.m
    lam = sim_dat.source_wavelength
    intensities = sim_dat.source_normalized_intensity
    phia = helios.acceptance_angles

    files = list(sim_dat.file_name.keys())
    num_heliostats = [helios.tilt[f].shape[0] for f in files]
    same_dust = np.zeros((len(files),len(files)))
    same_ext = [ [[] for n in range(num_heliostats[f])] for f in files]
                
    for ii,f in enumerate(files):
        for jj,g in enumerate(files):
            if len(D[f]) == len(D[g]):
                same_diameters =  np.all( D[f] == D[g])
            else:
                same_diameters = False

            if len(lam[f]) == len(lam[g]):
                same_lams =  np.all(lam[f] == sim_dat.source_wavelength[f])
            else:
                same_lams = False

            if len(intensities[f]) == len(intensities[g]):
                same_intensity =  np.all(intensities[f] == intensities[g])
            
            same_ref_ind =  (refractive_index[f] == refractive_index[g])
            same_dust[ii,jj] = same_diameters and same_lams \
                and same_intensity and same_ref_ind

    for ii,f in enumerate(files):
        for jj in range(num_heliostats[f]):
            for kk,g in enumerate(files):
                if same_dust[ii,kk]:
                    a = phia[f][jj]
                    idx = [(g,mm) for mm,pg in enumerate(phia[g]) if pg==a]
                    same_ext[ii][jj].extend(idx)
                
    return same_ext

class DustDistribution():
    """
        This needs a docstring :(
    """
    def __init__(self,params=None,type=None):
        
        self.n_components = None
        self.sub_dists = []
        self.weights = []
        self.type = []
        self.units = []
        if params is not None:
            assert type.lower() in ["mass","number","area"], "Please supply a type (mass, number, area)"
            N = len(params)/3
            assert np.abs(N-np.floor(N))<np.finfo(float).eps, \
                "Please specify parameters of each component as a 1D numpy.array([weights,mu,sigma])."
            
            N = int(np.floor(N))
            w,mu,sig = params[0:N],params[N:2*N],params[2*N::]
            self.n_components = N
            self.sub_dists = [sps.norm(loc=mu[ii],scale=sig[ii]) for ii in range(N)]
            self.weights = w
            self.type = type
            self._set_units()

    def pdf(self,x):
        pdf = 0
        for ii in range(self.n_components):
            pdf += self.weights[ii]*self.sub_dists[ii].pdf(x)
        return pdf
    
    def cdf(self,x):
        cdf = 0
        for ii in range(self.n_components):
            cdf += self.weights[ii]*self.sub_dists[ii].cdf(x)
        return cdf
    
    def mean(self):
        m = 0
        sum_weights = sum(self.weights)
        for ii in range(self.n_components):
            m += self.weights[ii]/sum_weights*self.sub_dists[ii].mean()
        return m
    
    def icdf(self,p):
        fun = lambda x: self.cdf(x)-p
        res = spo.fsolve(fun,self.mean())
        return res
    
    def _sse(self,params,log_diameter_values,pm_values):

        N = len(params)/3
        assert np.abs(N-np.floor(N))<np.finfo(float).eps, \
            "Please specify parameters of each component as a 1D numpy.array([weights,mu,sigma])."
        N = int(np.floor(N))
        w,mu,sig = params[0:N],params[N:2*N],params[2*N::]
        self.n_components = N
        self.sub_dists = [sps.norm(loc=mu[ii],scale=sig[ii]) for ii in range(N)]
        self.weights = w

        return np.sum( (self.cdf(log_diameter_values)-pm_values)**2 )
    
    def fit(self,params0,log_diameter_values,pm_values,tol=1e-3):

        N = len(params0)/3
        assert np.abs(N-np.floor(N))<np.finfo(float).eps, \
            "Please specify parameters of each component as a 1D numpy.array([weights,mu,sigma])."
        N = int(np.floor(N))

        fun = lambda x: self._sse(x,log_diameter_values,pm_values)
        
        # construct bounds
        lower_bound_w = [0]*N
        lower_bound_mu = [-np.inf]*N
        lower_bound_sig = [0+tol]*N
        lb = lower_bound_w + lower_bound_mu + lower_bound_sig # join lists
        ub = [np.inf]*len(lb)

        bnds = spo.Bounds(lb=lb,ub=ub,keep_feasible=True)
        res = spo.minimize(fun,params0,bounds=bnds,tol=1e-8)

        params = res.x
        N = int(np.floor(len(params)/3))
        w,mu,sig = params[0:N],params[N:2*N],params[2*N::]
        self.n_components = N
        self.sub_dists = [sps.norm(loc=mu[ii],scale=sig[ii]) for ii in range(N)]
        self.weights = w
        self.type = "mass"
        self._set_units()

        return res

    def _set_units(self):
        if self.type.lower() == "mass":
            self.units = r"$\frac{\mu g \cdot m^{{-3}}}{d(\log(D))}$"
        elif self.type.lower() == "number":
            self.units = r"$\frac{cm^{{-3}} ] }{d(\log(D))}$"
        elif self.type.lower() == "area":
            self.units = r"$\frac{ m^2 \cdot m^{{-3}}}{d(\log(D))}$"

    def convert_to_number(self,rho=None):

        if self.type.lower() == "number":
            print("Type is already ""number"" ")
        elif self.type.lower()=="mass":
            assert isinstance(rho,float), "Particle density must be a scalar float."
            new_subs = []
            new_weights = []
            for ii in range(self.n_components):
                Mi = self.weights[ii]
                mbari = self.sub_dists[ii].mean()
                si = self.sub_dists[ii].std()

                mi = mbari-3*si**2/np.log10(np.e)
                b = 2*mi+6*si**2/np.log10(np.e)
                Ni = 6*Mi/np.pi/rho * np.exp((mi**2-0.25*b**2)/2/si**2)

                new_weights.append(Ni*1e3)
                ns = sps.norm(loc=mi,scale=si)
                new_subs.append(ns)
            
            self.weights = new_weights
            self.sub_dists = new_subs
            self.type = "number"
            self._set_units()
        elif self.type.lower()=="area":
            new_subs = []
            new_weights = []
            for ii in range(self.n_components):
                Ai = self.weights[ii]
                mbari = self.sub_dists[ii].mean()
                si = self.sub_dists[ii].std()

                mi = mbari-2*si**2/np.log10(np.e)
                Ni = Ai/np.pi*4 * np.exp((mi**2-(mi-si**2/np.log10(np.e))**2)/2/si**2)

                new_weights.append(Ni*1e6)
                ns = sps.norm(loc=mi,scale=si)
                new_subs.append(ns)
            
            self.weights = new_weights
            self.sub_dists = new_subs
            self.type = "number"
            self._set_units()

    def convert_to_mass(self,rho):
        assert isinstance(rho,float), "Particle density must be a scalar float."
        if self.type.lower() == "mass":
            print("Type is already ""mass"" ")
        elif self.type.lower() == "number":
            new_subs = []
            new_weights = []
            for ii in range(self.n_components):
                Ni = self.weights[ii]
                mi = self.sub_dists[ii].mean()
                si = self.sub_dists[ii].std()

                b = 2*mi+6*si**2/np.log10(np.e)
                mbari = b/2.0
                Mi = Ni*np.pi*rho/6 * np.exp(-(mi**2-0.25*b**2)/2/si**2)

                new_weights.append(Mi*1e-3)
                ns = sps.norm(loc=mbari,scale=si)
                new_subs.append(ns)
            
            self.weights = new_weights
            self.sub_dists = new_subs
            self.type = "mass"
            self._set_units()
        elif self.type.lower()=="area":
            print("Convert to number first. ")

    def convert_to_area(self,rho=None):

        if self.type.lower() == "area":
            print("Type is already ""area"" ")
        elif self.type.lower() == "number":
            new_subs = []
            new_weights = []
            for ii in range(self.n_components):
                Ni = self.weights[ii]
                mi = self.sub_dists[ii].mean()
                si = self.sub_dists[ii].std()

                mbari = mi+2*si**2/np.log10(np.e)
                Ai = Ni*np.pi/4*np.exp(-(mi**2-(mi+si**2/np.log10(np.e))**2)/2/si**2)

                new_weights.append(Ai*1e-6)
                ns = sps.norm(loc=mbari,scale=si)
                new_subs.append(ns)
            
            self.weights = new_weights
            self.sub_dists = new_subs
            self.type = "area"
            self._set_units()
        elif self.type.lower() == "mass":
            if rho is None:
                raise ValueError("Rho cannot be None to convert from mass")
            
            assert isinstance(rho,float), "Particle density must be a scalar float."
            self.convert_to_number(rho)
            self.convert_to_area()

    def write_to_file(self,file_name,sheet_name,kind='number',rho=None,verbose=True):
        _print_if("Writing dust distribution to file "+file_name,verbose)
        
        # ensure kind is correct
        if kind.lower() == 'number':
            self.convert_to_number()
        elif kind.lower() == 'mass':
            self.convert_to_mass()
        elif kind.lower() == 'area':
            self.convert_to_area()
        else:
            raise ValueError("kind not recognized.")

        # convert to strings and join with ";" delimiter
        weight_str = [str(s)+";" for s in self.weights]
        mu_str = [str(10**(self.sub_dists[ii].mean())) for ii in range(self.n_components)]
        sig_str = [str(10**(self.sub_dists[ii].std())) for ii in range(self.n_components)]
        weight_str = "".join(weight_str)[0:-1]
        mu_str = "".join(mu_str)[0:-1]
        sig_str = "".join(sig_str)[0:-1]

        # write data
        wb = load_workbook(file_name)
        ws = wb[sheet_name]
        for cell in ws['A']:
            if cell.value == "N_size":
                ws.cell(row=cell.row, column=2).value = self.n_components
                ws.cell(row=cell.row, column=4).value = ""
            elif cell.value == "Nd":
                ws.cell(row=cell.row, column=2).value = weight_str
                ws.cell(row=cell.row, column=4).value = ""
            elif cell.value == "mu":
                ws.cell(row=cell.row, column=2).value = mu_str
                ws.cell(row=cell.row, column=4).value = ""
            elif cell.value == "sigma":
                ws.cell(row=cell.row, column=2).value = sig_str
                ws.cell(row=cell.row, column=4).value = ""
        
        wb.save(filename=file_name)
        wb.close()