import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import pymcmcstat
from pymcmcstat.MCMC import MCMC
from pymcmcstat.plotting import MCMCPlotting
# Importing into dataframe
df = pd.read_csv('ReactorData.csv')

# grabbing units in second row and appending to the columns header
new_header = []  
units = df.iloc[0] 
for i,j in enumerate(df.columns):
    new_header.append(j + ' ' + units[i])
    
df.columns = new_header
df = df.drop(0) # drop the units row 

df = df.astype('float64')
df['F0 (m3/s)'] = df['F0 (m3/hr)']*(1/3600) 

# Creating spline functions 

order = 1
ex = True
time = df['t (s)']
CA0 = BSpline(time,df.loc[::1,'C0 (mol/m3)'],order,extrapolate=ex)
h = BSpline(time,df.loc[::1,'h (m)'],order,extrapolate=ex)
Temp = BSpline(time[::1],df.loc[::1,'T (K)'],order,extrapolate=ex)
F0 = BSpline(time,df.loc[::1,'F0 (m3/s)'],order,extrapolate=ex)
CA = BSpline(time,df.loc[::1,'C (mol/m3)'],order,extrapolate=ex)

global Ac, R
D = 0.2 # Diameter of reactor [m]
R = 8.314 # J/mol.K 
Ac = np.pi * (D/2)**2 # Cross section area of reaction [m2]


def cstr_model(y,t,parameters): 
    k,C,Ea,Tin,delH,rho,Cp,Ua,Fc,Tc_in,rhoc,Cpc,use = parameters
	
    if use == 'ODE': # When using integrator to solve
        Ca, T = y
    else: # When using sum of squares to fit
        Ca = CA(t)
        T = Temp(t)

    F  = C*h(t) # Flow out [m3/s] is equal to C*height(t) - spline function
    V = Ac*h(t) # Volume of reactor [m3]

    # Assumed that the jacket temperature time constant is negligible 
    # i.e. Tc/dt = 0
    Tc = (Fc*rhoc*Cpc*Tc_in + Ua*T)/(Ua + Fc*rhoc*Cpc) # Temperature of coolant [K] 
    # Fc [m3/s], rhoc [kg/m3], Cpc [J/kg.K], Tc_in [K], Ua[W/K], T(t) [K]
    # First order rate expression
    r_A = -k*np.exp(Ea/(R*T))*Ca
    # k [1/s], Ea [J/mol], R [J/mol.K], CA(t) [mol/m3]
	
    # Component Balance
    dotC = (F0(t)*CA0(t) / V) - (F*Ca / V) + r_A # [mol/m3.s]
    # F0(t) [m3/s], CA0(t) [mol/m3], V [m3], F [mol/s], r_A [mol/m3.s]

    # Engery Balance
    dotT = (Tin*F0(t) / V) - (T*F / V) + (delH*r_A / rho*Cp) - (Ua*(T-Tc) / V*rho*Cp) # [K/s]
    # Tin [K], delH [J/mol], rho [kg/m3], Cp [J/kg.K]

    dotY = [dotC, dotT]
    return dotY

# Sum of squares function
def cstr_ss(x,data): 
    k,C,Ea,Tin,delH,rho,Cp,Ua,Fc,Tc_in,rhoc,Cpc = x
    use = '' 
    x = [k,C,Ea,Tin,delH,rho,Cp,Ua,Fc,Tc_in,rhoc,Cpc,use]
    
    t = time
    dotX_model = cstr_model([],t,x)
    dotX_model = np.array(dotX_model)
    '''
    fitting the ode to the derivatives of the spline functions because its less
    computationally costly than solving for ode each time and then fitting
    '''
    dotC = CA.derivative()
    dotT = Temp.derivative()
    derivatives = np.array([dotC(t),dotT(t)])
    res = dotX_model - derivatives
    ss = (res**2).sum(axis=1)
    return ss

# Markov Chain Monte Carlo - DRAM - Delayed rejection adaptive metropolis method 
# DRAM technique tunes the posterior distribution of the parameters which eliminates tedious guess work
# The objective funciton to be sampled is the sum of squares function 

mcstat_full = MCMC() # initiate MCMC 

# data is called from the spline functions - not passed through here
mcstat_full.data.add_data_set(x=[],
                         y=[]) 

# Setting up the parameters to be tuned. 
# Initial guesses (theta0) - obtained from literature / thermodynamics
# Some boundaries are setup up


mcstat_full.parameters.add_model_parameter(name='k',theta0=0.0003,minimum=0,prior_sigma=0.0001)
mcstat_full.parameters.add_model_parameter(name='C',theta0=0.0002,minimum=0,prior_sigma=0.0001)
mcstat_full.parameters.add_model_parameter(name='Ea',theta0=5000,minimum=0,prior_sigma=100)
mcstat_full.parameters.add_model_parameter(name='Tin',theta0=340,minimum=280,prior_sigma=1)
mcstat_full.parameters.add_model_parameter(name='delH',theta0=-100000,prior_sigma=100)
mcstat_full.parameters.add_model_parameter(name='rho',theta0=1000,minimum=0,prior_sigma=10)
mcstat_full.parameters.add_model_parameter(name='Cp',theta0=2000,minimum=0,prior_sigma=10)
mcstat_full.parameters.add_model_parameter(name='Ua',theta0=50,minimum=0,prior_sigma=1)
mcstat_full.parameters.add_model_parameter(name='Fc',theta0=0.01,minimum=0,prior_sigma=0.0001)
mcstat_full.parameters.add_model_parameter(name='Tc_in',theta0=298,minimum=280,prior_sigma=1)
mcstat_full.parameters.add_model_parameter(name='rhoc',theta0=1000,minimum=0,prior_sigma=100)
mcstat_full.parameters.add_model_parameter(name='Cpc',theta0=4000,minimum=0,prior_sigma=10)


# Simulation options - selecting DRAM and number of simulations
mcstat_full.simulation_options.define_simulation_options(nsimu=20000, updatesigma=True,method='dram')
# Specifying the function to be sampled
mcstat_full.model_settings.define_model_settings(
sos_function = cstr_ss)
mcstat_full.run_simulation()

results = mcstat_full.simulation_results.results
chain = results['chain'][15000:20000,:]
names = results['names']
mcstat_full.chainstats(chain, results)
var = list(results['theta'])
var.append('ODE')
var = tuple(var)

#
Ca = odeint(cstr_model,[CA(time)[0],Temp(time)[0]],time,args=(var,))