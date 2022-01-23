# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:50:47 2020

@author: Stefano
"""

import scipy as sp
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import matplotlib as mat
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint
#%% 
def f (y,t,ps): #for Odeint first comes the functions' vector y then the time
    """
THE ODE NTH DIMENSIONAL SYSTEM: has arguments
                                             the time array
                                             the nth vector y
                                             the parameters which have to be fitted (the others are defined within the function)
the ODE system function f returns a nth vector with each element being a function (the derivative)
    """
    try:
       # b=ps["b"].value
        ln=ps["ln"].value
        p=ps["p"].value
        a=ps["a"].value
        Tr=ps["Tr"].value
        Trh=ps["Trh"].value
        Tqs=ps["Tqs"].value
        Tqn=ps["Tqn"].value
        lamb=ps["lamb"].value
        bf=ps["bf"].value
    except:
        ln, p, a, Tr, Trh, Tqs, Tqn, lamb, bf =ps
    tau=5.1
    CFR=0.062
    b=y[9]
    Td=11
    lu=(p+(1-p)*ln)/2
#DEFINE THE POSSIBLE STATES OF THE INFECTION 
    IU=y[0]
    IS=y[1]
    IN=y[2]
    ISQ=y[3]
    INQ=y[4]
    D=y[5]
    R1=y[6]
    R2=y[7]
    NS=y[8]
    g=1-a*(IU+IS+IN+ISQ+INQ)/(ISQ+0.001) #gamma
#FINALLY DEFINE THE DIFFERENTIAL EQUATIONS + the infectiousness varying equation
    dIU=b*(IS+ln*IN+lu*IU)*NS/(IU+IS+IN+ISQ+INQ+R1+R2+NS)-IU/tau
    dIS=p*IU/tau-IS/Tr-IS/Tqs
    dIN=(1-p)*IU/tau-IN/Tr-IN/Tqn
    dISQ=IS/Tqs-a*(IU+IS+IN+ISQ+INQ)/Td-g*ISQ/Trh
    dINQ=IN/Tqn-INQ/Trh
    dD=a*(IU+IS+IN+ISQ+INQ)/Td
    dD2=CFR*(ISQ+INQ)/Td                        #the one with the CFR (at the moment unused)
    dR1=g*(ISQ)/Trh+INQ/Trh                     # the registered ricovered
    dR2=IS/Tr+IN/Tr                          #the non registered ricovered
    dNS=-b*(IS+ln*IN+lu*IU)*NS/(IU+IS+IN+ISQ+INQ+R1+R2+NS)
    db=-lamb*(b-bf)
    f=[dIU,dIS,dIN,dISQ,dINQ,dD,dR1,dR2,dNS,db]
    return f

def sol(t, y0, ps):
    """
    SOLUTION TO THE ODEs SYSTEM with initial conditions y(0) = y0
    """
    T=sp.linspace(0,t,t+1)
    solu=odeint(f, y0, T, args=(ps,))# 
    return solu

def residual_TD(ps, t, data): #must return an array
    """
    RESIDUALS FUNCTION between the model with its parameters and the real data:
        it considers the total cases detected (both active and recovered) together 
        and the deaths
    """
    #y0 = ps['y0'].value #as for us the initial conditions don't represent a parameter to be fitted
    y0=ps['IU'].value, ps['IS'].value, ps['IN'].value, ps['ISQ'].value, ps['INQ'].value, ps['D'].value, ps['R1'].value, ps['R2'].value, ps['NS'].value, ps["b0"].value
    model = sol(t, y0, ps)
    #model = solution.y
    ISQ=model[:,3]
    INQ=model[:,4]
    R1=model[:,6]
    D=model[:,5]
    delta=ISQ+INQ+R1+D-data[0]
    rDet=(delta/sp.sqrt(data[0])).ravel()
    rD=((D-data[1])/sp.sqrt(data[1])).ravel()
    Res=sp.sqrt(9*rDet**2+rD**2) #GIVES MORE IMPORTANCE TO THE INFECTED CURVE
    return Res
#%%
"""
SWEDEN
"""
#SWEDEN DOESN'T MAKE AVILABLE THE NUMBER OF ACTIVE CASES AND THE ONE OF THE RECOVERED

#the data for sweden is blocked at the 18th of June worlodmeter, because apparently they stopped counting
TOTCASES_swe=sp.array([1040,1121,1196,1301,1439,1639,1770,1934,2046,2299,2526,2840,3069,3447,3700,4028,4435,4947,5568,6131,6443,6830,7206,7693,8419,9141,9685,10151,10483,10948,11445,11927,12540,13216,13822,14385,14777,15322,16004,16755,17567,18177,18640,18926,19621,20302,21092,21520,22082,22317,22721,23216,23918,24623,25265,25921,26322,26670,27272,27909,28582,29207,29677,30143,30377,30799,31523,32172,32809,33188,33459,33843,34440,35088,35727,36476,37113,37814,38589,40803,41883,42939,43887,44730,45186,45729,46665,48092,49479,49684,50931,51614,52383,53323,54562,56043])
dead_swe=sp.array([3,7,8,10,11,16,20,21,27,40,62,77,105,105,110,146,180,239,308,358,373,401,477,591,687,793,870,887,899,919,1033,1203,1333,1400,1511,1540,1580,1765,1937,2021,2152,2192,2194,2274,2355,2462,2586,2653,2669,2679,2769,2854,2941,3040,3175,3220,3225,3256,3313,3460,3529,3646,3674,3679,3698,3743,3831,3871,3925,3992,3998,4029,4125,4220,4266,4350,4395,4395,4403,4468,4542,4562,4639,4656,4659,4694,4717,4795,4814,4854,4874,4874,4891,4939,5041,5053])
data_swe=[TOTCASES_swe, dead_swe]
t_swe = len(TOTCASES_swe)-1

y0_swe=[1000,400,200,900,140,3,50,200,10100000]# 30 March
#SWEDEN IMPOSED NO LOCKDOWN, ONLY MADE RECOMMENDATIONS TO ITS CITIZEN 
#%%
# set parameters incluing their bounds
params = Parameters()
#params.add('y0', value=float(data[0]), min=0, max=100) #as for us the initial cond don't represent a parameter to be fitted
params.add('IU', value= 1600,min=1000, max=2000, vary=False)
params.add('IS', value= 500,min=250, max=1000, vary=False )
params.add('IN', value= 250,min=100, max=400, vary=False)
params.add('ISQ', value= y0_swe[3],vary=False)
params.add('INQ', value= y0_swe[4],vary=False)
params.add('D', value= y0_swe[5],vary=False)
params.add('R1', value= 30,vary=False)
params.add('R2', value= 200,min=200, max=500, vary=False)
params.add('NS', value= y0_swe[8],vary=False)
params.add('b0', value= 0.237, min=0.235, max=0.27)
params.add("ln",value=0.59, min=0.58, max=0.61)
params.add("p",value=0.81, min=0.795, max=0.82)
params.add('a', value= 0.033, min=0.031, max=0.040)
params.add("Tr",value=9, min=8.5, max=11)
params.add("Trh",value=20.0, min=24, max=30)
params.add("Tqs",value=5.5, min=5, max=7)
params.add("Tqn",value=15, min=10, max=30)
params.add("lamb", value=0.0055, min=0.002, max=0.0057)
params.add("bf", value=0.01244, min=0.01, max=0.02)
#%%
# fit model and find predicted values
result_swe = minimize(residual_TD, params, args=(t_swe, data_swe), method='emcee')
#final_brazil = data_brazil + result_brazil.residual.reshape(data_brazil.shape)
#THIS STEP CANNOT WORK AS THE RESIDUALS, AS WE DEFINED THEM ARE THE SUM OF THE TWO RESIDUALS COMING FROM 
#THE DETECTED INFECTED CURVE AND THE REGISTERED RECOVERED CURVE
# display fitted statistics
report_fit(result_swe)
fit=result_swe.params

"""
NOW it's necessary to evaluate the system with the brand new parameters
"""
def solnew(t,ps):
    y0=ps['IU'].value, ps['IS'].value, ps['IN'].value, ps['ISQ'].value, ps['INQ'].value, ps['D'].value, ps['R1'].value, ps['R2'].value, ps['NS'].value, ps["b0"].value
    T=sp.linspace(0,t,t+1)
    solu=odeint(f, y0, T, args=(ps,))# 
    return solu
final=solnew(t_swe,fit)

IU_swe= final[:,0]
IS_swe= final[:,1]
IN_swe= final[:,2]
ISQ_swe=final[:,3]
INQ_swe=final[:,4]
D_swe=  final[:,5]
R1_swe= final[:,6]
R2_swe= final[:,7]
NS_swe= final[:,8]
b_swe=  final[:,9]
#%%
params.add('IU', value= 1600)
params.add('IS', value= 500)
params.add('IN', value= 250)
params.add('ISQ', value= y0_swe[3])
params.add('INQ', value= y0_swe[4])
params.add('D', value= y0_swe[5])
params.add('R1', value= 30)
params.add('R2', value= 200)
params.add('NS', value= y0_swe[8])
params.add('b0', value= 0.223)
params.add("ln",value=0.59753177)
params.add("p",value=0.81777889)
params.add('a', value= 0.039)
params.add("Tr",value=10.8311755)
params.add("Trh",value=28)
params.add("Tqs",value=5.02306687)
params.add("Tqn",value=10.3012851)
params.add("lamb", value=0.003)
params.add("bf", value=0.01491441)

def solnew(t,ps):
    y0=ps['IU'].value, ps['IS'].value, ps['IN'].value, ps['ISQ'].value, ps['INQ'].value, ps['D'].value, ps['R1'].value, ps['R2'].value, ps['NS'].value, ps["b0"].value
    T=sp.linspace(0,t,t+1)
    solu=odeint(f, y0, T, args=(ps,))# 
    return solu
final=solnew(t_swe,params)

IU_swe= final[:,0]
IS_swe= final[:,1]
IN_swe= final[:,2]
ISQ_swe=final[:,3]
INQ_swe=final[:,4]
D_swe=  final[:,5]
R1_swe= final[:,6]
R2_swe= final[:,7]
NS_swe= final[:,8]
b_swe=  final[:,9]
#%%
# plot data and fitted curves
T=sp.linspace(0,t_swe,t_swe+1)

plt.title("Evolution of COVID-19 pandemic in Sweden")
plt.plot(T, ISQ_swe+INQ_swe, '--', c='Red', label="Estimated Detected Active Cases")
plt.plot(T, R1_swe, '--', c='Cyan',label="Estimated Registered Recovered")
plt.plot(T, data_swe[1], c="Black",label="Official Dead")
plt.plot(T, D_swe, "--", c="Brown",label="Estimated Dead")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()

plt.title("Evolution of COVID-19 pandemic in Sweden")
plt.plot(T, data_swe[0], c="Purple",label="Official Total Detected Cases")
plt.plot(T, ISQ_swe+INQ_swe+R1_swe+D_swe, '--', c='Blue',label="Estimated Total Detected Cases")
plt.plot(T, data_swe[1], c="Black",label="Official Dead")
plt.plot(T, D_swe, "--", c="Brown",label="Estimated Dead")
plt.plot(T, IU_swe+IS_swe+IN_swe+ISQ_swe+INQ_swe+R1_swe+R2_swe+D_swe, "--", c="Red",label="Estimated Total Cases")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()

plt.title("Evolution of COVID-19 pandemic in Sweden")
plt.plot(T, data_swe[0], c="Purple",label="Official Total Detected Cases")
plt.plot(T, ISQ_swe+INQ_swe+R1_swe+D_swe, '--', c='Blue',label="Estimated Total Detected Cases")
plt.plot(T, data_swe[1], c="Black",label="Official Dead")
plt.plot(T, D_swe, "--", c="Brown",label="Estimated Dead")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()

plt.title("Evolution of the Infectiousness in Sweden")
plt.plot(T,b_swe)
plt.xlabel("Time (days)")
plt.grid("major", "both")
plt.show()
#%%
params = Parameters()
#params.add('y0', value=float(data[0]), min=0, max=100) #as for us the initial cond don't represent a parameter to be fitted
params.add('IU', value= 1600)
params.add('IS', value= 500)
params.add('IN', value= 250)
params.add('ISQ', value= y0_swe[3])
params.add('INQ', value= y0_swe[4])
params.add('D', value= y0_swe[5])
params.add('R1', value= 30)
params.add('R2', value= 200)
params.add('NS', value= y0_swe[8])
params.add('b0', value= 0.223)
params.add("ln",value=0.59753177)
params.add("p",value=0.81777889)
params.add('a', value= 0.039)
params.add("Tr",value=10.8311755)
params.add("Trh",value=28)
params.add("Tqs",value=5.02306687)
params.add("Tqn",value=10.3012851)
params.add("lamb", value=0.003)
params.add("bf", value=0.01491441)
def solnew(t,ps):
    y0=ps['IU'].value, ps['IS'].value, ps['IN'].value, ps['ISQ'].value, ps['INQ'].value, ps['D'].value, ps['R1'].value, ps['R2'].value, ps['NS'].value, ps["b0"].value
    T=sp.linspace(0,t,t+1)
    solu=odeint(f, y0, T, args=(ps,))# 
    return solu
final=solnew(t_swe+41,params)
T2=sp.linspace(0,t_swe+41,t_swe+1+41)

IU_swe= final[:,0]
IS_swe= final[:,1]
IN_swe= final[:,2]
ISQ_swe=final[:,3]
INQ_swe=final[:,4]
D_swe=  final[:,5]
R1_swe= final[:,6]
R2_swe= final[:,7]
NS_swe= final[:,8]
b_swe=  final[:,9]
plt.title("Evolution of COVID-19 pandemic in Sweden")
plt.plot(T, data_swe[0], c="Purple",label="Official Total Detected Cases")
plt.plot(T2, ISQ_swe+INQ_swe+R1_swe+D_swe, '--', c='Blue',label="Estimated Total Detected Cases")
plt.plot(T, data_swe[1], c="Black",label="Official Dead")
plt.plot(T2, D_swe, "--", c="Brown",label="Estimated Dead")
plt.plot(T2, IU_swe+IS_swe+IN_swe+ISQ_swe+INQ_swe+R1_swe+R2_swe+D_swe, "--", c="Red",label="Estimated Total Cases")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()

plt.title("Evolution of COVID-19 pandemic in Sweden")
plt.plot(T, data_swe[0], c="Purple",label="Official Total Detected Cases")
plt.plot(T2, ISQ_swe+INQ_swe+R1_swe+D_swe, '--', c='Blue',label="Estimated Total Detected Cases")
plt.plot(T, data_swe[1], c="Black",label="Official Dead")
plt.plot(T2, D_swe, "--", c="Brown",label="Estimated Dead")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()