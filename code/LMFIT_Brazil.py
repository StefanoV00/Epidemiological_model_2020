# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 18:55:39 2020

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
    delta=ISQ+INQ+R1+D-data[0]-data[1]-data[2]
    rDet=(delta/sp.sqrt(data[0]+data[1])).ravel()
    rD=((D-data[2])/sp.sqrt(data[2])).ravel()
    Res=sp.sqrt(9*rDet**2+rD**2) #GIVES MORE IMPORTANCE TO THE INFECTED CURVE
    return Res

def residual_IRD(ps, t, data): #must return an array
    """
    RESIDUALS FUNCTION between the model with its parameters and the real data:
        it considers, more or less separately, the detected infected and the recovered
        it doesn't consider the deaths
    """
    #y0 = ps['y0'].value #as for us the initial conditions don't represent a parameter to be fitted
    y0=ps['IU'].value, ps['IS'].value, ps['IN'].value, ps['ISQ'].value, ps['INQ'].value, ps['D'].value, ps['R1'].value, ps['R2'].value, ps['NS'].value, ps["b0"].value
    model = sol(t, y0, ps)
    #model = solution.y
    ISQ=model[:,3]
    INQ=model[:,4]
    ID=ISQ+INQ
    R1=model[:,6]
    D=model[:,5]
    delta=ID-data[0]
    rI=(delta/sp.sqrt(data[0])).ravel()
    rR=((R1-data[1])/sp.sqrt(data[1])).ravel()
    rD=((D-data[2])/sp.sqrt(data[2])).ravel()
    Res=sp.sqrt(9*rI**2+4*rR**2+rD**2) #GIVES MORE IMPORTANCE TO THE INFECTED CURVE
    return Res
#%%
"""
BRAZIL
"""
y0_brazil=[1422,400,200,1000,158,18,2,250,212413908] #21 March, the first day the detected infected became 1000

infec_brazil=sp.array([1158,1519,1888,2199,2493,2902,3319,3784,4114,4347,5389,6511,7593,8704,9788,10641,11492,13221,15241,17018,18548,19649,20796,21929,9704,12827,14710,17515,20335,14062,16026,16013,17533,19606,21670,25991,28436,30816,35292,39718,43544,47660,48872,51131,55108,58573,66653,71155,76603,83720,86619,90134,92601,97575,109446,118436,127837,130840,138056,147108,157780,164879,174412,182798,190991,199314,209218,222317,218867,247213,264365,278980,288279,284763,285301,306834,322307,338366,353379,347973,377985,355087,367899,391309,380395,386981,393870,418601,410137,431983,469118,476895])
dead_brazil=sp.array([18,25,34,46,59,77,92,114,136,163,201,242,324,363,445,486,564,686,820,954,1068,1140,1223,1328,1532,1757,1947,2141,2361,2462,2587,2741,2906,3313,3670,4045,4271,4543,5063,5511,5901,6410,6750,7025,7343,7921,8588,9188,9992,10656,11123,11625,12404,13158,13993,14817,15633,16118,16853,17983,18894,20082,21048,22013,22716,23522,24549,25697,26764,27944,28834,29314,30046,31278,32547,34039,35047,36044,36499,37912,38497,39497,41058,41901,42791,43389,44118,45456,46665,47869,49090,50058])
TOTCASES_brazil=sp.array([1178,1546,1924,2247,2554,2985,3417,3904,4256,4630,5717,6880,8044,9194,10360,11254,12183,14034,16188,18145,19789,20962,22192,23430,25262,28610,30683,33682,36722,38654,40743,43079,45757,49492,52995,59196,62859,66501,72899,79361,85380,92109,96559,101147,108266,114715,126611,135693,145892,156061,162699,169143,177602,189157,202918,218223,233142,241080,255368,271825,293357,310921,330890,347398,363618,376669,392360,414661,438812,468338,498440,514849,529405,556668,583980,615870,646006,673587,691962,710887,742084,775184,805649,829902,850756,867882,891556,928834,960309,983359,1038569,1070139])

recov_brazil=TOTCASES_brazil-infec_brazil-dead_brazil

data_brazil=sp.array([infec_brazil,recov_brazil,dead_brazil])
t_brazil = len(infec_brazil)-1
#%%
# set parameters incluing their bounds
params = Parameters()
#params.add('y0', value=float(data[0]), min=0, max=100) #as for us the initial cond don't represent a parameter to be fitted
params.add('IU', value= 1400,min=1000, max=2000, vary=False)
params.add('IS', value= 500,min=250, max=1000, vary=False)
params.add('IN', value= 200,min=100, max=400, vary=False)
params.add('ISQ', value= y0_brazil[3],vary=False)
params.add('INQ', value= y0_brazil[4],vary=False)
params.add('D', value= y0_brazil[5],vary=False)
params.add('R1', value= y0_brazil[6],vary=False)
params.add('R2', value= 235,min=200, max=500, vary=False)
params.add('NS', value= y0_brazil[8],vary=False)
params.add('b0', value= 0.35, min=0.347, max=0.38)
params.add("ln",value=0.595, min=0.58, max=0.61)
params.add("p",value=0.8, min=0.79, max=0.82)
params.add('a', value= 0.0153, min=0.014, max=0.018)
params.add("Tr",value=9, min=8.5, max=11)
params.add("Trh",value=19.6, min=18, max=22)
params.add("Tqs",value=7.8, min=7, max=12)
params.add("Tqn",value=40, min=25, max=50)
params.add("lamb", value=0.0099, min=0.008, max=0.013)
params.add("bf", value=0.018, min=0.01, max=0.03)
#%%
# fit model and find predicted values
result_brazil = minimize(residual_IRD, params, args=(t_brazil, data_brazil), method='emcee')
#final_brazil = data_brazil + result_brazil.residual.reshape(data_brazil.shape)
#THIS STEP CANNOT WORK AS THE RESIDUALS, AS WE DEFINED THEM ARE THE SUM OF THE TWO RESIDUALS COMING FROM 
#THE DETECTED INFECTED CURVE AND THE REGISTERED RECOVERED CURVE
# display fitted statistics
report_fit(result_brazil)
fit=result_brazil.params
#%%
"""
NOW it's necessary to evaluate the system with the brand new parameters
"""
def solnew(t,ps):
    y0=ps['IU'].value, ps['IS'].value, ps['IN'].value, ps['ISQ'].value, ps['INQ'].value, ps['D'].value, ps['R1'].value, ps['R2'].value, ps['NS'].value, ps["b0"].value
    T=sp.linspace(0,t,t+1)
    solu=odeint(f, y0, T, args=(ps,))# 
    return solu
#%%
final_brazil=solnew(t_brazil,fit)

IU_brazil=final_brazil[:,0]
IS_brazil=final_brazil[:,1]
IN_brazil=final_brazil[:,2]
ISQ_brazil=final_brazil[:,3]
INQ_brazil=final_brazil[:,4]
D_brazil=final_brazil[:,5]
R1_brazil=final_brazil[:,6]
R2_brazil=final_brazil[:,7]
NS_brazil=final_brazil[:,8]
b_brazil=final_brazil[:,9]
#%%
params = Parameters()
#params.add('y0', value=float(data[0]), min=0, max=100) #as for us the initial cond don't represent a parameter to be fitted
params.add('IU', value= 1400)
params.add('IS', value= 500)
params.add('IN', value= 200)
params.add('ISQ', value= y0_brazil[3])
params.add('INQ', value= y0_brazil[4])
params.add('D', value= y0_brazil[5])
params.add('R1', value= y0_brazil[6])
params.add('R2', value= 235)
params.add('NS', value= y0_brazil[8])
params.add('b0', value= 0.3484)
params.add("ln",value=0.5919)
params.add("p",value=0.8151)
params.add('a', value= 0.0170)
params.add("Tr",value=10.9399)
params.add("Trh",value=20.3722)
params.add("Tqs",value=10.109)
params.add("Tqn",value=29.92)
params.add("lamb", value=0.01274196)
params.add("bf", value=0.02881)
final_brazil=solnew(t_brazil,params)

IU_brazil=final_brazil[:,0]
IS_brazil=final_brazil[:,1]
IN_brazil=final_brazil[:,2]
ISQ_brazil=final_brazil[:,3]
INQ_brazil=final_brazil[:,4]
D_brazil=final_brazil[:,5]
R1_brazil=final_brazil[:,6]
R2_brazil=final_brazil[:,7]
NS_brazil=final_brazil[:,8]
b_brazil=final_brazil[:,9]
#%%
# plot data and fitted curves
T=sp.linspace(0,t_brazil,t_brazil+1)

plt.title("Evolution of COVID-19 pandemic in Brazil")
plt.plot(T, data_brazil[0], c="Orange", label="Official Detected Active Cases")
plt.plot(T, ISQ_brazil+INQ_brazil, '--', c='Red', label="Estimated Detected Active Cases")
plt.plot(T, data_brazil[1], c="Green",label="Official Registered Recovered")
plt.plot(T, R1_brazil, '--', c='Cyan',label="Estimated Registered Recovered")
plt.plot(T, data_brazil[2], c="Black",label="Official Dead")
plt.plot(T, D_brazil, "--", c="Brown",label="Estimated Dead")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()

plt.title("Evolution of COVID-19 pandemic in Brazil")
plt.plot(T, data_brazil[0]+data_brazil[1]+data_brazil[2], c="Purple",label="Official Total Detected Cases")
plt.plot(T, ISQ_brazil+INQ_brazil+R1_brazil+D_brazil, '--', c='Blue',label="Estimated Total Detected Cases")
plt.plot(T, data_brazil[2], c="Black",label="Official Dead")
plt.plot(T, D_brazil, "--", c="Brown",label="Estimated Dead")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()

plt.title("Evolution of COVID-19 pandemic in Brazil")
plt.plot(T, data_brazil[0]+data_brazil[1]+data_brazil[2], c="Purple",label="Official Total Detected Cases")
plt.plot(T, ISQ_brazil+INQ_brazil+R1_brazil+D_brazil, '--', c='Blue',label="Estimated Total Detected Cases")
plt.plot(T, data_brazil[2], c="Black",label="Official Dead")
plt.plot(T, D_brazil, "--", c="Brown",label="Estimated Dead")
plt.plot(T, IU_brazil+IS_brazil+IN_brazil+ISQ_brazil+INQ_brazil+R1_brazil+R2_brazil+D_brazil, "--", c="Red",label="Estimated Total Cases")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()

plt.title("Evolution of the Infectiousness in Brazil")
plt.plot(T,b_brazil)
plt.xlabel("Time (days)")
plt.grid("major", "both")
plt.show()
#%%
final_brazil=solnew(t_brazil+41,params)
T2=sp.linspace(0,t_brazil+41,t_brazil+1+41)

IU_brazil=final_brazil[:,0]
IS_brazil=final_brazil[:,1]
IN_brazil=final_brazil[:,2]
ISQ_brazil=final_brazil[:,3]
INQ_brazil=final_brazil[:,4]
D_brazil=final_brazil[:,5]
R1_brazil=final_brazil[:,6]
R2_brazil=final_brazil[:,7]
NS_brazil=final_brazil[:,8]
b_brazil=final_brazil[:,9]

plt.title("Evolution of COVID-19 pandemic in Brazil")
plt.plot(T, data_brazil[0]+data_brazil[1]+data_brazil[2], c="Purple",label="Official Total Detected Cases")
plt.plot(T2, ISQ_brazil+INQ_brazil+R1_brazil+D_brazil, '--', c='Blue',label="Estimated Total Detected Cases")
plt.plot(T, data_brazil[2], c="Black",label="Official Dead")
plt.plot(T2, D_brazil, "--", c="Brown",label="Estimated Dead")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()

plt.title("Evolution of COVID-19 pandemic in Brazil")
plt.plot(T, data_brazil[0]+data_brazil[1]+data_brazil[2], c="Purple",label="Official Total Detected Cases")
plt.plot(T2, ISQ_brazil+INQ_brazil+R1_brazil+D_brazil, '--', c='Blue',label="Estimated Total Detected Cases")
plt.plot(T, data_brazil[2], c="Black",label="Official Dead")
plt.plot(T2, D_brazil, "--", c="Brown",label="Estimated Dead")
plt.plot(T2, IU_brazil+IS_brazil+IN_brazil+ISQ_brazil+INQ_brazil+R1_brazil+R2_brazil+D_brazil, "--", c="Red",label="Estimated Total Cases")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()