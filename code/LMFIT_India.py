# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:06:47 2020

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
"""
This code works exclusively for India
"""

def f0 (y,t,ps): #for Odeint first comes the functions' vector y then the time
    """
THE ODE NTH DIMENSIONAL SYSTEM: has arguments
                                             the time array
                                             the nth vector y
                                             the parameters which have to be fitted (the others are defined within the function)
the ODE system function f returns a nth vector with each element being a function (the derivative)
    """
#NN is ignored, as, with epsilon equal to 1 and NN0=0, NN=R
#  IT HAS THE REGISTERED AND UNREGISTERED RICOVERED SYSTEM
#  IT USES THE INFECTION FATALITY RATE (IFR) SYSTEM
#DEFINE THE FITTABLE PARAMETERS (WITH THE FOLLOWING TRY-EXCEPT SYSTEM)
    try:
        b=ps["b0"].value
        ln=ps["ln"].value
        p=ps["p"].value
        a=ps["a"].value
        Tr=ps["Tr"].value
        Trh=ps["Trh"].value
        Tqs=ps["Tqs"].value
        Tqn=ps["Tqn"].value
    except:
        b,ln, p,a,Tr, Trh, Tqs, Tqn =ps
#DEFINE THE CONSTANT PARAMETERS
    #b=0.5
    #a=0.028
    #Tqn=7
    #ln=0.6
    #p=0.8
    #Tqs=2
    #Trs=18
    #Trn=20
    #Trh=17
    tau=5.1
    CFR=0.062
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
#FINALLY DEFINE THE DIFFERENTIAL EQUATIONS
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
    f=[dIU,dIS,dIN,dISQ,dINQ,dD,dR1,dR2,dNS]
    return f

def f1 (y,t,ps): #for Odeint first comes the functions' vector y then the time
    try:
        b=ps["b1"].value
        ln=ps["ln"].value
        p=ps["p"].value
        a=ps["a"].value
        Tr=ps["Tr"].value
        Trh=ps["Trh"].value
        Tqs=ps["Tqs1"].value
        Tqn=ps["Tqn1"].value
    except:
        b,ln, p,a,Tr, Trh, Tqs1, Tqn =ps
    tau=5.1
    CFR=0.062
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
#FINALLY DEFINE THE DIFFERENTIAL EQUATIONS
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
    f=[dIU,dIS,dIN,dISQ,dINQ,dD,dR1,dR2,dNS]
    return f
#%%
"""
INDIA
"""
infec_india=sp.array([1117,1239,1792,2280,2781,3260,3843,4267,4723,5232,5863,6577,7189,7794,8914,9735,10440,11214,11825,13381,14202,14674,15460,16319,17306,18171,19519,20486,21375,22569,23546,24641,26027,27557,29339,32024,33565,35871,37686,39823,41406,43980,45925,47457,49104,51379,52773,53553,55878,57939,60864,63172,66089,69244,73170,76820,80072,82172,85803,89755,85884,89706,93349,97008,101077,106655,111900,116302,120952,126431,129360,132896,138069,142810,146482,150814,153574,152791,154688,160564,163688,168636,170269])
dead_india=sp.array([32,35,58,72,86,99,118,136,160,178,227,249,288,331,358,393,422,448,486,521,559,592,645,681,721,780,825,881,939,1008,1079,1154,1223,1323,1391,1566,1693,1785,1889,1985,2101,2212,2294,2415,2551,2649,2753,2871,3025,3156,3302,3434,3584,3726,3868,4024,4172,4344,4534,4711,4980,5185,5408,5608,5829,6088,6363,6649,6950,7207,7473,7719,8107,8501,8890,9204,9520,9915,11921,12262,12606,12970,13277])
perc_india=sp.array([2388,2215,2816,2738,3094,3018,2646,2661,2548,2602,2633,2434,2291,2346,2326,2243,2187,2022,1923,1746,1638,1532,1396,1348,1258,1243,1220,1190,1163,1151,1134,1129,1089,1090,1057,1087,1069,1043,1012,999,982,954,923,900,881,865,834,774,760,745,724,703,687,671,664,652,643,633,627,623,568,563,556,553,549,553,554,555,553,550,547,544,544,547,545,536,531,521,598,593,579,571,550])
perc_india=perc_india/100

y0=[1900,500,200,1000,117,32,102,200,1380004385]# 30 March

recov_india=sp.linspace(0,len(infec_india)-1,len(infec_india))
for i in range(0,len(dead_india)): 
   # print(i)
    recov_india[i]=((100-perc_india[i])*dead_india[i]/perc_india[i])
    
data_india=sp.array([infec_india,recov_india, dead_india])
t_india = len(perc_india)-1

#DISTINGUISH AMONG THE PERIODS (ACCORDING TO WIKIPEDIA)
    #PHASE 1 (LOCKDOWN): even if not homogeneous at the beginning, India was in a nationwide lockdown 
                       # by the 30 March, which is the beginning of the period we are interested in
    #UNLOCK 1: since the 8TH OF JUNE, some areas will progressively unlock some services
infec_india0=[]
recov_india0=[]
dead_india0=[]
infec_india1=[]
recov_india1=[]
dead_india1=[]
for i in range (0,len(infec_india)):     
    if i in range(0,70):                   
        infec_india0.append(infec_india[i])
        recov_india0.append(recov_india[i])
        dead_india0.append(dead_india[i])
    else:                                  
        infec_india1.append(infec_india[i])
        recov_india1.append(recov_india[i])
        dead_india1.append(dead_india[i])
data_india0=sp.array([infec_india0,recov_india0,dead_india0])
data_india1=sp.array([infec_india1,recov_india1,dead_india1])
t_india0=sp.arange(0,70,1)
t_india1=sp.arange(70,len(infec_india),1)
#%%
"""
As for the system of ODES, there must also be 4 solving functions, each working with a 
different fi
"""
def sol0(t, y0, ps):
    """
    SOLUTION TO THE ODEs SYSTEM with initial conditions y(0) = y0
    """
    solu=odeint(f0, y0, t, args=(ps,))# 
    return solu

def sol1(t, y0, ps):
    solu=odeint(f1, y0, t, args=(ps,))# 
    return solu

def residual_TD(ps, t, data): #must return an array
    """
    RESIDUALS FUNCTION between the model with its parameters and the real data:
        it considers the total cases detected (both active and recovered) together 
        and the deaths
    """
    y0=ps['IU'].value, ps['IS'].value, ps['IN'].value, ps['ISQ'].value, ps['INQ'].value, ps['D'].value, ps['R1'].value, ps['R2'].value, ps['NS'].value,
    model0 = sol0(t_india0, y0, ps)
    y1=[0,1,2,3,4,5,6,7,8]
    y1[0]= model0[:,0][-1]+(model0[:,0][-1]-model0[:,0][-2])
    y1[1]= model0[:,1][-1]+(model0[:,1][-1]-model0[:,1][-2])
    y1[2]= model0[:,2][-1]+(model0[:,2][-1]-model0[:,2][-2])
    y1[3]= model0[:,3][-1]+(model0[:,3][-1]-model0[:,3][-2])
    y1[4]= model0[:,4][-1]+(model0[:,4][-1]-model0[:,4][-2])
    y1[5]= model0[:,5][-1]+(model0[:,5][-1]-model0[:,5][-2])
    y1[6]= model0[:,6][-1]+(model0[:,6][-1]-model0[:,6][-2])
    y1[7]= model0[:,7][-1]+(model0[:,7][-1]-model0[:,7][-2])
    y1[8]= model0[:,8][-1]+(model0[:,8][-1]-model0[:,8][-2])
    model1 = sol1(t_india1, y1, ps) #the data at the end of the previous phase is the initial cond of the next
    ISQ=sp.append(model0[:,3],model1[:,3])
    INQ=sp.append(model0[:,4],model1[:,4])
    ID=ISQ+INQ
    R1=sp.append(model0[:,6],model1[:,6])
    Det=ID+R1
    D=sp.append(model0[:,5],model1[:,5])
    delta=Det+D-data[0]-data[1]-data[2]
    rDet=(delta/sp.sqrt(data[0]+data[1]+data[2])).ravel()
    rD=((D-data[2])/sp.sqrt(data[2])).ravel()
    Res=sp.sqrt(9*rDet**2+rD**2) #GIVES MORE IMPORTANCE TO THE INFECTED CURVE
    return Res

def residual_IRD(ps, t, data): #must return an array
    """
    RESIDUALS FUNCTION between the model with its parameters and the real data:
        it considers the total cases detected (both active and recovered) together 
        and the deaths
    """
    y0=ps['IU'].value, ps['IS'].value, ps['IN'].value, ps['ISQ'].value, ps['INQ'].value, ps['D'].value, ps['R1'].value, ps['R2'].value, ps['NS'].value,
    model0 = sol0(t_india0, y0, ps)
    y1=[0,1,2,3,4,5,6,7,8]
    y1[0]= model0[:,0][-1]+(model0[:,0][-1]-model0[:,0][-2])
    y1[1]= model0[:,1][-1]+(model0[:,1][-1]-model0[:,1][-2])
    y1[2]= model0[:,2][-1]+(model0[:,2][-1]-model0[:,2][-2])
    y1[3]= model0[:,3][-1]+(model0[:,3][-1]-model0[:,3][-2])
    y1[4]= model0[:,4][-1]+(model0[:,4][-1]-model0[:,4][-2])
    y1[5]= model0[:,5][-1]+(model0[:,5][-1]-model0[:,5][-2])
    y1[6]= model0[:,6][-1]+(model0[:,6][-1]-model0[:,6][-2])
    y1[7]= model0[:,7][-1]+(model0[:,7][-1]-model0[:,7][-2])
    y1[8]= model0[:,8][-1]+(model0[:,8][-1]-model0[:,8][-2])
    model1 = sol1(t_india1, y1, ps) #the data at the end of the previous phase is the initial cond of the next
    ISQ=sp.append(model0[:,3],model1[:,3])
    INQ=sp.append(model0[:,4],model1[:,4])
    ID=ISQ+INQ
    R1=sp.append(model0[:,6],model1[:,6])
    Det=ID+R1
    D=sp.append(model0[:,5],model1[:,5])
    delta=ID-data[0]
    rI=(delta/sp.sqrt(data[0])).ravel()
    rR=((R1-data[1]/sp.sqrt(data[1]))).ravel()
    rD=((D-data[2])/sp.sqrt(data[2])).ravel()
    Res=sp.sqrt(9*rI**2+9*rR**2+rD**2) #GIVES MORE IMPORTANCE TO THE INFECTED CURVE
    return Res

def solnew(ps):
    y0=ps['IU'].value, ps['IS'].value, ps['IN'].value, ps['ISQ'].value, ps['INQ'].value, ps['D'].value, ps['R1'].value, ps['R2'].value, ps['NS'].value,
    model0 = sol0(t_india0, y0, ps)
    y1=[0,1,2,3,4,5,6,7,8]
    y1[0]= model0[:,0][-1]+(model0[:,0][-1]-model0[:,0][-2])
    y1[1]= model0[:,1][-1]+(model0[:,1][-1]-model0[:,1][-2])
    y1[2]= model0[:,2][-1]+(model0[:,2][-1]-model0[:,2][-2])
    y1[3]= model0[:,3][-1]+(model0[:,3][-1]-model0[:,3][-2])
    y1[4]= model0[:,4][-1]+(model0[:,4][-1]-model0[:,4][-2])
    y1[5]= model0[:,5][-1]+(model0[:,5][-1]-model0[:,5][-2])
    y1[6]= model0[:,6][-1]+(model0[:,6][-1]-model0[:,6][-2])
    y1[7]= model0[:,7][-1]+(model0[:,7][-1]-model0[:,7][-2])
    y1[8]= model0[:,8][-1]+(model0[:,8][-1]-model0[:,8][-2])
    model1 = sol1(t_india1, y1, ps) #the data at the end of the previous phase is the initial cond of the next
    solu=[0,1,2,3,4,5,6,7,8]
    solu[0]=sp.append(model0[:,0],model1[:,0])
    solu[1]=sp.append(model0[:,1],model1[:,1])
    solu[2]=sp.append(model0[:,2],model1[:,2])
    solu[3]=sp.append(model0[:,3],model1[:,3])
    solu[4]=sp.append(model0[:,4],model1[:,4])
    solu[5]=sp.append(model0[:,5],model1[:,5])
    solu[6]=sp.append(model0[:,6],model1[:,6])
    solu[7]=sp.append(model0[:,7],model1[:,7])
    solu[8]=sp.append(model0[:,8],model1[:,8])
    return solu
#%%
"""
INDIA: PARAMETERS
"""
# set parameters including their bounds
params = Parameters()
params.add('IU', value= 2500,min=1800, max=3000, vary=True)
params.add('IS', value= 450,min=400, max=600, vary=True)
params.add('IN', value= 350,min=300, max=400, vary=True)
params.add('ISQ', value= y0[3],vary=False)
params.add('INQ', value= y0[4],vary=False)
params.add('D', value= y0[5],vary=False)
params.add('R1', value= y0[6],vary=False)
params.add('R2', value= 200,min=100, max=500, vary=False)
params.add('NS', value= y0[8],vary=False)
params.add('b0', value= 0.26, min=0.25, max=0.30)
params.add('b1', value= 0.10, min=0.05, max=0.15)
params.add('ln', value= 0.59, min=0.57, max=0.62)
params.add('p', value= 0.81, min=0.79, max=0.82)
params.add('a', value= 0.011, min=0.008, max=0.015)
params.add('Tr', value= 9, min=8.5, max=12)
params.add("Trh",value=18, min=15, max=20)
params.add("Tqs",value=6.5, min=6, max=9)
params.add("Tqs1",value=6.5, min=6, max=9)
params.add("Tqn",value=25, min=15, max=40)
params.add("Tqn1",value=20, min=15, max=40)
#%%
"""
FIT THE MODEL BY MINIMIZING THE RESIDUALS
"""
# fit model and find predicted values
india = minimize(residual_TD, params, args=(t_india,data_india), method='emcee')
#final_brazil = data_brazil + result_brazil.residual.reshape(data_brazil.shape)
#THIS STEP CANNOT WORK AS THE RESIDUALS, AS WE DEFINED THEM ARE THE SUM OF THE TWO RESIDUALS COMING FROM 
#THE DETECTED INFECTED CURVE AND THE REGISTERED RECOVERED CURVE
# display fitted statistics
report_fit(india)
fit=india.params

"""
NOW it's necessary to evaluate the system with the brand new parameters
"""
final=solnew(fit)

IU_india= final[0]
IS_india= final[1]
IN_india= final[2]
ISQ_india=final[3]
INQ_india=final[4]
D_india=  final[5]
R1_india= final[6]
R2_india= final[7]
NS_india= final[8]
#%%
params = Parameters()
params.add('IU', value= 2940)
params.add('IS', value= 510)
params.add('IN', value= 324)
params.add('ISQ', value= y0[3])
params.add('INQ', value= y0[4])
params.add('D', value= y0[5])
params.add('R1', value= y0[6])
params.add('R2', value= 200)
params.add('NS', value= y0[8])
params.add('b0', value= 0.250601)
params.add('b1', value= 0.129209)
params.add('ln', value= 0.589703)
params.add('p', value= 0.806730)
params.add('a', value= 0.0133984)
params.add('Tr', value= 9.4808)
params.add("Trh",value=18)
params.add("Tqs",value=6.03373)
params.add("Tqs1",value=7.91485)
params.add("Tqn",value=16.261241)
params.add("Tqn1",value=20.4900)

final=solnew(params)

IU_india= final[0]
IS_india= final[1]
IN_india= final[2]
ISQ_india=final[3]
INQ_india=final[4]
D_india=  final[5]
R1_india= final[6]
R2_india= final[7]
NS_india= final[8]
#%%
"""
PLOT MORE THAN ONE GRAPH AND COMPARE THE OFFICIAL AND THE FITTED CURVES
"""
try:
    t=sp.linspace(0,t_india,t_india+1)
except:
    t=t_india

plt.title("Evolution of COVID-19 pandemic in India")
plt.plot(t, infec_india, c="Orange", label="Official Detected Active Cases")
plt.plot(t, ISQ_india+INQ_india, '--', c='Red',label="Estimated Detected Active Cases")
plt.plot(t, recov_india, c="Green", label="Official Registered Recovered")
plt.plot(t, R1_india, '--', c='Cyan',label="Estimated Registered Recovered")
plt.plot(t, dead_india, c="Black",label="Official Dead") 
plt.plot(t, D_india,"--", c="Brown",label="Estimated Dead")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()

plt.title("Evolution of COVID-19 pandemic in India")
plt.plot(t, infec_india, c="Orange", label="Official Detected Active Cases")
plt.plot(t, ISQ_india+INQ_india, '--', c='Red',label="Estimated Detected Active Cases")
plt.plot(t, recov_india, c="Green", label="Official Registered Recovered")
plt.plot(t, R1_india, '--', c='Cyan',label="Estimated Registered Recovered")
plt.plot(t, dead_india, c="Black",label="Official Dead") 
plt.plot(t, D_india,"--", c="Brown",label="Estimated Dead")
plt.plot(t, infec_india+recov_india+dead_india, c="Purple",label="Official Total Detected Cases")
plt.plot(t, ISQ_india+INQ_india+R1_india+D_india,"--", c="Blue",label="Estimated Total Detected Cases")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()

plt.title("Evolution of COVID-19 pandemic in India")
plt.plot(t, infec_india, c="Orange", label="Official Detected Active Cases")
plt.plot(t, ISQ_india+INQ_india, '--', c='Red',label="Estimated Detected Active Cases")
plt.plot(t, recov_india, c="Green", label="Official Registered Recovered")
plt.plot(t, R1_india, '--', c='Cyan',label="Estimated Registered Recovered")
plt.plot(t, dead_india, c="Black",label="Official Dead") 
plt.plot(t, D_india,"--", c="Brown",label="Estimated Dead")
plt.plot(t, IU_india+ IS_india+ IN_india,"--", c="Purple",label="Estimated Undiscovered Active Cases")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()

plt.title("Evolution of COVID-19 pandemic in India")
plt.plot(t, dead_india, c="Black",label="Official Dead")
plt.plot(t, D_india,"--", c="Brown",label="Estimated Dead")
plt.plot(t, infec_india+recov_india+dead_india, c="Purple",label="Official Total Detected Cases")
plt.plot(t, ISQ_india+INQ_india+R1_india+D_india,"--", c="Blue",label="Estimated Total Detected Cases")
plt.plot(t, IU_india+IS_india+IN_india+ISQ_india+INQ_india+R1_india+D_india+R2_india,"--", c="Red",label="Estimated Total Cases")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()

plt.title("Evolution of COVID-19 pandemic in India")
plt.plot(t, dead_india, c="Black",label="Official Dead")
plt.plot(t, D_india,"--", c="Brown",label="Estimated Dead")
plt.plot(t, infec_india+recov_india+dead_india, c="Purple",label="Official Total Detected Cases")
plt.plot(t, ISQ_india+INQ_india+R1_india+D_india,"--", c="Blue",label="Estimated Total Detected Cases")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()
#%%
def solnew(ps,t):
    y0=ps['IU'].value, ps['IS'].value, ps['IN'].value, ps['ISQ'].value, ps['INQ'].value, ps['D'].value, ps['R1'].value, ps['R2'].value, ps['NS'].value,
    model0 = sol0(t_india0, y0, ps)
    y1=[0,1,2,3,4,5,6,7,8]
    y1[0]= model0[:,0][-1]+(model0[:,0][-1]-model0[:,0][-2])
    y1[1]= model0[:,1][-1]+(model0[:,1][-1]-model0[:,1][-2])
    y1[2]= model0[:,2][-1]+(model0[:,2][-1]-model0[:,2][-2])
    y1[3]= model0[:,3][-1]+(model0[:,3][-1]-model0[:,3][-2])
    y1[4]= model0[:,4][-1]+(model0[:,4][-1]-model0[:,4][-2])
    y1[5]= model0[:,5][-1]+(model0[:,5][-1]-model0[:,5][-2])
    y1[6]= model0[:,6][-1]+(model0[:,6][-1]-model0[:,6][-2])
    y1[7]= model0[:,7][-1]+(model0[:,7][-1]-model0[:,7][-2])
    y1[8]= model0[:,8][-1]+(model0[:,8][-1]-model0[:,8][-2])
    model1 = sol1(t_india1, y1, ps) #the data at the end of the previous phase is the initial cond of the next
    y2=[0,1,2,3,4,5,6,7,8]
    y1[0]= model1[:,0][-1]+(model1[:,0][-1]-model1[:,0][-2])
    y1[1]= model1[:,1][-1]+(model1[:,1][-1]-model1[:,1][-2])
    y1[2]= model1[:,2][-1]+(model1[:,2][-1]-model1[:,2][-2])
    y1[3]= model1[:,3][-1]+(model1[:,3][-1]-model1[:,3][-2])
    y1[4]= model1[:,4][-1]+(model1[:,4][-1]-model1[:,4][-2])
    y1[5]= model1[:,5][-1]+(model1[:,5][-1]-model1[:,5][-2])
    y1[6]= model1[:,6][-1]+(model1[:,6][-1]-model1[:,6][-2])
    y1[7]= model1[:,7][-1]+(model1[:,7][-1]-model1[:,7][-2])
    y1[8]= model1[:,8][-1]+(model1[:,8][-1]-model1[:,8][-2])
    T=sp.linspace(t_india+1,t_india+t,t)
    model2 = sol1(T, y1, ps)
    print(sp.size(model0[:,1]),sp.size(model1[:,1]),sp.size(model2[:,1]))
    solu=[0,1,2,3,4,5,6,7,8]
    solu[0]=sp.append(model0[:,0],sp.append(model1[:,0], model2[:,0]))
    solu[1]=sp.append(model0[:,1],sp.append(model1[:,1], model2[:,1]))
    solu[2]=sp.append(model0[:,2],sp.append(model1[:,2], model2[:,2]))
    solu[3]=sp.append(model0[:,3],sp.append(model1[:,3], model2[:,3]))
    solu[4]=sp.append(model0[:,4],sp.append(model1[:,4], model2[:,4]))
    solu[5]=sp.append(model0[:,5],sp.append(model1[:,5], model2[:,5]))
    solu[6]=sp.append(model0[:,6],sp.append(model1[:,6], model2[:,6]))
    solu[7]=sp.append(model0[:,7],sp.append(model1[:,7], model2[:,7]))
    solu[8]=sp.append(model0[:,8],sp.append(model1[:,8], model2[:,8]))
    return solu
params = Parameters()
params.add('IU', value= 2940)
params.add('IS', value= 510)
params.add('IN', value= 324)
params.add('ISQ', value= y0[3])
params.add('INQ', value= y0[4])
params.add('D', value= y0[5])
params.add('R1', value= y0[6])
params.add('R2', value= 200)
params.add('NS', value= y0[8])
params.add('b0', value= 0.250601)
params.add('b1', value= 0.129209)
params.add('ln', value= 0.589703)
params.add('p', value= 0.806730)
params.add('a', value= 0.0133984)
params.add('Tr', value= 9.4808)
params.add("Trh",value=18)
params.add("Tqs",value=6.03373)
params.add("Tqs1",value=7.91485)
params.add("Tqn",value=16.261241)
params.add("Tqn1",value=20.4900)

final=solnew(params, 41)

IU_india= final[0]
IS_india= final[1]
IN_india= final[2]
ISQ_india=final[3]
INQ_india=final[4]
D_india=  final[5]
R1_india= final[6]
R2_india= final[7]
NS_india= final[8]
#%%
T2=sp.linspace(0,t_india+41,t_india+1+41)
plt.title("Evolution of COVID-19 pandemic in India")
plt.plot(t, dead_india, c="Black",label="Official Dead")
plt.plot(T2, D_india,"--", c="Brown",label="Estimated Dead")
plt.plot(t, infec_india+recov_india+dead_india, c="Purple",label="Official Total Detected Cases")
plt.plot(T2, ISQ_india+INQ_india+R1_india+D_india,"--", c="Blue",label="Estimated Total Detected Cases")
plt.plot(T2, IU_india+IS_india+IN_india+ISQ_india+INQ_india+R1_india+D_india+R2_india,"--", c="Red",label="Estimated Total Cases")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()

plt.title("Evolution of COVID-19 pandemic in India")
plt.plot(t, dead_india, c="Black",label="Official Dead")
plt.plot(T2, D_india,"--", c="Brown",label="Estimated Dead")
plt.plot(t, infec_india+recov_india+dead_india, c="Purple",label="Official Total Detected Cases")
plt.plot(T2, ISQ_india+INQ_india+R1_india+D_india,"--", c="Blue",label="Estimated Total Detected Cases")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()
