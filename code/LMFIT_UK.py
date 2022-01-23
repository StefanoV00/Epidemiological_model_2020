# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:49:52 2020

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
This code works exclusively for UK
As in Italy we can individuate 4 fundamental phases (0,1,2,3), it has been necessary to develop
4 different ODE system functions f0, f1, f2, f3.
They all work exactly the same, the difference between them consists in which paramters they fit
and/or they depend upon. In fact, whil some paramters as p (probability of displaying symptoms)
should be constant, others, infectiousness above all, changes (abruptly) between different phases
Therefore f0 takes as paramter b0, f1 takes b1 etc... The same for Tqn, i.e. the average time it 
takes to detect an asymptomatic times the inverse probability an asymptomatic is detected, which 
depends o the "state of alert" and "attention" of the country, which is something that, more or 
less, might vary in time.
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
        Tqs=ps["Tqs"].value
        Tqn=ps["Tqn1"].value
    except:
        b,ln, p,a,Tr, Trh, Tqs, Tqn =ps
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

def f2 (y,t,ps): #for Odeint first comes the functions' vector y then the time
    try:
        b=ps["b2"].value
        ln=ps["ln"].value
        p=ps["p"].value
        a=ps["a"].value
        Tr=ps["Tr"].value
        Trh=ps["Trh"].value
        Tqs=ps["Tqs"].value
        Tqn=ps["Tqn2"].value
    except:
        b,ln, p,a,Tr, Trh, Tqs, Tqn =ps
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

def f3 (y,t,ps): #for Odeint first comes the functions' vector y then the time
    try:
        b=ps["b3"].value
        ln=ps["ln"].value
        p=ps["p"].value
        a=ps["a"].value
        Tr=ps["Tr"].value
        Trh=ps["Trh"].value
        Tqs=ps["Tqs"].value
        Tqn=ps["Tqn3"].value
    except:
        b,ln, p,a,Tr, Trh, Tqs, Tqn =ps
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
UNITED KINGDOM
"""
#THE UNITED KINGDOM DOESN'T MAKE AVILABLE THE NUMBER OF ACTIVE CASES AND THE ONE OF THE RECOVERED

# 14March
TOTCASES_UK=sp.array([1140,1391,1543,1950,2626,3269,3983,5018,5683,6650,8077,9529,11658,14543,17089,19522,22141,25150,29474,33718,38168,41903,47806,51608,55242,60733,65077,73758,78991,84279,88621,93873,98476,103093,108692,114217,120067,124743,129044,133495,138078,143464,148377,152840,157149,161145,165221,171253,177454,182260,186599,190584,194990,201101,206715,211364,215260,219183,223060,226463,229705,233151,236711,240161,243695,246406,248818,250332,252947,256234,259193,261598,263223,265227,267240,269127,271222,272826,274762,276332,277985,279856,281661,283311,284868,286194,287399,289140,290143,291409,292950,294375,295889,296857,298136,299251,300469,301815,303110])
dead_UK=sp.array([28,43,65,81,115,158,194,250,285,359,508,694,877,1161,1455,1669,2043,2425,3095,3747,4461,5221,5865,6433,7471,8505,9608,10760,11599,12285,13029,14073,14915,15944,16879,17994,18492,19051,20223,21060,21787,22792,23635,24055,24393,25302,26097,26771,27510,28131,28446,28734,29427,30076,30615,31241,31587,31855,32065,32692,33186,33614,33998,34466,34636,34796,35341,35704,36042,36393,36675,37116,37237,37373,37807,38220,38593,38819,38934,39045,39369,39728,39904,40261,40465,40542,40597,40883,41279,41279,41481,41662,41698,41736,41969,42153,42288,42461,42589])
#ALL THE OTHER NEEDED DATA ARE NOT AVAILABLE
data_UK=sp.array([TOTCASES_UK, dead_UK])
t_UK=len(TOTCASES_UK)-1

y0_UK=[1400, 400, 200, 1000, 140, 28, 50, 200, 67886011]

#NOW DISTINGUISH AMONG THE PERIODS:
infec_UK0=[]
dead_UK0=[]
infec_UK1=[]
dead_UK1=[]
infec_UK2=[]
dead_UK2=[]
infec_UK3=[]
dead_UK3=[]
for i in range (0,len(TOTCASES_UK)):     
    if i in range(0,12): 
        infec_UK0.append(TOTCASES_UK[i])          
        dead_UK0.append(dead_UK[i])
    elif i in range(12,60):               #26th March: UK lockdown legally starts
        infec_UK1.append(TOTCASES_UK[i])
        dead_UK1.append(dead_UK[i])
    elif i in range(60,78):               #10th May: A social relaxation of the lockdown begins in Englnd: Stay Home becomes Stay Alert
        infec_UK2.append(TOTCASES_UK[i])
        dead_UK2.append(dead_UK[i])
    else:                                 #28th May: Scotland announces a first relaxation of the lockdown measures
        infec_UK3.append(TOTCASES_UK[i])
        dead_UK3.append(dead_UK[i])
data_UK0=sp.array([infec_UK0,dead_UK0])
data_UK1=sp.array([infec_UK1,dead_UK1])
data_UK2=sp.array([infec_UK2,dead_UK2])
data_UK3=sp.array([infec_UK3,dead_UK3])
t_UK0=sp.arange(0,12,1)
t_UK1=sp.arange(12,60,1)
t_UK2=sp.arange(60,78,1)
t_UK3=sp.arange(78,len(TOTCASES_UK),1)
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

def sol2(t, y0, ps):
    solu=odeint(f2, y0, t, args=(ps,))# 
    return solu

def sol3(t, y0, ps):
    solu=odeint(f3, y0, t, args=(ps,))# 
    return solu

def residual_TD(ps, t, data): #must return an array
    """
    RESIDUALS FUNCTION between the model with its parameters and the real data:
        it considers the total cases detected (both active and recovered) together 
        and the deaths
    """
    #y0 = ps['y0'].value #as for us the initial conditions don't represent a parameter to be fitted
    y0=ps['IU'].value, ps['IS'].value, ps['IN'].value, ps['ISQ'].value, ps['INQ'].value, ps['D'].value, ps['R1'].value, ps['R2'].value, ps['NS'].value,
    model0 = sol0(t_UK0, y0, ps)
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
    model1 = sol1(t_UK1, y1, ps) #the data at the end of the previous phase is the initial cond of the next
    y2=[0,1,2,3,4,5,6,7,8]
    y2[0]= model1[:,0][-1]+(model1[:,0][-1]-model1[:,0][-2])
    y2[1]= model1[:,1][-1]+(model1[:,1][-1]-model1[:,1][-2])
    y2[2]= model1[:,2][-1]+(model1[:,2][-1]-model1[:,2][-2])
    y2[3]= model1[:,3][-1]+(model1[:,3][-1]-model1[:,3][-2])
    y2[4]= model1[:,4][-1]+(model1[:,4][-1]-model1[:,4][-2])
    y2[5]= model1[:,5][-1]+(model1[:,5][-1]-model1[:,5][-2])
    y2[6]= model1[:,6][-1]+(model1[:,6][-1]-model1[:,6][-2])
    y2[7]= model1[:,7][-1]+(model1[:,7][-1]-model1[:,7][-2])
    y2[8]= model1[:,8][-1]+(model1[:,8][-1]-model1[:,8][-2])
    model2 = sol2(t_UK2, y2, ps) #the data at the end of the previous phase is the initial cond of the next
    y3=[0,1,2,3,4,5,6,7,8]
    y3[0]= model2[:,0][-1]+(model2[:,0][-1]-model2[:,0][-2])
    y3[1]= model2[:,1][-1]+(model2[:,1][-1]-model2[:,1][-2])
    y3[2]= model2[:,2][-1]+(model2[:,2][-1]-model2[:,2][-2])
    y3[3]= model2[:,3][-1]+(model2[:,3][-1]-model2[:,3][-2])
    y3[4]= model2[:,4][-1]+(model2[:,4][-1]-model2[:,4][-2])
    y3[5]= model2[:,5][-1]+(model2[:,5][-1]-model2[:,5][-2])
    y3[6]= model2[:,6][-1]+(model2[:,6][-1]-model2[:,6][-2])
    y3[7]= model2[:,7][-1]+(model2[:,7][-1]-model2[:,7][-2])
    y3[8]= model2[:,8][-1]+(model2[:,8][-1]-model2[:,8][-2])
    model3 = sol3(t_UK3, y3, ps) #the data at the end of the previous phase is the initial cond of the next
    ISQ=sp.append(model0[:,3],sp.append(model1[:,3],sp.append(model2[:,3], model3[:,3])))
    INQ=sp.append(model0[:,4],sp.append(model1[:,4],sp.append(model2[:,4], model3[:,4])))
    ID=ISQ+INQ
    R1=sp.append(model0[:,6],sp.append(model1[:,6],sp.append(model2[:,6], model3[:,6])))
    Det=ID+R1
    D=sp.append(model0[:,5],sp.append(model1[:,5],sp.append(model2[:,5], model3[:,5])))
    delta=Det+D-data[0]
    rDet=(delta/sp.sqrt(data[0])).ravel()
    rD=((D-data[1])/sp.sqrt(data[1])).ravel()
    Res=sp.sqrt(16*rDet**2+rD**2) #GIVES MORE IMPORTANCE TO THE INFECTED CURVE
    return Res

def solnew(ps):
    y0=ps['IU'].value, ps['IS'].value, ps['IN'].value, ps['ISQ'].value, ps['INQ'].value, ps['D'].value, ps['R1'].value, ps['R2'].value, ps['NS'].value,
    model0 = sol0(t_UK0, y0, ps)
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
    model1 = sol1(t_UK1, y1, ps) #the data at the end of the previous phase is the initial cond of the next
    y2=[0,1,2,3,4,5,6,7,8]
    y2[0]= model1[:,0][-1]+(model1[:,0][-1]-model1[:,0][-2])
    y2[1]= model1[:,1][-1]+(model1[:,1][-1]-model1[:,1][-2])
    y2[2]= model1[:,2][-1]+(model1[:,2][-1]-model1[:,2][-2])
    y2[3]= model1[:,3][-1]+(model1[:,3][-1]-model1[:,3][-2])
    y2[4]= model1[:,4][-1]+(model1[:,4][-1]-model1[:,4][-2])
    y2[5]= model1[:,5][-1]+(model1[:,5][-1]-model1[:,5][-2])
    y2[6]= model1[:,6][-1]+(model1[:,6][-1]-model1[:,6][-2])
    y2[7]= model1[:,7][-1]+(model1[:,7][-1]-model1[:,7][-2])
    y2[8]= model1[:,8][-1]+(model1[:,8][-1]-model1[:,8][-2])
    model2 = sol2(t_UK2, y2, ps) #the data at the end of the previous phase is the initial cond of the next
    y3=[0,1,2,3,4,5,6,7,8]
    y3[0]= model2[:,0][-1]+(model2[:,0][-1]-model2[:,0][-2])
    y3[1]= model2[:,1][-1]+(model2[:,1][-1]-model2[:,1][-2])
    y3[2]= model2[:,2][-1]+(model2[:,2][-1]-model2[:,2][-2])
    y3[3]= model2[:,3][-1]+(model2[:,3][-1]-model2[:,3][-2])
    y3[4]= model2[:,4][-1]+(model2[:,4][-1]-model2[:,4][-2])
    y3[5]= model2[:,5][-1]+(model2[:,5][-1]-model2[:,5][-2])
    y3[6]= model2[:,6][-1]+(model2[:,6][-1]-model2[:,6][-2])
    y3[7]= model2[:,7][-1]+(model2[:,7][-1]-model2[:,7][-2])
    y3[8]= model2[:,8][-1]+(model2[:,8][-1]-model2[:,8][-2])
    model3 = sol3(t_UK3, y3, ps) #the data at the end of the previous phase is the initial cond of the next
    solu=[0,1,2,3,4,5,6,7,8]
    solu[0]=sp.append(model0[:,0],sp.append(model1[:,0],sp.append(model2[:,0], model3[:,0])))
    solu[1]=sp.append(model0[:,1],sp.append(model1[:,1],sp.append(model2[:,1], model3[:,1])))
    solu[2]=sp.append(model0[:,2],sp.append(model1[:,2],sp.append(model2[:,2], model3[:,2])))
    solu[3]=sp.append(model0[:,3],sp.append(model1[:,3],sp.append(model2[:,3], model3[:,3])))
    solu[4]=sp.append(model0[:,4],sp.append(model1[:,4],sp.append(model2[:,4], model3[:,4])))
    solu[5]=sp.append(model0[:,5],sp.append(model1[:,5],sp.append(model2[:,5], model3[:,5])))
    solu[6]=sp.append(model0[:,6],sp.append(model1[:,6],sp.append(model2[:,6], model3[:,6])))
    solu[7]=sp.append(model0[:,7],sp.append(model1[:,7],sp.append(model2[:,7], model3[:,7])))
    solu[8]=sp.append(model0[:,8],sp.append(model1[:,8],sp.append(model2[:,8], model3[:,8])))
    return solu
#%%
"""
UNITED KINGDOM: PARAMETERS
"""
# set parameters including their bounds
params = Parameters()
params.add('IU', value= 1400,min=1000, max=2000, vary=False)
params.add('IS', value= 400,min=200, max=1500, vary=False)
params.add('IN', value= 200,min=100, max=1000, vary=False)
params.add('ISQ', value= y0_UK[3],vary=False)
params.add('INQ', value= y0_UK[4],vary=False)
params.add('D', value= y0_UK[5],vary=False)
params.add('R1', value= y0_UK[6],vary=False)
params.add('R2', value= 183,min=100, max=500, vary=False)
params.add('NS', value= y0_UK[8],vary=False)
params.add('b0', value= 0.6655, min=0.665, max=0.667)
params.add('b1', value= 0.1652, min=0.165, max=0.166)
params.add('b2', value= 0.1226, min=0.122, max=0.123)
params.add('b3', value= 0.1204, min=0.1202, max=0.1205)
params.add('ln', value= 0.5893, min=0.588, max=0.591)
params.add('p', value= 0.792, min=0.79, max=0.8)
params.add('a', value= 0.048, min=0.045, max=0.055)
params.add('Tr', value= 9, min=8.5, max=9.1)
params.add("Trh",value=28, min=22, max=30)
params.add("Tqs",value=5.9, min=5.5, max=6)
params.add("Tqn",value=25.14, min=24, max=30)
params.add("Tqn1",value=19.98, min=18, max=25)
params.add("Tqn2",value=19.95, min=18, max=25)
params.add("Tqn3",value=20.9, min=18, max=25)
#%%
"""
FIT THE MODEL BY MINIMIZING THE RESIDUALS
"""
# fit model and find predicted values
UK = minimize(residual_TD, params, args=(t_UK,data_UK), method='emcee')
#final_brazil = data_brazil + result_brazil.residual.reshape(data_brazil.shape)
#THIS STEP CANNOT WORK AS THE RESIDUALS, AS WE DEFINED THEM ARE THE SUM OF THE TWO RESIDUALS COMING FROM 
#THE DETECTED INFECTED CURVE AND THE REGISTERED RECOVERED CURVE
# display fitted statistics
report_fit(UK)
UK_fit=UK.params

"""
NOW it's necessary to evaluate the system with the brand new parameters
"""
final=solnew(UK_fit)

IU_UK= final[0]
IS_UK= final[1]
IN_UK= final[2]
ISQ_UK=final[3]
INQ_UK=final[4]
D_UK=  final[5]
R1_UK= final[6]
R2_UK= final[7]
NS_UK= final[8]
#%%
params = Parameters()
params.add('IU', value= 1400)
params.add('IS', value= 400)
params.add('IN', value= 200)
params.add('ISQ', value= y0_UK[3])
params.add('INQ', value= y0_UK[4])
params.add('D', value= y0_UK[5])
params.add('R1', value= y0_UK[6])
params.add('R2', value= 183)
params.add('NS', value= y0_UK[8])
params.add('b0', value= 0.66619046)
params.add('b1', value= 0.16548619)
params.add('b2', value= 0.11240116)
params.add('b3', value= 0.11233319)
params.add('ln', value= 0.5895)
params.add('p', value= 0.79118)
params.add('a', value= 0.053)
params.add('Tr', value= 8.6)
params.add("Trh",value=28)
params.add("Tqs",value=5.83)
params.add("Tqn",value=25.644)
params.add("Tqn1",value=23.22)
params.add("Tqn2",value=20.954)
params.add("Tqn3",value=20.1167)
final=solnew(params)

IU_UK= final[0]
IS_UK= final[1]
IN_UK= final[2]
ISQ_UK=final[3]
INQ_UK=final[4]
D_UK=  final[5]
R1_UK= final[6]
R2_UK= final[7]
NS_UK= final[8]
#%%
"""
PLOT MORE THAN ONE GRAPH AND COMPARE THE OFFICIAL AND THE FITTED CURVES
"""
try:
    t=sp.linspace(0,t_UK,t_UK+1)
except:
    t=t_UK
plt.title("Evolution of COVID-19 pandemic in UK")
plt.plot(t, dead_UK, c="Black",label="Official Dead") 
plt.plot(t, D_UK,"--", c="Brown",label="Estimated Dead")
plt.plot(t, TOTCASES_UK, c="Purple",label="Official Total Detected Cases")
plt.plot(t, ISQ_UK+INQ_UK+R1_UK+D_UK,"--", c="Blue", label="Estimated Total Detected Cases")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()

plt.title("Evolution of COVID-19 pandemic in UK")
plt.plot(t, ISQ_UK+INQ_UK, '--', c='Red', label="Estimated Detected Active Cases") 
plt.plot(t, R1_UK, '--', c='Cyan',label="Estimated Registered Recovered") 
plt.plot(t, dead_UK, c="Black",label="Official Dead") 
plt.plot(t, D_UK,"--", c="Brown",label="Estimated Dead")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()

plt.title("Evolution of COVID-19 pandemic in UK")
plt.plot(t, ISQ_UK+INQ_UK, '--', c='Red', label="Estimated Detected Active Cases")
plt.plot(t, R1_UK, '--', c='Cyan',label="Estimated Registered Recovered")
plt.plot(t, dead_UK, c="Black",label="Official Dead")
plt.plot(t, D_UK,"--", c="Brown",label="Estimated Dead")
plt.plot(t, IU_UK+ IS_UK+ IN_UK,"--", c="Purple",label="Estimated Undiscovered Active Cases")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()

plt.title("Evolution of COVID-19 pandemic in UK")
plt.plot(t, dead_UK, c="Black",label="Official Dead")
plt.plot(t, D_UK,"--", c="Brown",label="Estimated Dead")
plt.plot(t, TOTCASES_UK, c="Purple" , label="Official Total Detected Cases")
plt.plot(t, ISQ_UK+INQ_UK+R1_UK+D_UK,"--", c="Blue",label="Estimated Total Detected Cases" )
plt.plot(t, IU_UK+IS_UK+IN_UK+ISQ_UK+INQ_UK+R1_UK+D_UK+R2_UK,"--", c="Red", label="Estimated Total Cases")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()