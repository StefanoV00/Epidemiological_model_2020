# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:01:54 2020

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
This code works exclusively for ITALY
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
ITALY
"""
y0_ita=[1100,300,150,949,100,29,50,200,60468537] #29 February, the first day the infected overcame 1000

infec_ita=sp.array([1049,1577,1835,2263,2706,3296,3916,5061,6387,7985,8514,10590,12839,14955,17750,20603,23073,26062,28710,33190,37860,42681,46638,50418,54030,57521,62013,66414,70065,73880,75528,77635,80572,83049,85388,88274,91246,93187,94067,95262,96877,98273,100269,102253,103616,104291,105418,106607,106962,107771,108257,108237,107709,107699,106848,106527,105847,106103,105813,105205,104657,102551,100943,100704,100179,99980,98470,91528,89624,87961,84842,83324,82488,81266,78457,76440,72070,70817,68351,66553,65129,62752,60960,59322,57752,56594,55300,52942,50966,47986,46175,43691,42075,41367,39893,39297,38429,36976,35877,35262,34730,32870,31710,30637,28997,27485,26274,25909,24369,23625,22702,21543,21212])
dead_ita=sp.array([29,41,52,79,107,148,197,233,366,463,631,827,1016,1266,1441,1809,2158,2503,2978,3405,4032,4825,5476,6077,6820,7503,8215,9134,10023,10779,11591,12428,13155,13915,14681,15362,15887,16523,17127,17669,18279,18849,19468,19899,20465,21067,21645,22170,22745,23227,23660,24114,24648,25085,25549,25969,26384,26644,26967,27359,27682,27967,28236,28710,28884,29079,29315,29684,29958,30201,30395,30560,30739,30911,31106,31168,31610,31763,31908,32007,32169,32330,32486,32616,32735,32785,32877,32955,33072,33142,33229,33340,33415,33475,33530,33601,33689,33774,33846,33899,33964,34043,34114,34167,34223,34301,34345,34371,34400,34460,34508,34564,34610])
TOTCASES_ita=sp.array([1128,1701,2036,2502,3089,3858,4636,5883,7775,9172,10149,12462,15113,17660,21157,24747,27980,31506,35713,41035,47021,53578,59138,63927,69176,74386,80589,86498,92472,97689,101739,105792,110574,115242,119827,124632,128948,132547,135586,139422,143626,147577,152271,156363,159516,162488,165155,168941,172434,175925,178972,181228,183957,187327,189973,192994,195351,197675,199414,201505,203591,205463,207428,209328,210717,211938,213013,214457,215858,217185,218268,219070,219814,221216,222104,223096,223885,224760,225435,225886,226699,227364,228006,228658,229327,229858,230158,230555,231139,231732,232248,232664,232997,233197,233515,233836,234013,234531,234801,234998,235278,235561,235763,236142,236305,236500,236700,236891,237100,237429,237820,238011,238275])

recov_ita=TOTCASES_ita-infec_ita-dead_ita
    
data_ita=sp.array([infec_ita,recov_ita, dead_ita])
t_ita = len(infec_ita)-1

"""
NOW DISTINGUISH AMONG THE PERIODS:
    PHASE 0 (NORMALITY)             :29 FEB -> 8 MARCH
    PHASE 1 (LOCKDOWN)              : 9 MAR -> 3 MAY
    PHASE 2 (FIRST  RIAPERTURE)     : 4 MAY -> 2 JUNE
    PHASE 3 (INTER-REGIONS APERTURE): 3 JUN ->
"""
infec_ita0=[]
recov_ita0=[]
dead_ita0=[]
infec_ita1=[]
recov_ita1=[]
dead_ita1=[]
infec_ita2=[]
recov_ita2=[]
dead_ita2=[]
infec_ita3=[]
recov_ita3=[]
dead_ita3=[]
for i in range (0,len(infec_ita)):
    if i in range(0,9):
        infec_ita0.append(infec_ita[i])
        recov_ita0.append(recov_ita[i])
        dead_ita0.append(dead_ita[i])
    elif i in range(9,65):
        infec_ita1.append(infec_ita[i])
        recov_ita1.append(recov_ita[i])
        dead_ita1.append(dead_ita[i])
    elif i in range(65,95):
        infec_ita2.append(infec_ita[i])
        recov_ita2.append(recov_ita[i])
        dead_ita2.append(dead_ita[i])
    else:
        infec_ita3.append(infec_ita[i])
        recov_ita3.append(recov_ita[i])
        dead_ita3.append(dead_ita[i])
data_ita0=sp.array([infec_ita0,recov_ita0,dead_ita0])
data_ita1=sp.array([infec_ita1,recov_ita1,dead_ita1])
data_ita2=sp.array([infec_ita2,recov_ita2,dead_ita2])
data_ita3=sp.array([infec_ita3,recov_ita3,dead_ita3])
t_ita0=sp.arange(0,9,1)
t_ita1=sp.arange(9,65,1)
t_ita2=sp.arange(65,95,1)
t_ita3=sp.arange(95,len(infec_ita),1)
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

def residual_IRD(ps, t, data): #must return an array
    """
    RESIDUALS FUNCTION between the model with its parameters and the real data:
        it considers, more or less separately, the detected infected and the recovered and the deaths
    """
    #y0 = ps['y0'].value #as for us the initial conditions don't represent a parameter to be fitted
    y0=ps['IU'].value, ps['IS'].value, ps['IN'].value, ps['ISQ'].value, ps['INQ'].value, ps['D'].value, ps['R1'].value, ps['R2'].value, ps['NS'].value,
    model0 = sol0(t_ita0, y0, ps)
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
    model1 = sol1(t_ita1, y1, ps) #the data at the end of the previous phase is the initial cond of the next
    y2=[0,1,2,3,4,5,6,7,8]
    y2[0]= model1[:,0][-1]+(model1[:,0][-1]-model1[:,0][-2])
    y2[1]= model1[:,1][-1]+(model1[:,0][-1]-model1[:,0][-2])
    y2[2]= model1[:,2][-1]+(model1[:,0][-1]-model1[:,0][-2])
    y2[3]= model1[:,3][-1]+(model1[:,0][-1]-model1[:,0][-2])
    y2[4]= model1[:,4][-1]+(model1[:,0][-1]-model1[:,0][-2])
    y2[5]= model1[:,5][-1]+(model1[:,0][-1]-model1[:,0][-2])
    y2[6]= model1[:,6][-1]+(model1[:,0][-1]-model1[:,0][-2])
    y2[7]= model1[:,7][-1]+(model1[:,0][-1]-model1[:,0][-2])
    y2[8]= model1[:,8][-1]+(model1[:,0][-1]-model1[:,0][-2])
    model2 = sol2(t_ita2, y2, ps) #the data at the end of the previous phase is the initial cond of the next
    y3=[0,1,2,3,4,5,6,7,8]
    y3[0]= model2[:,0][-1]+(model2[:,0][-1]-model1[:,0][-2])
    y3[1]= model2[:,1][-1]+(model2[:,0][-1]-model1[:,0][-2])
    y3[2]= model2[:,2][-1]+(model2[:,0][-1]-model1[:,0][-2])
    y3[3]= model2[:,3][-1]+(model2[:,0][-1]-model1[:,0][-2])
    y3[4]= model2[:,4][-1]+(model2[:,0][-1]-model1[:,0][-2])
    y3[5]= model2[:,5][-1]+(model2[:,0][-1]-model1[:,0][-2])
    y3[6]= model2[:,6][-1]+(model2[:,0][-1]-model1[:,0][-2])
    y3[7]= model2[:,7][-1]+(model2[:,0][-1]-model1[:,0][-2])
    y3[8]= model2[:,8][-1]+(model2[:,0][-1]-model1[:,0][-2])
    model3 = sol3(t_ita3, y3, ps) #the data at the end of the previous phase is the initial cond of the next
    ISQ=sp.append(model0[:,3],sp.append(model1[:,3],sp.append(model2[:,3], model3[:,3])))
    INQ=sp.append(model0[:,4],sp.append(model1[:,4],sp.append(model2[:,4], model3[:,4])))
    ID=ISQ+INQ
    R1=sp.append(model0[:,6],sp.append(model1[:,6],sp.append(model2[:,6], model3[:,6])))
    D=sp.append(model0[:,5],sp.append(model1[:,5],sp.append(model2[:,5], model3[:,5])))
    delta=ID-data[0]
    rI=(delta/sp.sqrt(data[0])).ravel()
    rR=((R1-data[1])/sp.sqrt(data[1])).ravel()
    rD=((D-data[2])/sp.sqrt(data[2])).ravel()
    Res=sp.sqrt(9*rI**2+rR**2+rD**2) #GIVES MORE IMPORTANCE TO THE INFECTED CURVE
    return Res

def residual_TD(ps, t, data): #must return an array
    """
    RESIDUALS FUNCTION between the model with its parameters and the real data:
        it considers the total cases detected (both active and recovered) together 
        and the deaths
    """
    #y0 = ps['y0'].value #as for us the initial conditions don't represent a parameter to be fitted
    y0=ps['IU'].value, ps['IS'].value, ps['IN'].value, ps['ISQ'].value, ps['INQ'].value, ps['D'].value, ps['R1'].value, ps['R2'].value, ps['NS'].value,
    model0 = sol0(t_ita0, y0, ps)
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
    model1 = sol1(t_ita1, y1, ps) #the data at the end of the previous phase is the initial cond of the next
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
    model2 = sol2(t_ita2, y2, ps) #the data at the end of the previous phase is the initial cond of the next
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
    model3 = sol3(t_ita3, y3, ps) #the data at the end of the previous phase is the initial cond of the next
    ISQ=sp.append(model0[:,3],sp.append(model1[:,3],sp.append(model2[:,3], model3[:,3])))
    INQ=sp.append(model0[:,4],sp.append(model1[:,4],sp.append(model2[:,4], model3[:,4])))
    ID=ISQ+INQ
    R1=sp.append(model0[:,6],sp.append(model1[:,6],sp.append(model2[:,6], model3[:,6])))
    D=sp.append(model0[:,5],sp.append(model1[:,5],sp.append(model2[:,5], model3[:,5])))
    Det=ID+R1+D
    delta=Det-data[0]-data[1]-data[2]
    rDet=(delta/sp.sqrt(data[0]+data[1]+data[2])).ravel()
    rD=((D-data[2])/sp.sqrt(data[2])).ravel()
    Res=sp.sqrt(9*rDet**2+rD**2) #GIVES MORE IMPORTANCE TO THE INFECTED CURVE
    return Res

def solnew(ps):
    y0=ps['IU'].value, ps['IS'].value, ps['IN'].value, ps['ISQ'].value, ps['INQ'].value, ps['D'].value, ps['R1'].value, ps['R2'].value, ps['NS'].value,
    model0 = sol0(t_ita0, y0, ps)
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
    model1 = sol1(t_ita1, y1, ps) #the data at the end of the previous phase is the initial cond of the next
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
    model2 = sol2(t_ita2, y2, ps) #the data at the end of the previous phase is the initial cond of the next
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
    model3 = sol3(t_ita3, y3, ps) #the data at the end of the previous phase is the initial cond of the next
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
ITALY: PARAMETERS
"""
# set parameters including their bounds
params = Parameters()
params.add('IU', value= 1800,min=1000, max=2000, vary=False)
params.add('IS', value= 500,min=200, max=1500, vary=False)
params.add('IN', value= 350,min=100, max=1000, vary=False)
params.add('ISQ', value= y0_ita[3],vary=False)
params.add('INQ', value= y0_ita[4],vary=False)
params.add('D', value= y0_ita[5],vary=False)
params.add('R1', value= y0_ita[6],vary=False)
params.add('R2', value= 333,min=100, max=500, vary=False)
params.add('NS', value= y0_ita[8],vary=False)
params.add('b0', value= 0.85, min=0.848, max=0.88)
params.add('b1', value= 0.14938, min=0.1443, max=0.19)
params.add('b2', value= 0.040, min=0.0399, max=0.07)
params.add('b3', value= 0.1607, min=0.159, max=0.25)
params.add('ln', value= 0.602, min=0.599, max=0.607)
params.add('p', value= 0.817, min=0.812, max=0.82)
params.add('a', value= 0.0386, min=0.038, max=0.039)
#params.add('a2', value= 0.03, min=0.003, max=0.070)
params.add('Tr', value= 8.5, min=8, max=11)
params.add("Trh",value=40, min=33, max=50)
params.add("Tqs",value=5.79, min=5.5, max=6)
params.add("Tqn",value=16.68, min=15, max=20)
params.add("Tqn1",value=15.224, min=15, max=20)
params.add("Tqn2",value=25, min=15, max=35)
params.add("Tqn3",value=25, min=15, max=30)
#%%
"""
FIT THE MODEL BY MINIMIZING THE RESIDUALS
"""
# fit model and find predicted values
ita = minimize(residual_IRD, params, args=(t_ita,data_ita), method='emcee')
#final_brazil = data_brazil + result_brazil.residual.reshape(data_brazil.shape)
#THIS STEP CANNOT WORK AS THE RESIDUALS, AS WE DEFINED THEM ARE THE SUM OF THE TWO RESIDUALS COMING FROM 
#THE DETECTED INFECTED CURVE AND THE REGISTERED RECOVERED CURVE
# display fitted statistics
report_fit(ita)
ita_fit=ita.params
"""
NOW it's necessary to evaluate the system with the brand new parameters
"""
final_ita=solnew(ita_fit)

IU_ita= final_ita[0]
IS_ita= final_ita[1]
IN_ita= final_ita[2]
ISQ_ita=final_ita[3]
INQ_ita=final_ita[4]
D_ita=  final_ita[5]
R1_ita= final_ita[6]
R2_ita= final_ita[7]
NS_ita= final_ita[8]
"""
PLOT MORE THAN ONE GRAPH AND COMPARE THE OFFICIAL AND THE FITTED CURVES
"""
try:
    t_ita=sp.linspace(0,t_ita,t_ita+1)
except:
    t_ita=t_ita
plt.plot(t_ita, infec_ita, c="Orange")  #OFFICIAL DETECTED ACTIVE INFECTED
plt.plot(t_ita, ISQ_ita+INQ_ita, '--', c='Red') #ESTIMATED DETECTED ACTIVE INFECTED
plt.plot(t_ita, recov_ita, c="Green") #OFFICIAL REGISTERED RECOVERED 
plt.plot(t_ita, R1_ita, '--', c='Blue') #ESTIMATED REGISTERED RECOVERED
plt.plot(t_ita, dead_ita, c="Black")  #OFFICIAL DEAD
plt.plot(t_ita, D_ita,"--", c="Brown") #ESTIMATED DEAD
plt.plot(t_ita, infec_ita+recov_ita+dead_ita, c="Purple" )
plt.plot(t_ita, ISQ_ita+INQ_ita+R1_ita+D_ita,"--", c="Cyan")
plt.show()
plt.plot(t_ita, infec_ita, c="Orange")  #OFFICIAL DETECTED ACTIVE INFECTED
plt.plot(t_ita, ISQ_ita+INQ_ita, '--', c='Red') #ESTIMATED DETECTED ACTIVE INFECTED
plt.plot(t_ita, recov_ita, c="Green") #OFFICIAL REGISTERED RECOVERED 
plt.plot(t_ita, R1_ita, '--', c='Blue') #ESTIMATED REGISTERED RECOVERED
plt.plot(t_ita, dead_ita, c="Black")  #OFFICIAL DEAD
plt.plot(t_ita, D_ita,"--", c="Brown") #ESTIMATED DEAD
plt.show()
#%%
# set parameters including their bounds
params = Parameters()
params.add('IU', value= 1800)
params.add('IS', value= 500)
params.add('IN', value= 350)
params.add('ISQ', value= y0_ita[3])
params.add('INQ', value= y0_ita[4])
params.add('D', value= y0_ita[5])
params.add('R1', value= y0_ita[6])
params.add('R2', value= 333)
params.add('NS', value= y0_ita[8])
params.add('b0', value= 0.777)
params.add('b1', value= 0.159)
params.add('b2', value= 0.048)
params.add('b3', value= 0.16385)
params.add('ln', value= 0.60233)
params.add('p', value= 0.814)
params.add('a', value= 0.037)
params.add('Tr', value= 8.534)
params.add("Trh",value=33)
params.add("Tqs",value=5.808)
params.add("Tqn",value=16.654)
params.add("Tqn1",value=17.266)
params.add("Tqn2",value=25.09)
params.add("Tqn3",value=25.558)
final_ita=solnew(params)

IU_ita= final_ita[0]
IS_ita= final_ita[1]
IN_ita= final_ita[2]
ISQ_ita=final_ita[3]
INQ_ita=final_ita[4]
D_ita=  final_ita[5]
R1_ita= final_ita[6]
R2_ita= final_ita[7]
NS_ita= final_ita[8]

"""
PLOT MORE THAN ONE GRAPH AND COMPARE THE OFFICIAL AND THE FITTED CURVES
"""
try:
    t_ita=sp.linspace(0,t_ita,t_ita+1)
except:
    t_ita=t_ita
 
plt.title("Evolution of COVID-19 pandemic in Italy")
plt.plot(t_ita, dead_ita, c="Black", label="Official Dead") 
plt.plot(t_ita, D_ita,"--", c="Brown", label="Estimated Dead") 
plt.plot(t_ita, infec_ita+recov_ita+dead_ita, c="Purple", label="Official Total Detected Cases" )
plt.plot(t_ita, ISQ_ita+INQ_ita+R1_ita+D_ita,"--", c="Blue", label="Estimated Total Detected Cases")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()
#%%
plt.title("Evolution of COVID-19 pandemic in Italy")
plt.plot(t_ita, infec_ita, c="Orange", label="Official Detected Active Cases") 
plt.plot(t_ita, ISQ_ita+INQ_ita, '--', c='Red', label="Estimated Detected Active Cases") 
plt.plot(t_ita, recov_ita, c="Green",label="Official Registered Recovered") 
plt.plot(t_ita, R1_ita, '--', c='Cyan',label="Estimated Registered Recovered") 
plt.plot(t_ita, dead_ita, c="Black",label="Official Dead") 
plt.plot(t_ita, D_ita,"--", c="Brown",label="Estimated Dead") 
#plt.plot(t_ita, IU_ita+ IS_ita+ IN_ita,"--", c="Purple", label="Estimated Undiscovered Active Cases")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()

plt.title("Evolution of COVID-19 pandemic in Italy")
plt.plot(t_ita, dead_ita, c="Black",label="Official Dead") 
plt.plot(t_ita, D_ita,"--", c="Brown",label="Estimated Dead")
plt.plot(t_ita, infec_ita+recov_ita+dead_ita, c="Purple", label="Official Total Detected Cases")
plt.plot(t_ita, ISQ_ita+INQ_ita+R1_ita+D_ita,"--", c="Blue", label="Estimated Total Detected Cases")
plt.plot(t_ita, IU_ita+IS_ita+IN_ita+ISQ_ita+INQ_ita+R1_ita+D_ita+R2_ita,"--", c="Red", label="Estimated Total Cases")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.legend()
plt.grid("major", "both")
plt.show()