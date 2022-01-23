# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:08:55 2020

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
GENERAL NOTES APPLICABLE TO ALL THE COUNTRIES IN EXAM
"""
#1) the following data comes from Worldometer
#2) the first day taken into account is either the day before or after the detected infected overcame 1000
#3) the only certain initial conditions regard the detcted infected, the registered recovered, the deaths:
#   the other initial conditions might then be fitted together with the other paramers with lmfit
#4) the recovered array isn't directly available, but it's easily computable from deaths and the per-day CFR
#%%
"""
BRAZIL
"""
y0_brazil=[1422,400,200,1000,158,18,2,250,212413908] #21 March, the fisrt day the detected infected became 1000

infec_brazil=sp.array([1158,1519,1888,2199,2493,2902,3319,3784,4114,4347,5389,6511,7593,8704,9788,10641,11492,13221,15241,17018,18548,19649,20796,21929,9704,12827,14710,17515,20335,14062,16026,16013,17533,19606,21670,25991,28436,30816,35292,39718,43544,47660,48872,51131,55108,58573,66653,71155,76603,83720,86619,90134,92601,97575,109446,118436,127837,130840,138056,147108,157780,164879,174412,182798,190991,199314,209218,222317,218867,247213,264365,278980,288279,284763,285301,306834,322307,338366,353379,347973,377985,355087,367899,391309,380395,386981,393870,418601,410137,431983,469118,476895])
dead_brazil=sp.array([18,25,34,46,59,77,92,114,136,163,201,242,324,363,445,486,564,686,820,954,1068,1140,1223,1328,1532,1757,1947,2141,2361,2462,2587,2741,2906,3313,3670,4045,4271,4543,5063,5511,5901,6410,6750,7025,7343,7921,8588,9188,9992,10656,11123,11625,12404,13158,13993,14817,15633,16118,16853,17983,18894,20082,21048,22013,22716,23522,24549,25697,26764,27944,28834,29314,30046,31278,32547,34039,35047,36044,36499,37912,38497,39497,41058,41901,42791,43389,44118,45456,46665,47869,49090,50058])
#now the percentage dead people represents over the closed cases:
perc_brazil=sp.array([90,92.59,94.44,95.83,96.72,92.77,93.88,95,95.77,57.6,65.58,71.84,74.08,77.8,79.28,81.62,84.38,86.59,84.65,86.06,86.82,87.61,88.47,9.85,11.13,12.19,13.24,14.41,10.01,10.47,10.13,10.30,11.09,11.72,12.18,12.41,12.73,13.46,13.90,14.11,14.42,14.15,14.05,13.81,14.11,14.32,14.24,14.42,14.73,14.62,14.71,14.59,14.37,14.97,14.85,14.85,14.62,14.37,14.41,13.94,13.75,13.45,13.37,13.16,13.26,13,13.40,13.36,12.17,12.64,12.31,12.43,12.46,11.50,10.90,11.01,10.83,10.66,10.78,10.28,10.57,9.47,9.38,9.55,9.10,9.02,8.86,8.91,8.48,8.68])
TOTCASES_brazil=sp.array([1178,1546,1924,2247,2554,2985,3417,3904,4256,4630,5717,6880,8044,9194,10360,11254,12183,14034,16188,18145,19789,20962,22192,23430,25262,28610,30683,33682,36722,38654,40743,43079,45757,49492,52995,59196,62859,66501,72899,79361,85380,92109,96559,101147,108266,114715,126611,135693,145892,156061,162699,169143,177602,189157,202918,218223,233142,241080,255368,271825,293357,310921,330890,347398,363618,376669,392360,414661,438812,468338,498440,514849,529405,556668,583980,615870,646006,673587,691962,710887,742084,775184,805649,829902,850756,867882,891556,928834,960309,983359,1038569,1070139])

recov_brazil=TOTCASES_brazil-infec_brazil-dead_brazil

data_brazil=sp.array([infec_brazil,recov_brazil])
t_brazil = len(infec_brazil)-1
#%%
"""
ITALY
"""
y0_ita=[1100,300,150,949,100,29,50,200,60468537] #29 February, the first day the infected overcame 1000

infec_ita=sp.array([1049,1577,1835,2263,2706,3296,3916,5061,6387,7985,8514,10590,12839,14955,17750,20603,23073,26062,28710,33190,37860,42681,46638,50418,54030,57521,62013,66414,70065,73880,75528,77635,80572,83049,85388,88274,91246,93187,94067,95262,96877,98273,100269,102253,103616,104291,105418,106607,106962,107771,108257,108237,107709,107699,106848,106527,105847,106103,105813,105205,104657,102551,100943,100704,100179,99980,98470,91528,89624,87961,84842,83324,82488,81266,78457,76440,72070,70817,68351,66553,65129,62752,60960,59322,57752,56594,55300,52942,50966,47986,46175,43691,42075,41367,39893,39297,38429,36976,35877,35262,34730,32870,31710,30637,28997,27485,26274,25909,24369,23625,22702,21543,21212])
dead_ita=sp.array([29,41,52,79,107,148,197,233,366,463,631,827,1016,1266,1441,1809,2158,2503,2978,3405,4032,4825,5476,6077,6820,7503,8215,9134,10023,10779,11591,12428,13155,13915,14681,15362,15887,16523,17127,17669,18279,18849,19468,19899,20465,21067,21645,22170,22745,23227,23660,24114,24648,25085,25549,25969,26384,26644,26967,27359,27682,27967,28236,28710,28884,29079,29315,29684,29958,30201,30395,30560,30739,30911,31106,31168,31610,31763,31908,32007,32169,32330,32486,32616,32735,32785,32877,32955,33072,33142,33229,33340,33415,33475,33530,33601,33689,33774,33846,33899,33964,34043,34114,34167,34223,34301,34345,34371,34400,34460,34508,34564,34610])
perc_ita=sp.array([3671,3306,2587,3305,2794,2633,2736,2835,3704,3901,3859,4418,4468,4680,4230,4365,4398,4598,4252,4340,4401,4428,4381,4498,4503,4449,4422,4548,4473,4527,4422,4414,4370,4385,4322,4263,4225,4214,4148,4125,4001,3910,3823,3744,3678,3661,3620,3623,3557,3474,3408,3346,3304,3233,3150,3074,3003,2948,2910,2882,2841,2798,2691,2652,2643,2613,2597,2509,2415,2373,2278,2251,2238,2209,2165,2139,2082,2055,2031,2009,1991,1964,1945,1926,1908,1892,1880,1855,1836,1804,1786,1764,1750,1745,1732,1727,1722,1710,1705,1697,1694,1680,1672,1663,1651,1640,1630,1626,1616,1610,1605])
perc_ita=perc_ita/100
TOTCASES_ita=sp.array([1128,1701,2036,2502,3089,3858,4636,5883,7775,9172,10149,12462,15113,17660,21157,24747,27980,31506,35713,41035,47021,53578,59138,63927,69176,74386,80589,86498,92472,97689,101739,105792,110574,115242,119827,124632,128948,132547,135586,139422,143626,147577,152271,156363,159516,162488,165155,168941,172434,175925,178972,181228,183957,187327,189973,192994,195351,197675,199414,201505,203591,205463,207428,209328,210717,211938,213013,214457,215858,217185,218268,219070,219814,221216,222104,223096,223885,224760,225435,225886,226699,227364,228006,228658,229327,229858,230158,230555,231139,231732,232248,232664,232997,233197,233515,233836,234013,234531,234801,234998,235278,235561,235763,236142,236305,236500,236700,236891,237100,237429,237820,238011,238275])

recov_ita=TOTCASES_ita-infec_ita-dead_ita
    
data_ita=sp.array([infec_ita,recov_ita])
t_ita = len(infec_ita)-1

#NOW DISTINGUISH AMONG THE PERIODS:
    #PHASE 0 (NORMALITY)             :29 FEB -> 8 MARCH
    #PHASE 1 (LOCKDOWN)              : 9 MAR -> 3 MAY
    #PHASE 2 (FIRTS  RIAPERTURE)     : 4 MAY -> 2 JUNE
    #PHASE 3 (INTER-REGIONS APERTURE): 3 JUN ->
infec_ita0=[]
recov_ita0=[]
infec_ita1=[]
recov_ita1=[]
infec_ita2=[]
recov_ita2=[]
infec_ita3=[]
recov_ita3=[]
for i in range (0,len(infec_ita)):
    if i in range(0,9):
        infec_ita0.append(infec_ita[i])
        recov_ita0.append(recov_ita[i])
    elif i in range(9,65):
        infec_ita1.append(infec_ita[i])
        recov_ita1.append(recov_ita[i])
    elif i in range(65,95):
        infec_ita2.append(infec_ita[i])
        recov_ita2.append(recov_ita[i])
    else:
        infec_ita3.append(infec_ita[i])
        recov_ita3.append(recov_ita[i])
data_ita0=sp.array([infec_ita0,recov_ita0])
data_ita1=sp.array([infec_ita1,recov_ita1])
data_ita2=sp.array([infec_ita2,recov_ita2])
data_ita3=sp.array([infec_ita3,recov_ita3])
#TIME MUST BE DEFINED ACCORIDNGLY, BOTH HERE AND IN THE FUNCTIONS
plt.plot(sp.linspace(0,t_ita,t_ita+1),infec_ita,c="Red")
plt.plot(sp.linspace(0,t_ita,t_ita+1),recov_ita,c="Green")
plt.plot(sp.linspace(0,t_ita,t_ita+1),dead_ita,c="Black")
plt.plot(sp.linspace(0,t_ita,t_ita+1),infec_ita+recov_ita+dead_ita,c="Purple")
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

plt.plot(sp.linspace(0,t_UK,t_UK+1),TOTCASES_UK,c="Purple")
plt.plot(sp.linspace(0,t_UK,t_UK+1),dead_UK,c="Black")
#%%
"""
INDIA
"""
#y0=[IU,IS,IN,1000,117,32,102,R2,1380004385]# 30 March

infec_india=sp.array([1117,1239,1792,2280,2781,3260,3843,4267,4723,5232,5863,6577,7189,7794,8914,9735,10440,11214,11825,13381,14202,14674,15460,16319,17306,18171,19519,20486,21375,22569,23546,24641,26027,27557,29339,32024,33565,35871,37686,39823,41406,43980,45925,47457,49104,51379,52773,53553,55878,57939,60864,63172,66089,69244,73170,76820,80072,82172,85803,89755,85884,89706,93349,97008,101077,106655,111900,116302,120952,126431,129360,132896,138069,142810,146482,150814,153574,152791,154688,160564,163688,168636,170269])
dead_india=sp.array([32,35,58,72,86,99,118,136,160,178,227,249,288,331,358,393,422,448,486,521,559,592,645,681,721,780,825,881,939,1008,1079,1154,1223,1323,1391,1566,1693,1785,1889,1985,2101,2212,2294,2415,2551,2649,2753,2871,3025,3156,3302,3434,3584,3726,3868,4024,4172,4344,4534,4711,4980,5185,5408,5608,5829,6088,6363,6649,6950,7207,7473,7719,8107,8501,8890,9204,9520,9915,11921,12262,12606,12970,13277])
perc_india=sp.array([2388,2215,2816,2738,3094,3018,2646,2661,2548,2602,2633,2434,2291,2346,2326,2243,2187,2022,1923,1746,1638,1532,1396,1348,1258,1243,1220,1190,1163,1151,1134,1129,1089,1090,1057,1087,1069,1043,1012,999,982,954,923,900,881,865,834,774,760,745,724,703,687,671,664,652,643,633,627,623,568,563,556,553,549,553,554,555,553,550,547,544,544,547,545,536,531,521,598,593,579,571,550])
perc_india=perc_india/100

recov_india=sp.linspace(0,len(infec_india)-1,len(infec_india))
for i in range(0,len(dead_india)): 
   # print(i)
    recov_india[i]=((100-perc_india[i])*dead_india[i]/perc_india[i])
    
data_india=sp.array([infec_india,recov_india, dead_india])
t_india = len(infec_india)-1

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
t_india0=sp.arange(0,70,1)
t_india1=sp.arange(70,len(infec_india),1)

plt.plot(sp.linspace(0,t_india,t_india+1),infec_india,c="Red")
plt.plot(sp.linspace(0,t_india,t_india+1),dead_india,c="Black")
plt.plot(sp.linspace(0,t_india,t_india+1),recov_india,c="Green")
plt.plot(sp.linspace(0,t_india,t_india+1),infec_india+dead_india+recov_india,c="Purple")
#%%
"""
SWEDEN
"""
#SWEDEN DOESN'T MAKE AVILABLE THE NUMBER OF ACTIVE CASES AND THE ONE OF THE RECOVERED

#y0=[IU,IS,IN,900,140,3,R1,R2,10100000]# 30 March
#the data for sweden is blocked at the 18th of June worlodmeter, because apparently they stopped counting
TOTCASES_swe=sp.array([1040,1121,1196,1301,1439,1639,1770,1934,2046,2299,2526,2840,3069,3447,3700,4028,4435,4947,5568,6131,6443,6830,7206,7693,8419,9141,9685,10151,10483,10948,11445,11927,12540,13216,13822,14385,14777,15322,16004,16755,17567,18177,18640,18926,19621,20302,21092,21520,22082,22317,22721,23216,23918,24623,25265,25921,26322,26670,27272,27909,28582,29207,29677,30143,30377,30799,31523,32172,32809,33188,33459,33843,34440,35088,35727,36476,37113,37814,38589,40803,41883,42939,43887,44730,45186,45729,46665,48092,49479,49684,50931,51614,52383,53323,54562,56043])
dead_swe=sp.array([3,7,8,10,11,16,20,21,27,40,62,77,105,105,110,146,180,239,308,358,373,401,477,591,687,793,870,887,899,919,1033,1203,1333,1400,1511,1540,1580,1765,1937,2021,2152,2192,2194,2274,2355,2462,2586,2653,2669,2679,2769,2854,2941,3040,3175,3220,3225,3256,3313,3460,3529,3646,3674,3679,3698,3743,3831,3871,3925,3992,3998,4029,4125,4220,4266,4350,4395,4395,4403,4468,4542,4562,4639,4656,4659,4694,4717,4795,4814,4854,4874,4874,4891,4939,5041,5053])

t_swe = len(TOTCASES_swe)-1

plt.plot(sp.linspace(0,t_swe,t_swe+1),TOTCASES_swe,c="Purple")
plt.plot(sp.linspace(0,t_swe,t_swe+1),dead_swe,c="Black")

#SWEDEN IMPOSED NO LOCKDOWN, ONLY MADE RECOMMENDATIONS TO ITS CITIZEN 
