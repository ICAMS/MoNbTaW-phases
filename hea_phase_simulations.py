# Authors: Yury Lysogorskiy^1, Alberto Ferrari^1,2
# 1: Interdisciplinary Centre for Advanced Materials Simulation, Ruhr-University Bochum, 44801 Bochum, Germany
# 2: Materials Science and Engineering, Delft University of Technology, 2628CD Delft, The Netherlands

#
# License: see LICENSE file
#



# coding: utf-8

# In[1]:

#https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html for details on the optimizer


# In[2]:

import concurrent.futures
import os
import pickle
import random
import re
import sys
import warnings
import yaml

import numpy as np
import pandas as pd

from collections.abc import Iterable
from contextlib import contextmanager
from functools import partial
from itertools import combinations
from numba import jit
from scipy.optimize import minimize,Bounds,LinearConstraint, NonlinearConstraint,SR1,curve_fit
from scipy import interpolate
from scipy.linalg import block_diag
from scipy.spatial import ConvexHull


# In[3]:

# import mendeleev
from ase.eos import EquationOfState


# In[4]:

warnings.simplefilter("ignore")


# # Parameters

# In[5]:

input_dict_default = {'N_noise_attempts': 0,
 'Ts': [100, 200, 300],
 'disord_energy_std': 0.003,
 'initial_configuration_guesses': 2,
 'max_distance_to_convex_hull': 0.006,
 'max_workers': -1,
 'maxiter': 400,
 'nominal_concentration': {'Mo': 0.25, 'Nb': 0.25, 'Ta': 0.25, 'W': 0.25},
 'ord_energy_std': 0.002,
 'DATA_PATH': 'data'}


# In[6]:

print("Loading input")

with open("input.yaml","r") as f:
    input_dict = yaml.load(f)

print("Input parameters: ",input_dict)


# In[7]:

nominal_concentration=input_dict["nominal_concentration"]
initial_configuration_guesses=input_dict["initial_configuration_guesses"] #Every phase, then N_hops with gaussian noise, then N_hops with random guesses
Ts=input_dict["Ts"]

max_distance_to_convex_hull=input_dict["max_distance_to_convex_hull"] #filtering of ordered structures based on distance to conv_hull

N_noise_attempts=input_dict["N_noise_attempts"] #0 is no noise, >0 for statistics
disord_energy_std =input_dict["disord_energy_std"]
ord_energy_std =input_dict["ord_energy_std"]

DATA_PATH =input_dict["DATA_PATH"]
max_workers =input_dict["max_workers"] # -1 if all cores to use
maxiter =input_dict["maxiter"]#optmizer max iter

# # Routines

# In[8]:

sigma_disord = disord_energy_std
sigma_ord = ord_energy_std

conc_dict = nominal_concentration

N_hops = initial_configuration_guesses

thresh=max_distance_to_convex_hull

if max_workers==-1:
    max_workers = None
print("Maximum number of parallel workers (max_workers option): ",max_workers)
N_noise_attempts+=1

NUM_VOL_POINTS=5
st_dev=0.1 #Leave it as it is


# In[9]:

def pprint_configuration(x, threshold = 8e-3,prefix="--"):
    x=np.array(x)
    disordered_phase_to_show=x[:N]>threshold 

    for i,isshown,disord_phase_name in zip(range(N),disordered_phase_to_show,config_columns[:N]):
        if isshown:
            print(prefix,disord_phase_name,end=":")
            phase_comp = x[N+(i*N):N+(i*N)+N]
            for el, c in zip(elements,phase_comp):
                if c>threshold :
                    print(el,"({0:.2f})".format(c),end=" ")
    print()
    for col,c in zip(config_columns[N*(N+1):], x[N*(N+1):]):
        if c>threshold :
            print(prefix,col+"({0:.2f})".format(c))


# In[10]:

@jit
def func2corr(x, *pars):
    x_sum = x[0] + x[1]
    if x_sum == 0:
        return 0
    else:
        xr = x[0] / x_sum
        xr2 = xr**2
        res = (1 - xr) * (pars[0] * xr + pars[1] * (xr2) + pars[2] *
                                   (xr2 * xr) + pars[3] * (xr2 * xr2))
        res *= (x_sum)**2
    return res

@jit
def func3(x,*pars):   
    return pars[0]*x[0]*x[1]*x[2]


# In[11]:

def split_name(s):
    patt=re.compile("([A-Z][a-z]?)([0-9]*)")
    res=patt.findall(s)
    tot=0
    for i in range(0,len(res)):
        res[i]=list(res[i])
        if res[i][1]=='':
            res[i][1]=1
        res[i][1]=float(res[i][1])
        tot=tot+res[i][1]
    for i in range(0,len(res)):
        res[i][1]=res[i][1]/tot
    res=sorted(res)
    dict_res=dict(res)
    for key in elements:
        if not key in dict_res:
            dict_res[key]=0.0
    new_dict=dict()
    for key,v in dict_res.items():
        new_dict[key]=v
    return new_dict

def get_value_disord(x,qnty):
    Q_un=0
    Q_bin=0
    Q_tern=0
    for i in range(0,len(elements)):
        Q_un=Q_un+unary_dict[qnty][elements[i]]*x[i]    
    
        for j in range(0,len(elements)):
            if i<j:
                Q_bin=Q_bin+func2corr(np.array([x[i],x[j]]),*bin_dict[qnty][(elements[i],elements[j])])        
                for k in range(0,len(x)):
                    if j<k:
                        Q_tern=Q_tern+func3(np.array([x[i],x[j],x[k]]),*tern_dict[qnty][(elements[i],elements[j],elements[k])])
    Q=Q_un+Q_bin+Q_tern
    return Q

def get_all_values(x):
    q=[]
    for qnty in quantities:
        q.append(get_value_disord(x,qnty))
    return q[0],q[1],q[2],q[3]


# In[12]:

#Function for Gibbs phase rule
#Gibbs phase rule: at fixed T and p, N_ph=N_elements
alpha=20

z=np.linspace(-1,1)
def sigma(z):
    return np.tanh(alpha*z)
def d_sigma(z):
    return (1-np.tanh(alpha*z)**2)*alpha

#plt.plot(z,sigma(z))

# NONLINEAR CONSTRAINT

#@jit
def cons_f(x): #sum of concentrations weighted by phase fractions
    qc=[0.0]
    for element_num in range(N):
        element_conc=0
        for phase_num in range(N):
            phase_shift = N+phase_num*N
            element_conc+=x[phase_num]*x[phase_shift+element_num]
            
        ord_phase_shift = N*(N+1)
        for ord_phase_num in range(K):
            element_conc+=x[ord_phase_shift+ord_phase_num]* ordered_phases_dict[allphases[ord_phase_num]][element_num]
            
        qc.append(element_conc-conc[element_num])
    qc=np.array(qc)
    
    return qc[1:]

#Gibbs phase rule: at fixed T and p, N_ph=N_elements
def gibbs(x): 
    phases_conc=np.hstack((x[:N],x[-K:]))
    return np.sum(sigma(phases_conc))

#Jacobian of nonlinear constraint
def cons_J_vec(X): #jacobian
    X=np.array(X)
    I=np.eye(N)
    c_block = np.array([X[N+element_num:N*(N+1):N] for element_num in range(N)])
    phi_block=np.hstack([I*X[phase_num] for phase_num in range(N)]    )
    cprime_block=np.array([ordered_phases_dict[op] for op in allphases]).T
    J = np.hstack((c_block, phi_block,cprime_block))
    
    return J

def gibbs_J(X):
    X=np.array(X)
    sigma_block = X.copy()
    sigma_block[N:-K]=0
    sigma_block=d_sigma(sigma_block)
    sigma_block[N:-K]=0
    return sigma_block

def linf(x):
    r=[]
    for i in range(1,N+1):
        r.append(sum(x[i*N:(i+1)*N])-1)
    return r
def linf2(x):
    return sum(x[:N])+sum(x[-N:])-1


# In[45]:

def fake(x):
    return 1

@contextmanager
def try_interrupt_block(stop=False):
    try:
        yield
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print("Error:", e)
        if stop:
            raise

def minimize_gibbs_free_energy(Total_Free_Energy_func, x0, max_constr_violation = 0.01):
    res = minimize(Total_Free_Energy_func, x0,  method='trust-constr',  jac="2-point", hess=SR1(),
                        constraints=[linear_constraint, nonlinear_constraint, nonlinear_constraint_gibbs],
                        options={'verbose': 0,'gtol': 1E-3,'maxiter': maxiter,'xtol': 5E-4},
                      bounds=bounds
                  )
    if res.constr_violation>max_constr_violation:
            res.success=False
    return res

def fullfill_constraints(x0):
    res = minimize(fake, x0,  method='trust-constr',  jac="2-point", hess=SR1(),
                                constraints=[linear_constraint, nonlinear_constraint, nonlinear_constraint_gibbs],
                                options={'verbose': 0,'initial_constr_penalty': 10,'gtol': 1E-4,'maxiter': 350,'xtol': -1},
                              bounds=bounds,tol=1E-6
                          )
    success=res.constr_violation<0.0005
    return res, success

def fullfill_constraints_2(x0):
    res = minimize(fake, x0,  method='trust-constr',  jac="2-point", hess=SR1(),
                                constraints=[linear_constraint, nonlinear_constraint, nonlinear_constraint_gibbs],
                                options={'verbose': 0,'initial_constr_penalty': 10,'gtol': 1E-4,'maxiter': 1,'xtol': -1},
                              bounds=bounds,tol=1E-6
                          )
    success=res.constr_violation<0.0005
    return res, success 

def work_fullfill_constraints_1(*args,**kwargs):
    success = False
    with try_interrupt_block(stop=False):
        n_it = 0
        while not success and n_it < 1000:
            n_it += 1
            #x0=[random.uniform(0,1)]*(N*(N+1))+[0]*K
            which_phase = int(random.uniform(0, N + K))
            if which_phase < N:
                x0 = [1] + [0] * (N - 1) + list(conc + np.array(
                    [np.random.normal(scale=st_dev)])) * N + [0] * K
            if which_phase >= N:
                i = which_phase - N
                x0 = [0] + [0] * (N - 1) + list(conc) * N + [0] * i + [1] + [0] * (K - i - 1)
                rat = max(ordered_phases_dict[allphases[i]] / conc)
                x0[N * (N + 1) +
                   i] = x0[N *
                           (N + 1) + i] / rat + np.random.normal(scale=st_dev)
                x0[0] = 1 - x0[N *(N + 1) + i] + np.random.normal(scale=st_dev)
                for j in range(0, N):
                    x0[N +
                       j] = (conc[j] - ordered_phases_dict[allphases[i]][j] /
                             rat) / x0[0] + np.random.normal(scale=st_dev)
            #print(n_it)
            res, success = fullfill_constraints(x0)
    return res

def work_fullfill_constraints_2(*args,**kwargs):
    success = False
    with try_interrupt_block(stop=False):
        while not success:
            x0 = list(np.random.random(N) * 2 /
                      (N + K)) + list(np.random.random(N**2) * 1 / (N)) + list(
                          np.random.random(K) * 2 / (N + K))
            res, success = fullfill_constraints(x0)
        return res


# In[46]:

def compute_formation_energy(row):
    comp_dict=row["Comp_dict"]
    en = row["E"]
    n_at = row["N_AT"]
    e_ref = 0
    for el, conc in comp_dict.items():
        e_ref += conc*reference_energies_dict[el]
    return en/n_at-e_ref

def compute_free_energy(row):
    comp_dict=row["Comp_dict"]
    n_at = row["N_AT"]
    E0,V0,B,Bp=row["E"]/n_at,row["V"]/n_at,row["B"],row["Bp"]
    comp=[]
    for el in elements:
        comp.append(comp_dict[el])
    return Free_energy(comp,T_high,E0,V0,B,Bp)

def make_compdictio(row):
    compdictio={}
    for el in elements:
        compdictio[el]=row['c'+el]
    return compdictio

def compute_formation_free_energy(row):
    comp_dict=row["Comp_dict"]
    en = row["F_high_T"]
    e_ref = 0
    for el, conc in comp_dict.items():
        e_ref += conc*reference_free_energies_dict[el]
    return en-e_ref


# In[47]:

kB = 0.0000861673324
C = 41.63*np.sqrt(1/0.1)*np.sqrt(1/0.529177249)

def Free_energy(x,T,E0,V0,B0,Bp): #T in K, M is molar mass, E0 in eV, V0 in A^3, B0 in eV/A^3, x is concentration in atomic fraction
    try:
        i=0
        M=0
        for el in elements:
            M=M+mass_dict[el]*x[i]
            i=i+1
        M=abs(M)
        r0 = (3*V0 / 4 / np.pi)**(1.0/3.0)
        T_D0 = C*np.sqrt(r0*B0*160.21766208/M)
        g=1-(np.tanh(4*(T-T_D0)/T_D0)/2.0+0.5)/3.0 #2/3 for high T, 1 for low T
        gamma = -g + 0.5*(1+Bp)
        x = np.array(x)
        V = np.linspace(V0 - 0.1*V0,V0 + 0.1*V0,NUM_VOL_POINTS)
        F_temp=[]
        for Vi in V:
            E = E0 + 9*V0*B0/16*(((V0/Vi)**(2.0/3.0)-1)**3*Bp + ((V0/Vi)**(2.0/3.0)-1)**2*(6-4*(V0/Vi)**(2.0/3.0)))
            T_D = T_D0*(V0/Vi)**gamma
            x = T_D/T
            #t = np.linspace(0.000001,x,10000)
            #Deb_func = 3.0 / x**3 * np.trapz(t**3 / (np.exp(t) - 1), t)
            F_vib = 9.0/8.0*kB*T_D - kB*T*Debye(x) + 3*kB*T*np.log(1-np.exp(-(T_D/T)))
            F = E + F_vib
            F_temp.append(F)
        eos = EquationOfState(V,F_temp)
        vol_eq, F_eq, B_eq = eos.fit()
        return F_eq
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(x)
        raise

def get_all_values_ord(op_name):
    q=[]
    for qnty in quantities:
        q.append(ordered[qnty][op_name])
    return q[0],q[1],q[2],q[3]

def get_all_values_part_ord(op_name):       #not needed
    q=[]
    for qnty in quantities:
        q.append(part_ordered[qnty][op_name])
    return q[0],q[1],q[2],q[3]

# ordered_phase_free_energy={}
# for op_name in ordered_phases:
#     E0,V0,B,Bp=get_all_values_ord(op_name)
#     ordered_phase_free_energy[op_name]=Free_energy(ordered_phases_dict[op_name],T,E0,V0,B,Bp)

def Total_Free_Energy_T(x0,T,ordered_phase_free_energy,energy_noise):
    x0=np.array(x0)
    ord_x0 = x0[N*(N+1):]
    x0 = x0[:N*(N+1)].reshape(-1,number_of_phases)
    phase_fraction = x0[0]
    phase_content=x0[1:]  # per-phase conc of elements
    F=0
    for j in range(1,number_of_phases+1): # sum over phases   
        el_c=phase_content[j-1]
        if sum(el_c)>0:
            E0,V0,B0,Bp=get_all_values(el_c)
            F+=phase_fraction[j-1]*(Free_energy(el_c,T,E0,V0,B0,Bp) + energy_noise[0])
            el_c=el_c[el_c>0]
            S_conf = -kB*np.sum(el_c * np.log(el_c))
            F=F-phase_fraction[j-1]*T*S_conf
        
    for j, (c, ord_phase_name) in enumerate(zip(ord_x0, allphases)):
        F+=c*(ordered_phase_free_energy[ord_phase_name] + energy_noise[number_of_phases+j])
        
    return F


# # Main program

# In[48]:

print("BEGINNING PHASE DIAGRAM")
sys.stdout.flush()


# In[49]:

np.random.seed(8)


# In[50]:

elements_order=['Co','Cr','Mo','Nb','Ta','W']


# In[51]:

T_high=max(Ts)


# Import fitted parameters

# In[52]:

print("Readinfg the data from:",DATA_PATH)
sys.stdout.flush()


# In[53]:

with open(os.path.join(DATA_PATH,'E_binary.dat'), 'rb') as handle:
    E_bin_dict = pickle.loads(handle.read())
with open(os.path.join(DATA_PATH,'E_ternary.dat'), 'rb') as handle:
    E_tern_dict = pickle.loads(handle.read())
with open(os.path.join(DATA_PATH,'V_binary.dat'), 'rb') as handle:
    V_bin_dict = pickle.loads(handle.read())
with open(os.path.join(DATA_PATH,'V_ternary.dat'), 'rb') as handle:
    V_tern_dict = pickle.loads(handle.read())
with open(os.path.join(DATA_PATH,'B_binary.dat'), 'rb') as handle:
    B_bin_dict = pickle.loads(handle.read())
with open(os.path.join(DATA_PATH,'B_ternary.dat'), 'rb') as handle:
    B_tern_dict = pickle.loads(handle.read())
with open(os.path.join(DATA_PATH,'Bp_binary.dat'), 'rb') as handle:
    Bp_bin_dict = pickle.loads(handle.read())
with open(os.path.join(DATA_PATH,'Bp_ternary.dat'), 'rb') as handle:
    Bp_tern_dict = pickle.loads(handle.read())
bin_dict={'E': E_bin_dict, 'V': V_bin_dict, 'B': B_bin_dict, 'Bp': Bp_bin_dict}
tern_dict={'E': E_tern_dict, 'V': V_tern_dict, 'B': B_tern_dict, 'Bp': Bp_tern_dict}

# with open('data/unaries.dat', 'rb') as handle:
#     df_u = pickle.loads(handle.read())
df_u = pd.read_csv(os.path.join(DATA_PATH,"unaries.csv"))
df_u2=df_u.copy()
df_u['E']=0 #energies of unaries in bcc set to zero


# Nominal concentration

# In[54]:

# elements_df=mendeleev.get_table("elements")
# mass_dict=elements_df.set_index("symbol")["atomic_weight"].to_dict()
mass_dict = {'H': 1.008, 'He': 4.002602, 'Li': 6.94, 'Be': 9.0121831, 'B': 10.81, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998403163, 'Ne': 20.1797, 
            'Na': 22.98976928, 'Mg': 24.305, 'Al': 26.9815385, 'Si': 28.085, 'P': 30.973761998, 'S': 32.06, 'Cl': 35.45, 'Ar': 39.948, 'K': 39.0983, 'Ca': 40.078, 
            'Sc': 44.955908, 'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961, 'Mn': 54.938044, 'Fe': 55.845, 'Co': 58.933194, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.38, 
            'Ga': 69.723, 'Ge': 72.63, 'As': 74.921595, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90584, 'Zr': 91.224, 'Nb': 92.90637, 
            'Mo': 95.95, 'Tc': 97.90721, 'Ru': 101.07, 'Rh': 102.9055, 'Pd': 106.42, 'Ag': 107.8682, 'Cd': 112.414, 'In': 114.818, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.6, 
            'I': 126.90447, 'Xe': 131.293, 'Cs': 132.90545196, 'Ba': 137.327, 'La': 138.90547, 'Ce': 140.116, 'Pr': 140.90766, 'Nd': 144.242, 'Pm': 144.91276, 'Sm': 150.36, 
            'Eu': 151.964, 'Gd': 157.25, 'Tb': 158.92535, 'Dy': 162.5, 'Ho': 164.93033, 'Er': 167.259, 'Tm': 168.93422, 'Yb': 173.045, 'Lu': 174.9668, 'Hf': 178.49, 'Ta': 180.94788, 
            'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217, 'Pt': 195.084, 'Au': 196.966569, 'Hg': 200.592, 'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.9804, 'Po': 209.0, 'At': 210.0, 
            'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Th': 232.0377, 'Pa': 231.03588, 'U': 238.02891, 'Np': 237.0, 'Pu': 244.0, 'Am': 243.0, 'Cm': 247.0, 'Bk': 247.0, 'Cf': 251.0, 
            'Es': 252.0, 'Fm': 257.0, 'Md': 258.0, 'No': 259.0, 'Lr': 262.0, 'Rf': 267.0, 'Db': 268.0, 'Sg': 271.0, 'Bh': 274.0, 'Hs': 269.0, 'Mt': 276.0, 'Ds': 281.0, 'Rg': 281.0, 
            'Cn': 285.0, 'Nh': 286.0, 'Fl': 289.0, 'Mc': 288.0, 'Lv': 293.0, 'Ts': 294.0, 'Og': 294.0}

# In[55]:

# conc=[conc_dict['Co'],conc_dict['Cr'],conc_dict['Mo'],conc_dict['Nb'],conc_dict['Ta'],conc_dict['W']]
# conc = np.array(conc)
# conc=conc[conc>0]
conc = np.array([c for el, c in conc_dict.items() if c>0])

N=number_of_phases=len(conc)
elements = [x for x in elements_order if (x in conc_dict) and (conc_dict[x] != 0)]


# In[56]:

to_be_del=[]
j=0
for i in df_u2['composition'].index:
    if not df_u2['composition'][i] in elements:
        to_be_del.append(j)
    j=j+1
to_be_del=list(set(to_be_del))
df_u2=df_u2.drop(df_u2.index[to_be_del])


# Ordered phases

# In[57]:

df_ord=pd.read_csv(os.path.join(DATA_PATH,"filtered_struc.csv"))
df_ord_data=pd.read_csv(os.path.join(DATA_PATH,"data_ordered.csv"))
df_ord=df_ord.drop('id',axis=1).rename(columns={'Unnamed: 0': 'id'})
df_ord['id']=df_ord.index
df_ord=pd.merge(df_ord,df_ord_data,on='id')
df_ord_unique=df_ord.drop_duplicates(subset=['prefix'])
#df_ord_unique=df_ord_unique[df_ord_unique['E_f']<thresh]


# In[58]:

df_ord_unique['Comp_dict']=df_ord_unique['Composition'].map(split_name)


# In[59]:

to_be_del=[]
j=0
for i in df_ord_unique['Comp_dict'].index:
    for key,v in df_ord_unique['Comp_dict'][i].items():
        if not key in elements:
            to_be_del.append(j)
        if v==1:
            to_be_del.append(j)
    j=j+1
to_be_del=list(set(to_be_del))
df_ord_new=df_ord_unique.drop(df_ord_unique.index[to_be_del])
for el in elements:
    df_ord_new[el]=df_ord_new['Comp_dict'].map(lambda s:s[el])


# In[60]:

part_ord_indices=[60,61,62,63,64]
df_part_ord=pd.DataFrame()
for i in part_ord_indices:
    if i in df_ord_new['folder_name']:
        df_part_ord=df_part_ord.append(df_ord_new[df_ord_new['folder_name']==i])
        df_ord_new=df_ord_new.drop(i)


# ## Free Energy

# In[61]:

quantities=['E','V','B','Bp']


# In[62]:

unary_dict={}
for qnty in quantities:
    unary_dict[qnty]=df_u.set_index("composition")[qnty].to_dict()


# In[63]:

DEBEY_FILE_NAME = "debye_sampling.npy"
if os.path.isfile(DEBEY_FILE_NAME):
    debye_sampling_data=np.load(DEBEY_FILE_NAME)
    print("Cached Debye are loaded from ", DEBEY_FILE_NAME)
else:

    X = []
    Y = []
    for i in range(0,100000):
        x = 0.001 + i*0.01
        X.append(x)
        t = np.linspace(0.000001,x,10000)
        y = 3.0 / x**3 * np.trapz(t**3 / (np.exp(t) - 1), t)
        Y.append(y)
    debye_sampling_data=np.array([X,Y])
    np.save(DEBEY_FILE_NAME,debye_sampling_data)
    print("Cached Debye sampling stored to", DEBEY_FILE_NAME)

X,Y = debye_sampling_data

tck = interpolate.splrep(X, Y, s=0)
ynew = interpolate.splev(X, tck, der=0)

def Debye(x):
    return interpolate.splev(x, tck, der=0)


# ### Filtering ordered structures close to convex hull

# In[64]:

reference_energies_dict=df_u2.set_index("composition")["E"].to_dict()

df_u2['E_f']=0
df_u2['F_f']=0

df_u2["prefix"]=df_u2["composition"]

for el in elements:
    df_u2[el]=df_u2["c"+el]

df_ord_new["E_f_OQMD"] = df_ord_new["E_f"]

df_ord_new["E_f"]=df_ord_new.apply(compute_formation_energy, axis=1)

df_ord_new["dE_us_oqmd"] = df_ord_new["E_f"]-df_ord_new["E_f_OQMD"]

df_ord_new["F_high_T"]=df_ord_new.apply(compute_free_energy, axis=1)



df_u2["Comp_dict"]=df_u2.apply(make_compdictio, axis=1)

df_u2["F_high_T"]=df_u2.apply(compute_free_energy, axis=1)

reference_free_energies_dict=df_u2.set_index("composition")["F_high_T"].to_dict()

df_ord_new["F_f"]=df_ord_new.apply(compute_formation_free_energy, axis=1)

convex_columns=["E_f","F_f","prefix"]+elements

conv_df=pd.concat([df_ord_new[convex_columns], df_u2[convex_columns]], axis=0)


# In[ ]:




# In[65]:

for en in ["E","F"]:
    points=conv_df[elements[:-1]+[en+"_f"]]
    points = points.values
    hull=ConvexHull(points)
    point_to_chull_distane_list=[]
    for p in points:
        p_x=p[:-1]
        p_en=p[-1]

        distance_to_convex_hulls = []

        for eq in hull.equations:

            #c1 x1 + c2 cx + ... + c(n_el) x(n_el) + d *En + offset = 0
            # En == -(offset +(c1 x1 + c2 cx + ... + c(n_el) x(n_el) )) / d  , if d != 0
            # if d==0: check if point on the line, then dist = zero, else - dist=inf
            d = eq[-2]
            cs = eq[0:-2]
            off=eq[-1]
            if d!=0:
                conv_line_en = -(off + np.dot(p_x, cs)) / d
            dE=p_en - conv_line_en
            if dE>=0:
                distance_to_convex_hulls.append(dE)

        dist_to_conv_hull=min(distance_to_convex_hulls)
        point_to_chull_distane_list.append(dist_to_conv_hull)
    conv_df[en+"_hull"] = point_to_chull_distane_list


# In[66]:

near_chull_entries_df_E=conv_df[conv_df["E_hull"]<thresh]
phases_near_chull_E = near_chull_entries_df_E["prefix"].tolist()
near_chull_entries_df_F=conv_df[conv_df["F_hull"]<thresh]
phases_near_chull_F = near_chull_entries_df_F["prefix"].tolist()
phases_near_chull=set(phases_near_chull_E+phases_near_chull_F)


# In[67]:

print("Ordered phases after convex-hull filtering:", len(phases_near_chull))


# In[68]:

df_ord_new=df_ord_new[df_ord_new["prefix"].map(lambda p: p in phases_near_chull)]


# In[69]:

ordered_phases=[]
part_ordered_phases=[]
ordered_phases_dict={}
ordered_phase_energy={}
ordered_phase_volume={}
ordered_phase_B={}
ordered_phase_Bp={}
for name in df_ord_new['prefix']:
    ordered_phases.append(name)
    compo=[]
    for el in elements:
        compo.append(df_ord_new[df_ord_new['prefix']==name][el].values[0])
    ordered_phases_dict[name]=compo
    E=df_ord_new[df_ord_new['prefix']==name]['E'].values[0]/df_ord_new[df_ord_new['prefix']==name]['N_AT'].values[0]
    for i in range(0,len(elements)):
        E=E-df_u2[df_u2['c'+elements[i]]==1]['E'].values[0]*compo[i]
    ordered_phase_energy[name]=E
    V=df_ord_new[df_ord_new['prefix']==name]['V'].values[0]/df_ord_new[df_ord_new['prefix']==name]['N_AT'].values[0]
    ordered_phase_volume[name]=V
    B=df_ord_new[df_ord_new['prefix']==name]['B'].values[0]
    ordered_phase_B[name]=B
    Bp=df_ord_new[df_ord_new['prefix']==name]['Bp'].values[0]
    ordered_phase_Bp[name]=Bp

if N>2:
    for name in df_part_ord['prefix']:
        part_ordered_phases.append(name)
        compo=[]
        for el in elements:
            compo.append(df_part_ord[df_part_ord['prefix']==name][el].values[0])
        ordered_phases_dict[name]=compo
        E=df_part_ord[df_part_ord['prefix']==name]['E'].values[0]/df_part_ord[df_part_ord['prefix']==name]['N_AT'].values[0]
        for i in range(0,len(elements)):
            E=E-df_u2[df_u2['c'+elements[i]]==1]['E'].values[0]*compo[i]
        ordered_phase_energy[name]=E
        V=df_part_ord[df_part_ord['prefix']==name]['V'].values[0]/df_part_ord[df_part_ord['prefix']==name]['N_AT'].values[0]
        ordered_phase_volume[name]=V
        B=df_part_ord[df_part_ord['prefix']==name]['B'].values[0]
        ordered_phase_B[name]=B
        Bp=df_part_ord[df_part_ord['prefix']==name]['Bp'].values[0]
        ordered_phase_Bp[name]=Bp

allphases=ordered_phases+part_ordered_phases
ordered={'E': ordered_phase_energy, 'V': ordered_phase_volume, 'B': ordered_phase_B, 'Bp': ordered_phase_Bp}
K = len(ordered_phases)+len(part_ordered_phases)


# ## Constraints

# Boundary constraint: all variables between 0 and 1

# In[70]:

bounds=Bounds([0]*(N*(N+1)+K),[1]*(N*(N+1)+K)) #phase fractions and concentrations between 0 and 1


# Linear constraint: phase fractions and concentrations sum to 1

# In[71]:

lc=[]
lc.append([1]*number_of_phases+[0]*(number_of_phases**2)+[1]*K)
for i in range(0,number_of_phases):
    #lc.append([0]*(number_of_phases)+([0]*i+[1]+[0]*(number_of_phases-i-1))*number_of_phases)
    lc.append([0]*(number_of_phases)+[0]*N*i + [1]*N +[0]*(N**2-N-N*i)+[0]*K )
lc=np.array(lc)


# In[72]:

linear_constraint=LinearConstraint(lc,[1]*(number_of_phases+1),[1]*(number_of_phases+1)) #sum of phase fractions and sum of concentrations in each phase is 1


# Nonlinear constraint: mass conservation + Gibbs phase rule

# In[73]:

nonlinear_constraint = NonlinearConstraint(cons_f, 0, 0, jac=cons_J_vec, hess=SR1())


# In[74]:

nonlinear_constraint_gibbs = NonlinearConstraint(gibbs, 1, N, jac=gibbs_J, hess=SR1())


# Constraints for SLS

# In[75]:

eq_cons = {'type': 'eq', 'fun': lambda x: np.array([linf2(x)]+list(linf(x))+list(cons_f(x))+list(gibbs(x))), 'jac': lambda x: np.array(list(lc)+list(cons_J_vec(x))+list(gibbs_J))}


# # Initial guesses pool generation

# Initial condition; x0=[phase_fractions_disordered; c_el1_dis_phase_1, c_el2_dis_phase_1, ..., c_elN_dis_phase_N; phase_fractions_ordered]

# In[78]:

configurations_to_try = []

############################################33
print("Adding solid solution...", end="")
x0 = [1] + [0] * (N - 1) + list(conc) * N + [0] * K  # start with a one-phase solid solution
configurations_to_try.append(x0)
print("done")

############################################33
print("Adding ordered phases...",end="")
for i in range(0, K):
    x0 = [0] + [0] * (N - 1) + list(conc) * N + [0] * i + [1] + [0] * (K - i - 1)
    rat = max(ordered_phases_dict[allphases[i]] / conc)
    x0[N * (N + 1) + i] = x0[N * (N + 1) + i] / rat
    x0[0] = 1 - x0[N * (N + 1) + i]
    if x0[0] > 0:
        for j in range(0, N):
            x0[N + j] = (conc[j] - ordered_phases_dict[allphases[i]][j] / rat) / x0[0]
    configurations_to_try.append(x0)
print("done")
np.random.seed(42)
print("Adding phases with random concentrations...")
it=0
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    for res in executor.map(work_fullfill_constraints_1, range(N_hops)):
        x0 = res.x
        configurations_to_try.append(x0)
        # print('.', end='')
        it+=1        
        print("{}/{}".format(it,N_hops))
        sys.stdout.flush()
print()

############################################33
print("Adding random guessed phases...")
it=0
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    for res in executor.map(work_fullfill_constraints_2, range(N_hops)):
        x0 = res.x
        configurations_to_try.append(x0)
        # print('.', end='')
        it+=1        
        print("{}/{}".format(it,N_hops))
        sys.stdout.flush()
print()
############################################33
print("Adding pairs of ordered phases...")
for i in range(0,K-1):
    for j in range(i+1,K):
        x0 = [0]*N + list(conc)*N + [0]*i+[0.5]+[0]*(j-i-1)+[0.5]+[0]*(K-j-1)
        res, success = fullfill_constraints_2(x0)
        x0 = res.x
        if success:
            configurations_to_try.append(x0)
            print("Added ",allphases[i]," and ",allphases[j])
print()
print(len(configurations_to_try), " initial guesses generated")


# # main optimization

# In[79]:

config_columns=["Disord_phase_"+str(i+1) for i in range(N)] + ["Phase_"+str(i+1)+"/"+el for i in range(N) for el in elements] + allphases


# In[80]:

config_min = []
energy_min = []
all_energies = []
all_configs = []
all_Ts = []
all_noise_attempts = []
energy_noises = [[0] * (N + K)]

for noise_att in range(N_noise_attempts - 1):
    energy_noise = list(
        np.random.randn(number_of_phases) * sigma_disord) + list(
            np.random.randn(len(allphases)) * sigma_ord)
    energy_noises.append(energy_noise)

temp_it = 0
for T in Ts:
    temp_it+=1
    print("****** T=", T, "******")

    ordered_phase_free_energy = {}
    for op_name in ordered_phases:
        E0, V0, B, Bp = get_all_values_ord(op_name)
        ordered_phase_free_energy[op_name] = Free_energy(
            ordered_phases_dict[op_name], T, E0, V0, B, Bp)
    if N > 2:
        for op_name in part_ordered_phases:
            f = 1
            if op_name == 'B4_MoNbTaW':
                f = 2
            E0, V0, B, Bp = get_all_values_ord(op_name)
            ordered_phase_free_energy[op_name] = Free_energy(
                ordered_phases_dict[op_name], T, E0, V0, B,
                Bp) - 1/2 * f * kB * T * np.log(2)

    noise_attempts_energies = []
    noise_attempts_configs = []
    noise_attempts_temperatures = []
    noise_attempts_index = []
    for noise_att in range(N_noise_attempts):
        print("Random energy noise attempt #", noise_att + 1)
        #energy_noise[number_of_phase+len(allphases)]
        energy_noise = energy_noises[noise_att]
        if np.mean(np.abs(energy_noise))>0:
            print("energy_noise=", energy_noise)
        Total_Free_Energy_func = partial(
            Total_Free_Energy_T,
            T=T,
            ordered_phase_free_energy=ordered_phase_free_energy,
            energy_noise=energy_noise)

        def work(x0):
            with try_interrupt_block(stop=False):
                res = minimize_gibbs_free_energy(Total_Free_Energy_func, x0)
                return res

        energy = []
        config = []

        print("Starting optimization...")
        it = 0
        n_tot_it = len(configurations_to_try)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for res in executor.map(work, configurations_to_try):
                if res is not None and res.success:
                    #print('Done: success= ', res.success, ', free energy= ', res.fun)
                    energy.append(res.fun)
                    config.append(res.x)
                    # print(".", end="")
                    it+=1
                    print("{}/{} configurations, noise attempt {}/{}, temperature {}/{} ".format(it,n_tot_it,
                                noise_att + 1, N_noise_attempts,
                                temp_it, len(Ts)
                                )
                            )
                    sys.stdout.flush()
        print()

        noise_attempts_temperatures.append([T] * len(energy))
        noise_attempts_energies.append(energy)
        noise_attempts_configs.append(config)
        noise_attempts_index.append([noise_att] * len(energy))

        min_ind = np.argmin(energy)
        opt_x = config[min_ind]
        opt_energy = energy[min_ind]

        #config_min.append(opt_x)
        #energy_min.append(opt_energy)
        #print("Optimal solution: ", opt_x)
        #pprint_configuration(opt_x)
        #print("Free energy: ", opt_energy)
        configurations_to_try.append(opt_x)

    all_energies.append(noise_attempts_energies)
    all_configs.append(noise_attempts_configs)
    all_Ts.append(noise_attempts_temperatures)
    all_noise_attempts.append(noise_attempts_index)

    data_dict = {"T": [], "Energy": [], "Config": [], "noise_attempt": []}
    for i in range(len(all_energies)):
        for cur_en, cur_conf, cur_noise_attempt, cur_temp in zip(
                all_energies[i], all_configs[i], all_noise_attempts[i],
                all_Ts[i]):
            data_dict['T'] += cur_temp  #[T]*len(cur_en)
            data_dict['Energy'] += cur_en
            data_dict['Config'] += cur_conf
            data_dict[
                'noise_attempt'] += cur_noise_attempt  #[noise_attempt]*len(cur_en)

    config_df = pd.DataFrame(np.vstack(data_dict["Config"]),
                             columns=config_columns)
    opt_df = pd.DataFrame(data_dict).drop("Config", axis=1)

    total_df = pd.concat((opt_df, config_df), axis=1).sort_values(["noise_attempt","Energy"],ascending=True)
    total_df.to_csv("total_data.tsv", sep="\t", index=None)



    curr_energies = all_energies[-1]
    curr_configs = all_configs[-1]

    best_curr_it_configs = []
    best_curr_it_energies = []
    for curr_it_energies, curr_it_config in zip(curr_energies, curr_configs):
        cur_min_it_argmin = np.argmin(curr_it_energies)
        opt_en = curr_it_energies[cur_min_it_argmin]
        opt_config = curr_it_config[cur_min_it_argmin]
        best_curr_it_configs.append(opt_config)
        best_curr_it_energies.append(opt_en)

    best_curr_it_configs = np.array(best_curr_it_configs)
    best_curr_it_energies = np.array(best_curr_it_energies)

    print("Mean best energy: ", best_curr_it_energies.mean(), "+/-",
          best_curr_it_energies.std(), 'eV/atom')
    print("Mean best config: ")
    pprint_configuration(best_curr_it_configs.mean(axis=0))
    best_curr_it_configs_std= best_curr_it_configs.std(axis=0)
    if np.sum(best_curr_it_configs_std)>0:
        print("\n+/-STD:\n")
        pprint_configuration(best_curr_it_configs.std(axis=0))
    print()


# In[ ]:



