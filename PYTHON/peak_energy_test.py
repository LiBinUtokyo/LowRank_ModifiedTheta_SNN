'''

author: Bin Li
This is to test the peak energy of the output in Go-Nogo task under different network states
2025-5-7

'''
# import necessary libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import hilbert

from functions import Generate_Vectors, Generate_RandomMatrix
from functions import show_mn, show_conn
from functions import Draw_Output, Draw_Conductance,  load_config_yaml, Draw_RasterPlot, Draw_Voltage, Draw_Projection, save_model
from functions import plot_peak_envelope, peak_envelope
from functions import load_init
from lowranksnn import LowRankSNN

from pathlib import Path
import os
import csv
import datetime
import yaml

# stationary state
# Read the configuration file
config = load_config_yaml('./configures/config_peakenergy_stationary.yaml')

N_E = config['N_E']
N_I = config['N_I']
N = N_E + N_I
P_EE = config['P_EE']
P_EI = config['P_EI']
P_IE = config['P_IE']
P_II = config['P_II']
factor_mn = config['factor_mn'] # 组合成conn时乘在lowrank matrix上的常數
RS = config['RandomStrength'] # 组合成conn时乘在random matrix上的常數

taud_E = config['taud_E']
taud_I = config['taud_I']

eta_E = config['eta_E']
eta_I = config['eta_I']
delta_E = config['delta_E']
delta_I = config['delta_I']

mu = config['mu']
si = config['sigma']

si_rand = config['sigma_rand']
dt = config['dt'] #(ms/step)
T_pre = config['T_pre'] # length of time before sti (ms)
T_sti = config['T_sti'] # length of time for sti (ms)
T_after = config['T_after'] # length of time after sti (ms)

IS = config['InputStrength'] #Input Strength (maybe chage to norm in the future)

color_Go = config['color_Go']
color_Nogo = config['color_Nogo']

# num_phase = config['num_phase']
trails = config['trails']

for trail in range(trails):
    # Initialiazation
    LRSNN = LowRankSNN(N_E=N_E,N_I=N_I,taud_E=taud_E,taud_I=taud_I,RS=RS)
    # Go_NoGo Task
    # Prepare the Low Rank Connectivity (Rank = 1), Stimuli and Readout Vector
    # m, n, Sti_nogo = Generate_Vectors(N, mu, si)
    i = 0
    while i<100:
        m_test, n_test, Sti_nogo_test = Generate_Vectors(N, mu, si)
        if torch.sum(m_test[:N_E]).abs() < 1 and torch.sum(n_test[:N_E]).abs() < 1 and torch.sum(Sti_nogo_test[:N_E]).abs() < 1:
            print(N,mu,si)
            # sum of all the element in m and n and Sti_nogo_test
            print(torch.sum(m_test[:N_E]))
            print(torch.sum(n_test[:N_E]))
            print(torch.sum(Sti_nogo_test[:N_E]))
            print('i:',i)
            print('-----------------------------------')
            m = m_test
            n = n_test
            Sti_nogo = Sti_nogo_test
            break
        i += 1
        if i == 100:
            i = 0
            print('did not find the suitable m, n, Sti_nogo')

    m[N_E:] = 0
    n[N_E:] = 0
    Sti_nogo[N_E:] = 0
    Sti_go = n.clone()
    W_out = m.clone()
    W_rank1 = factor_mn*torch.ger(m.squeeze(), n.squeeze())
    conn_rand = Generate_RandomMatrix(N_E, N_I, P_EE, P_EI, P_IE, P_II, W_rank1, sigma = si_rand)

    # Assemble the Network
    LRSNN.add_lowrank(W_rank1, W_out)
    LRSNN.add_random(conn_rand)
 
    LRSNN.conn[LRSNN.conn>1] = 1
    LRSNN.conn[LRSNN.conn<0] = 0

    #simulation: get the reaction time for different phases
    #store the reaction time for different phases

    Input_go_rec = []
    Input_nogo_rec = []
    Out_go_rec = []
    Out_nogo_rec = []

    T = T_pre+T_sti+T_after

    Input_go = torch.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt))) #size:(N,time)
    Input_go[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = IS*Sti_go
    Input_nogo = torch.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt)))
    Input_nogo[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = IS*Sti_nogo

    Input_go_rec.append(Input_go.tolist())
    Input_nogo_rec.append(Input_nogo.tolist())

    # bias current
    bias = torch.zeros_like(Input_go)
    bias[:N_E,:] = (eta_E+delta_E*torch.tan(torch.tensor(np.pi*(np.arange(1,N_E+1)/(N_E+1)-1/2)))).unsqueeze(1)
    bias[N_E:,:] = (eta_I+delta_I*torch.tan(torch.tensor(np.pi*(np.arange(1,N_I+1)/(N_I+1)-1/2)))).unsqueeze(1)

    #将模型及相应属性移动到GPU
    device = torch.device('cuda:0')
    LRSNN = LRSNN.to(device)
    Input_go = Input_go.to(device)
    Input_nogo = Input_nogo.to(device)
    bias = bias.to(device)

        # Start Simulation
    Out_go, _,_,_, _,_,_,_ = LRSNN(dt,Input_go+bias)
    Out_nogo, _,_,_, _,_,_,_ = LRSNN(dt,Input_nogo+bias)

    Out_go_rec.append(Out_go.cpu().tolist())
    Out_nogo_rec.append(Out_nogo.cpu().tolist())

    Input_go_rec = np.array(Input_go_rec)
    Out_go_rec = np.array(Out_go_rec)
    Out_nogo_rec = np.array(Out_nogo_rec)
    now = datetime.datetime.now()
    folder = f'./data_peak_energy_stationary/{now.strftime("%y%m%d%H%M%S")}'
    os.makedirs(folder)
    np.save(folder+'/Input_go_rec'+'.npy', Input_go_rec)
    np.save(folder+'/Input_nogo_rec'+'.npy', Input_nogo_rec)
    np.save(folder+'/Out_go_rec'+'.npy', Out_go_rec)
    np.save(folder+'/Out_nogo_rec'+'.npy', Out_nogo_rec)



# gamma oscillation state
# Read the configuration file
config = load_config_yaml('./configures/config_peakenergy_gamma.yaml')

N_E = config['N_E']
N_I = config['N_I']
N = N_E + N_I
P_EE = config['P_EE']
P_EI = config['P_EI']
P_IE = config['P_IE']
P_II = config['P_II']
factor_mn = config['factor_mn'] # 组合成conn时乘在lowrank matrix上的常數
RS = config['RandomStrength'] # 组合成conn时乘在random matrix上的常數

taud_E = config['taud_E']
taud_I = config['taud_I']

eta_E = config['eta_E']
eta_I = config['eta_I']
delta_E = config['delta_E']
delta_I = config['delta_I']

mu = config['mu']
si = config['sigma']

si_rand = config['sigma_rand']
dt = config['dt'] #(ms/step)
T_pre = config['T_pre'] # length of time before sti (ms)
T_sti = config['T_sti'] # length of time for sti (ms)
T_after = config['T_after'] # length of time after sti (ms)

IS = config['InputStrength'] #Input Strength (maybe chage to norm in the future)

color_Go = config['color_Go']
color_Nogo = config['color_Nogo']

# num_phase = config['num_phase']
trails = config['trails']

for trail in range(trails):
    # Initialiazation
    LRSNN = LowRankSNN(N_E=N_E,N_I=N_I,taud_E=taud_E,taud_I=taud_I,RS=RS)
    # Go_NoGo Task
    # Prepare the Low Rank Connectivity (Rank = 1), Stimuli and Readout Vector
    # m, n, Sti_nogo = Generate_Vectors(N, mu, si)
    i = 0
    while i<100:
        m_test, n_test, Sti_nogo_test = Generate_Vectors(N, mu, si)
        if torch.sum(m_test[:N_E]).abs() < 1 and torch.sum(n_test[:N_E]).abs() < 1 and torch.sum(Sti_nogo_test[:N_E]).abs() < 1:
            print(N,mu,si)
            # sum of all the element in m and n and Sti_nogo_test
            print(torch.sum(m_test[:N_E]))
            print(torch.sum(n_test[:N_E]))
            print(torch.sum(Sti_nogo_test[:N_E]))
            print('i:',i)
            print('-----------------------------------')
            m = m_test
            n = n_test
            Sti_nogo = Sti_nogo_test
            break
        i += 1
        if i == 100:
            i = 0
            print('did not find the suitable m, n, Sti_nogo')

    m[N_E:] = 0
    n[N_E:] = 0
    Sti_nogo[N_E:] = 0
    Sti_go = n.clone()
    W_out = m.clone()
    W_rank1 = factor_mn*torch.ger(m.squeeze(), n.squeeze())
    conn_rand = Generate_RandomMatrix(N_E, N_I, P_EE, P_EI, P_IE, P_II, W_rank1, sigma = si_rand)

    # Assemble the Network
    LRSNN.add_lowrank(W_rank1, W_out)
    LRSNN.add_random(conn_rand)
 
    LRSNN.conn[LRSNN.conn>1] = 1
    LRSNN.conn[LRSNN.conn<0] = 0

    #simulation: get the reaction time for different phases
    #store the reaction time for different phases

    Input_go_rec = []
    Input_nogo_rec = []
    Out_go_rec = []
    Out_nogo_rec = []

    T = T_pre+T_sti+T_after

    Input_go = torch.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt))) #size:(N,time)
    Input_go[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = IS*Sti_go
    Input_nogo = torch.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt)))
    Input_nogo[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = IS*Sti_nogo

    Input_go_rec.append(Input_go.tolist())
    Input_nogo_rec.append(Input_nogo.tolist())

    # bias current
    bias = torch.zeros_like(Input_go)
    bias[:N_E,:] = (eta_E+delta_E*torch.tan(torch.tensor(np.pi*(np.arange(1,N_E+1)/(N_E+1)-1/2)))).unsqueeze(1)
    bias[N_E:,:] = (eta_I+delta_I*torch.tan(torch.tensor(np.pi*(np.arange(1,N_I+1)/(N_I+1)-1/2)))).unsqueeze(1)

    #将模型及相应属性移动到GPU
    device = torch.device('cuda:0')
    LRSNN = LRSNN.to(device)
    Input_go = Input_go.to(device)
    Input_nogo = Input_nogo.to(device)
    bias = bias.to(device)

        # Start Simulation
    Out_go, _,_,_, _,_,_,_ = LRSNN(dt,Input_go+bias)
    Out_nogo, _,_,_, _,_,_,_ = LRSNN(dt,Input_nogo+bias)

    Out_go_rec.append(Out_go.cpu().tolist())
    Out_nogo_rec.append(Out_nogo.cpu().tolist())

    Input_go_rec = np.array(Input_go_rec)
    Out_go_rec = np.array(Out_go_rec)
    Out_nogo_rec = np.array(Out_nogo_rec)
    now = datetime.datetime.now()
    folder = f'./data_peak_energy_gamma/{now.strftime("%y%m%d%H%M%S")}'
    os.makedirs(folder)
    np.save(folder+'/Input_go_rec'+'.npy', Input_go_rec)
    np.save(folder+'/Input_nogo_rec'+'.npy', Input_nogo_rec)
    np.save(folder+'/Out_go_rec'+'.npy', Out_go_rec)
    np.save(folder+'/Out_nogo_rec'+'.npy', Out_nogo_rec)


# low-frequency oscillation state
# Read the configuration file
config = load_config_yaml('./configures/config_peakenergy_lowfrequency.yaml')

N_E = config['N_E']
N_I = config['N_I']
N = N_E + N_I
P_EE = config['P_EE']
P_EI = config['P_EI']
P_IE = config['P_IE']
P_II = config['P_II']
factor_mn = config['factor_mn'] # 组合成conn时乘在lowrank matrix上的常數
RS = config['RandomStrength'] # 组合成conn时乘在random matrix上的常數

taud_E = config['taud_E']
taud_I = config['taud_I']

eta_E = config['eta_E']
eta_I = config['eta_I']
delta_E = config['delta_E']
delta_I = config['delta_I']

mu = config['mu']
si = config['sigma']

si_rand = config['sigma_rand']
dt = config['dt'] #(ms/step)
T_pre = config['T_pre'] # length of time before sti (ms)
T_sti = config['T_sti'] # length of time for sti (ms)
T_after = config['T_after'] # length of time after sti (ms)

IS = config['InputStrength'] #Input Strength (maybe chage to norm in the future)

color_Go = config['color_Go']
color_Nogo = config['color_Nogo']

# num_phase = config['num_phase']
trails = config['trails']

for trail in range(trails):
    # Initialiazation
    LRSNN = LowRankSNN(N_E=N_E,N_I=N_I,taud_E=taud_E,taud_I=taud_I,RS=RS)
    # Go_NoGo Task
    # Prepare the Low Rank Connectivity (Rank = 1), Stimuli and Readout Vector
    # m, n, Sti_nogo = Generate_Vectors(N, mu, si)
    i = 0
    while i<100:
        m_test, n_test, Sti_nogo_test = Generate_Vectors(N, mu, si)
        if torch.sum(m_test[:N_E]).abs() < 1 and torch.sum(n_test[:N_E]).abs() < 1 and torch.sum(Sti_nogo_test[:N_E]).abs() < 1:
            print(N,mu,si)
            # sum of all the element in m and n and Sti_nogo_test
            print(torch.sum(m_test[:N_E]))
            print(torch.sum(n_test[:N_E]))
            print(torch.sum(Sti_nogo_test[:N_E]))
            print('i:',i)
            print('-----------------------------------')
            m = m_test
            n = n_test
            Sti_nogo = Sti_nogo_test
            break
        i += 1
        if i == 100:
            i = 0
            print('did not find the suitable m, n, Sti_nogo')

    m[N_E:] = 0
    n[N_E:] = 0
    Sti_nogo[N_E:] = 0
    Sti_go = n.clone()
    W_out = m.clone()
    W_rank1 = factor_mn*torch.ger(m.squeeze(), n.squeeze())
    conn_rand = Generate_RandomMatrix(N_E, N_I, P_EE, P_EI, P_IE, P_II, W_rank1, sigma = si_rand)

    # Assemble the Network
    LRSNN.add_lowrank(W_rank1, W_out)
    LRSNN.add_random(conn_rand)
 
    LRSNN.conn[LRSNN.conn>1] = 1
    LRSNN.conn[LRSNN.conn<0] = 0

    #simulation: get the reaction time for different phases
    #store the reaction time for different phases

    Input_go_rec = []
    Input_nogo_rec = []
    Out_go_rec = []
    Out_nogo_rec = []

    T = T_pre+T_sti+T_after

    Input_go = torch.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt))) #size:(N,time)
    Input_go[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = IS*Sti_go
    Input_nogo = torch.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt)))
    Input_nogo[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = IS*Sti_nogo

    Input_go_rec.append(Input_go.tolist())
    Input_nogo_rec.append(Input_nogo.tolist())

    # bias current
    bias = torch.zeros_like(Input_go)
    bias[:N_E,:] = (eta_E+delta_E*torch.tan(torch.tensor(np.pi*(np.arange(1,N_E+1)/(N_E+1)-1/2)))).unsqueeze(1)
    bias[N_E:,:] = (eta_I+delta_I*torch.tan(torch.tensor(np.pi*(np.arange(1,N_I+1)/(N_I+1)-1/2)))).unsqueeze(1)

    #将模型及相应属性移动到GPU
    device = torch.device('cuda:0')
    LRSNN = LRSNN.to(device)
    Input_go = Input_go.to(device)
    Input_nogo = Input_nogo.to(device)
    bias = bias.to(device)

        # Start Simulation
    Out_go, _,_,_, _,_,_,_ = LRSNN(dt,Input_go+bias)
    Out_nogo, _,_,_, _,_,_,_ = LRSNN(dt,Input_nogo+bias)

    Out_go_rec.append(Out_go.cpu().tolist())
    Out_nogo_rec.append(Out_nogo.cpu().tolist())

    Input_go_rec = np.array(Input_go_rec)
    Out_go_rec = np.array(Out_go_rec)
    Out_nogo_rec = np.array(Out_nogo_rec)
    now = datetime.datetime.now()
    folder = f'./data_peak_energy_lowfrequency/{now.strftime("%y%m%d%H%M%S")}'
    os.makedirs(folder)
    np.save(folder+'/Input_go_rec'+'.npy', Input_go_rec)
    np.save(folder+'/Input_nogo_rec'+'.npy', Input_nogo_rec)
    np.save(folder+'/Out_go_rec'+'.npy', Out_go_rec)
    np.save(folder+'/Out_nogo_rec'+'.npy', Out_nogo_rec)












