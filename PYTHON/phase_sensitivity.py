'''

author: Bin Li
This is to test the phase sensitivity of the low rank SNNs when doing go-nogo task
2024-11-7

'''
# import necessary libraries
import torch
import numpy as np
from scipy.signal import hilbert

from functions import  load_config_yaml
from functions import Generate_Vectors, Generate_RandomMatrix
from lowranksnn import LowRankSNN
import csv
import datetime
# Read the configuration file
config = load_config_yaml('config_test_phase_sensitivity.yaml')

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

num_phase = config['num_phase']
trails = config['trails']

for trail in range(trails):
    # Initialiazation
    LRSNN = LowRankSNN(N_E=N_E,N_I=N_I,taud_E=taud_E,taud_I=taud_I,RS=RS)
    # Go_NoGo Task
    # Prepare the Low Rank Connectivity (Rank = 1), Stimuli and Readout Vector
    m, n, Sti_nogo = Generate_Vectors(N, mu, si)
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
    # # count the number of values outside the range of 0 and 1
    # print('Number of values outside the range of 0 and 1: ', torch.sum(LRSNN.conn>1)+torch.sum(LRSNN.conn<0))
    # # ratio of values outside the range of 0 and 1 to the total number of values
    # print('Ratio of values outside the range of 0 and 1 to the total number of values: ', (torch.sum(LRSNN.conn>1)+torch.sum(LRSNN.conn<0))/(N_E+N_I)**2)
    LRSNN.conn[LRSNN.conn>1] = 1
    LRSNN.conn[LRSNN.conn<0] = 0

    # 1st simulation: get the first zero phase time after 100 ms (use hilbert transform)

    T = T_pre+T_sti+T_after # length of Period time (ms）

    Input_go = torch.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt))) #size:(N,time)
    Input_go[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = IS*Sti_go
    Input_nogo = torch.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt)))
    Input_nogo[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = IS*Sti_nogo

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
    _, _, g_ref, _, _, _ = LRSNN(dt,bias)

    g_ref_II = g_ref[-1]
    g_ref_II = g_ref_II.cpu().detach().numpy()

    # do hilbert transform to get the phase of the conductance
    signal = np.mean(g_ref_II, axis=0)[int(T_pre/dt):]
    # centralize the signal
    signal = signal - np.mean(signal)
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.angle(analytic_signal)  # 相位信息

    # find out the first minimum phase
    # take phase_start as -pi, and phase_end as pi
    flag = 1
    for i in range(len(instantaneous_phase)):
        if flag == 1 and instantaneous_phase[i]-instantaneous_phase[i+1]>np.pi:
            phase_start = instantaneous_phase[i+1]
            phase_start_ind = i+1
            flag = 0
            continue
        if flag == 0 and instantaneous_phase[i]-instantaneous_phase[i+1]>np.pi:
            phase_end = instantaneous_phase[i]
            phase_end_ind = i
            break

    phases_eff = np.linspace(phase_start, phase_end, num_phase)

    def nearest_phase_ind(arr, phase_target):
        return np.argmin(np.abs(arr-phase_target))

    phases_eff_ind = phase_start_ind + np.array([nearest_phase_ind(instantaneous_phase[phase_start_ind:phase_end_ind+1],phase_target) for phase_target in phases_eff])
    phases_eff = instantaneous_phase[phases_eff_ind]

    phases_eff_times = phases_eff_ind*dt # the time of the effective phases after T_pre (ms)

    #simulation: get the reaction time for different phases
    #store the reaction time for different phases
    reaction_times = []

    for T_phase in phases_eff_times:
        T_pre += T_phase
        T = T_pre+T_sti+T_after # length of Period time (ms）

        Input_go = torch.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt))) #size:(N,time)
        Input_go[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = IS*Sti_go
        Input_nogo = torch.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt)))
        Input_nogo[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = IS*Sti_nogo

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
        Out_go, V_go, g_go, I_syn_go, spk_step_go, spk_ind_go = LRSNN(dt,Input_go+bias)
        Out_nogo, V_nogo, g_nogo, I_syn_nogo, spk_step_nogo, spk_ind_nogo = LRSNN(dt,Input_nogo+bias)

        g_go_EE = g_go[1]
        g_nogo_EE = g_nogo[1]

        # define the reaction time as performance
        # reaction time: 从施加刺激开始到输出不为0的时间（或者到go输出大于nogo输出的时间）
        # calculate the time when the output of go exceed the output of nogo
        difference = Out_go - Out_nogo
        exceed_time = torch.nonzero(difference.squeeze()>0)[0].item()*dt
        reaction_time = exceed_time-T_pre
        reaction_times.append(reaction_time)
        print('Phase: ', phases_eff[phases_eff_times==T_phase])
        print('Reaction Time: ', reaction_time, 'ms')
        T_pre -= T_phase

    # Save the reaction times and effective phases in to a csv file (named as 'reaction_times_yymmddhhmmss.csv')
    now = datetime.datetime.now()
    filename = './data_phase_to_reaction_times/reaction_times_'+now.strftime('%y%m%d%H%M%S')+'.csv'
    with open(filename, mode='w') as file:
        writer = csv.writer(file)
        for i in range(len(phases_eff)):
            writer.writerow([phases_eff[i], reaction_times[i]])













