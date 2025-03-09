'''

author: Bin Li
This is to test the phase sensitivity of the low rank SNNs when doing go-nogo task
2024-11-7

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


# Read the configuration file
config = load_config_yaml('./configures/config_test_phase_sensitivity.yaml')

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
    Out_ref, V_ref, [g_ref,g_ref_EE,g_ref_EI,g_ref_IE,g_ref_II],[I_ref_syn,I_ref_syn_EE,I_ref_syn_EI,I_ref_syn_IE,I_ref_syn_II], spk_step_ref, spk_ind_ref, spk_ref, phase_ref = LRSNN(dt,bias)

    # load the values at T_pre
    # to see whether the results are the same
    LRSNN = load_init(LRSNN, T_pre, dt, g_ref, g_ref_EE, g_ref_EI, g_ref_IE, g_ref_II, V_ref, phase_ref, I_ref_syn, I_ref_syn_EE, I_ref_syn_EI, I_ref_syn_IE, I_ref_syn_II, spk_ref)
    # do hilbert transform to get the phase of the conductance
    # g_ref_II_np = g_ref_II.clone().cpu().detach().numpy()
    g_ref_EE_np = g_ref_EE.clone().cpu().detach().numpy()

    # signal = np.mean(g_ref_II_np, axis=0)[int(T_pre/dt):]
    # signal = np.mean(g_ref_II_np, axis=0)[int((T_pre-50)/dt):]
    # signal = np.mean(g_ref_II_np, axis=0) # 从0ms开始，结果应该会更稳定
    signal = np.mean(g_ref_EE_np, axis=0) # 从0ms开始，结果应该会更稳定
    

    # # filter out the high frequency noise in the signal
    # from scipy.signal import butter, lfilter
    # def butter_lowpass_filter(data, cutoff, fs, order=5):
    #     nyquist = 0.5 * fs
    #     normal_cutoff = cutoff / nyquist
    #     b, a = butter(order, normal_cutoff, btype='low', analog=False)
    #     y = lfilter(b, a, data)
    #     return y
    # signal = butter_lowpass_filter(signal, 100, 1000/dt, order=5) # cutoff frequency higher than 100 Hz

    # centralize the signal
    mean_signal = np.mean(signal)
    signal = signal - mean_signal
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)  # 振幅包络
    instantaneous_phase = np.angle(analytic_signal)  # 相位信息
    phase_diff = np.diff(instantaneous_phase)  # 相位的一阶导数，即相位变化率
    t = np.array(range(len(signal)))*dt

    # 寻找第一个周期
    T_pre_ind = int(T_pre/dt)
    phase_diff_T_pre = phase_diff[T_pre_ind:]
    # 找到零相位点（相位跨越 2π 的位置）
    crossings = np.where(phase_diff_T_pre<-3)[0]  # 相位从 pi 到 -pi 跳变的位置
    # print(phase_diff_T_pre[crossings])

    # 确定第一个周期的起始点和结束点
    if len(crossings) >= 2:
        start_index = crossings[0]  # 第一个周期的起始点
        end_index = crossings[1]  # 第一个完整周期的结束点
    else:
        raise ValueError("未能找到完整的周期")

    # 计算第一个周期的起始相位和结束相位
    start_index += 1 # 过了这个就是第一个周期的开始
    end_index += 1 # 这样就不用再加1了
    phase_start = instantaneous_phase[T_pre_ind+start_index]
    phase_end = instantaneous_phase[T_pre_ind+end_index]

    print('First minimun phase:', phase_start)
    print('time:', T_pre+start_index*dt, 'ms')
    print('First maximum phase:', phase_end)
    print('time:', T_pre+end_index*dt, 'ms')
    print('period:', (end_index-start_index)*dt, 'ms')

    from scipy.interpolate import interp1d
    # obtain the corresponding time of the 33 phases
    time = t[T_pre_ind+start_index:T_pre_ind+end_index]  # 时间轴
    phase_period = instantaneous_phase[T_pre_ind+start_index:T_pre_ind+end_index]

    # 生成平均分布的 33 个相位点
    phase_target = np.linspace(-np.pi, np.pi, num_phase)

    # 使用插值方法找到这些相位对应的时间
    interp_func = interp1d(phase_period, time, kind='linear', fill_value="extrapolate")
    time_target = interp_func(phase_target) # the time of the 33 phases after T_pre (ms)

    # 把time_target限定在time的范围内
    time_target[time_target<=time[0]] = time[0]
    time_target[time_target>=time[-1]] = time[-1]
    time_target_ind = (time_target/dt).astype(int) #这里的time_target应该是近似的，所以有可能超过实际的周期时间

    #simulation: get the reaction time for different phases
    #store the reaction time for different phases
    reaction_times = []
    Input_go_rec = []
    Input_nogo_rec = []
    Out_go_rec = []
    Out_nogo_rec = []
    T_pre_origin = T_pre
    T_after_origin = T_after


    for T_phase in time_target-T_pre_origin:
        T_pre = T_phase
        T_after = T_after_origin-T_phase # length of time after sti (ms) for the 2nd simulation
        # T = T_pre+T_sti+T_after # length of Period time (ms）
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

        # Note: initial values has been loaded
        # Start Simulation
        Out_go, _,g_go,_, _,_,_,_ = LRSNN(dt,Input_go+bias)
        Out_nogo, _,g_nogo,_, _,_,_,_ = LRSNN(dt,Input_nogo+bias)

        Out_go_rec.append(Out_go.cpu().tolist())
        Out_nogo_rec.append(Out_nogo.cpu().tolist())

        # g_go_EE = g_go[1]
        # g_nogo_EE = g_nogo[1]

        # define the reaction time as performance
        # reaction time: 从施加刺激开始到输出不为0的时间（或者到go输出大于nogo输出的时间）
        # calculate the time when the output of go exceed the output of nogo
        # difference = Out_go - Out_nogo
        # exceed_time = torch.nonzero(difference.squeeze()>0)[0].item()*dt
        # reaction_time = exceed_time-T_pre
        # reaction_times.append(reaction_time)
        # print('Phase: ', phases_eff[phases_eff_times==T_phase])
        # print('Reaction Time: ', reaction_time, 'ms')
        # print('--------------------------------------------')

    T_pre = T_pre_origin
    T_after = T_after_origin
    # # Save the reaction times and effective phases in to a csv file (named as 'reaction_times_yymmddhhmmss.csv')
    # now = datetime.datetime.now()
    # filename = './data_phase_to_reaction_times/reaction_times_'+now.strftime('%y%m%d%H%M%S')+'.csv'
    # with open(filename, mode='w') as file:
    #     writer = csv.writer(file)
    #     for i in range(len(phases_eff)):
    #         writer.writerow([phases_eff[i], reaction_times[i]])
    Input_go_rec = np.array(Input_go_rec)
    Out_go_rec = np.array(Out_go_rec)
    Out_nogo_rec = np.array(Out_nogo_rec)
    now = datetime.datetime.now()
    folder = f'./data_phase_sensitivity/{now.strftime("%y%m%d%H%M%S")}'
    os.makedirs(folder)
    np.save(folder+'/Input_go_rec'+'.npy', Input_go_rec)
    np.save(folder+'/Input_nogo_rec'+'.npy', Input_nogo_rec)
    np.save(folder+'/Out_go_rec'+'.npy', Out_go_rec)
    np.save(folder+'/Out_nogo_rec'+'.npy', Out_nogo_rec)
    np.save(folder+'/phases'+'.npy', phase_target)
    np.save(folder+'/g_go'+'.npy', g_go)
    np.save(folder+'/g_nogo'+'.npy', g_nogo)













