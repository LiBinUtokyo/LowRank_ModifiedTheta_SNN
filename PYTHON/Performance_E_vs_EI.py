'''
用于对比同一个网络仅有兴奋性和既有兴奋性又有抑制性的任务表现情况
读取保存的模型，做gonogo任务测试，检测其中的N_E和N_I，如果N_I不是0，就把测试结果存在EI组，如果是，就存在E组，然后把N_I变为
0或1000，把N_E变为5000或4000，重复任务测试，把结果存在另一个组,最后把两组数据存在E_VS_EI.csv文件里，第一行是E组，第二行是EI组
'''
import torch
import numpy as np
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import torch.distributions as dist
# import functions
import os
import csv

path_models = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/models/'
path_result = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/'
perf_E = []
perf_EI = []

for filename in os.listdir(path_models):
    #读取模型
    # model = torch.load('/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/models/2024_03_02_07_56.pth')
    model = torch.load(path_models+filename)
    LRSNN = model['model']
    Input_go = model['Input_go'].cpu()
    Input_nogo = model['Input_nogo'].cpu()
    dt = model['dt']
    # print(filename)

    #将模型及相应属性移动到GPU
    device = torch.device('cuda:0')
    LRSNN = LRSNN.to(device)
    Input_go = Input_go.to(device)
    Input_nogo = Input_nogo.to(device)

    # Start Simulation
    Out_go, V_go, g_go, spk_go = LRSNN(dt,Input_go)
    Out_nogo, V_nogo, g_nogo, spk_nogo = LRSNN(dt,Input_nogo)
    Out_go = Out_go.cpu().numpy()
    Out_nogo = Out_nogo.cpu().numpy()
    perf = np.max(Out_go)/np.max(Out_nogo)
    if LRSNN.N_I == 0:
        perf_E.append(perf.item())
        LRSNN.N_I = torch.tensor(1000).to(device)
        LRSNN.N_E = torch.tensor(4000).to(device)
    elif LRSNN.N_I == 1000:
        perf_EI.append(perf.item())
        LRSNN.N_I = torch.tensor(0).to(device)
        LRSNN.N_E = torch.tensor(5000).to(device)
    print('Performance of E-only group: ', perf_E)
    print('Performance of E-I group: ', perf_EI)

    # Start Simulation again
    Out_go, V_go, g_go, spk_go = LRSNN(dt,Input_go)
    Out_nogo, V_nogo, g_nogo, spk_nogo = LRSNN(dt,Input_nogo)
    Out_go = Out_go.cpu().numpy()
    Out_nogo = Out_nogo.cpu().numpy()
    perf = np.max(Out_go)/np.max(Out_nogo)
    if LRSNN.N_I == 0:
        perf_E.append(perf.item())
    elif LRSNN.N_I == 1000:
        perf_EI.append(perf.item())
    print('Performance of E-only group: ', perf_E)
    print('Performance of E-I group: ', perf_EI)

# 保存
with open(path_result+'E_VS_EI.csv','a+',newline='') as f:
    csv.writer(f).writerow(perf_E)
    csv.writer(f).writerow(perf_EI)


















