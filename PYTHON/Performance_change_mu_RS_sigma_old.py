'''
2/26: 为了复现原来的结果，改了几个地方：连接矩阵在最后限制了所有大于一的值为一，随机矩阵的标准差变为根号下。然后IS=3，RS=1
改掉了Gamma分布的bug

2/16: 改为使用pytorch的代码,并测试5000个神经元时的RS为1和0.5的情况

1/23: 关于RS固定时候的值, 代码上面写的是1, 硕士论文里写的是0.5, 两个都试试吧

1/22: N_E改为1000看有没有什么不同以及原来的结果是不是基于1000的

用于重现精密工学会发表Figure1的图
改变Gamma分布的mu,random strength和sigma看performance的变化
N_E = 5000
mu从1到100一步一测 即range(1,101) RS = 0.1(?), sigma = 10
random Strength从 0 到 10,步长为0.5, 即range(0,10.5,0.5) mu = 50, sigma = 10
sigma从1到70 range(71) mu = 50, RS = 0.1(?)

得到的结果存到名为prop_change_mu.csv,prop_change_RS.csv,prop_change_sigma.csv的文件中, 第一行是自变量, 第二行是performance

先保证一次测试的结果和Matlab代码基本一致再开始大规模测试
'''
import lowrankSNN_GPU as lowrankSNN
# import lowrankSNN
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist


# path = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/change_mu_RS_sigma_N_5000_RS_0.5/'
# path = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/change_mu_RS_sigma_N_5000_RS_1/'
# path = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/change_mu_RS_sigma_N_1000_RS_1_GPU/'
# path = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/change_mu_RS_sigma_N_1000_RS_0.5_GPU/'
# path = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/change_mu_RS_sigma_N_1000_RS_1/'
path = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/change_mu_RS_sigma_N_1000_RS_0.5/'
def test(mu,RandomS,sigma):
    # Initialiazation
    # LRSNN = lowrankSNN.LowRankSNN(N_E=5000,N_I=0,RS=RandomS,taud_E=2,taud_I=5)
    LRSNN = lowrankSNN.LowRankSNN(N_E=1000,N_I=0,RS=RandomS,taud_E=2,taud_I=5)
    # LRSNN = lowrankSNN.LowRankSNN(N_E=500,N_I=0,RS=RandomS,taud_E=2,taud_I=5)
    #low rank文献的N=5000
    IS = 3 #Input strength
    # Go_NoGo Task

    # # Prepare the Stimuli and Readout Vector
    # temp = np.random.rand(1,LRSNN.N_E+LRSNN.N_I) #Size (1,N_E) for Sti_go and nogo #这里我想试试把Low Rank加到整个网络上
    # Sti_go = temp.copy()
    # Sti_nogo = temp.copy()
    # W_out = temp.copy()
    # Sti_go[Sti_go>1/30] = 0
    # Sti_nogo[Sti_nogo<14/30] = 0
    # Sti_nogo[Sti_nogo>15/30] = 0
    # W_out[W_out<29/30] = 0

    # Prepare the Stimuli and Readout Vector
    temp = torch.rand(1,LRSNN.N_E+LRSNN.N_I) #Size (1,N_E) for Sti_go and nogo #这里我想试试把Low Rank加到整个网络上
    Sti_go = temp.clone()
    Sti_nogo = temp.clone()
    W_out = temp.clone()
    Sti_go[Sti_go>1/3] = 0
    Sti_nogo[Sti_nogo<1/3] = 0
    Sti_nogo[Sti_nogo>2/3] = 0
    W_out[W_out<2/3] = 0

    # Use Gamma Distribution to generate Stimuli and Readout Vector
    # mean and std of Gamma Distribution(Deside Sti_go,Sti_nogo,W_out,conn_rand)
    # mu = 1
    # 创建Gamma分布
    si = sigma
    b = mu/si**2
    a = mu*b
    gamma_dist = dist.gamma.Gamma(a, b)

    Sti_go[Sti_go!=0] = gamma_dist.sample((len(torch.nonzero(Sti_go)),)) #random.gamma(shape(a), scale(b)=1.0, size=None),这个地方的Gamma分布及其参数选取需要进一步讨论
    Sti_nogo[Sti_nogo!=0] = gamma_dist.sample((len(torch.nonzero(Sti_nogo)),))
    W_out[W_out!=0] = gamma_dist.sample((len(torch.nonzero(W_out)),))
    W_out = np.transpose(W_out) #Size (N_E,1)
    # Low Rank Connectivity (Rank = 1)
    conn_LR = W_out*Sti_go/(LRSNN.N_E+LRSNN.N_I) # 为什么除以神经元总数?
    # Random Connectivity
    # conn_rand = gamma_dist.sample(((LRSNN.N_E+LRSNN.N_I,LRSNN.N_E+LRSNN.N_I))) #这里的Gamma分布取值也需要讨论
    conn_rand = np.abs(np.random.normal(0,np.sqrt(1/(LRSNN.N_E+LRSNN.N_I)),(LRSNN.N_E+LRSNN.N_I,LRSNN.N_E+LRSNN.N_I))) #改回和原来一样的形式
    conn_rand = torch.from_numpy(conn_rand)
    # # Use Gamma Distribution to generate Stimuli and Readout Vector
    # # mean and std of Gamma Distribution(Deside Sti_go,Sti_nogo,W_out,conn_rand)
    # # mu = 1
    # si = sigma
    # b = si**2/mu
    # a = mu/b

    # Sti_go[Sti_go!=0] = np.random.gamma(a,b,len(np.nonzero(Sti_go)[0])) #random.gamma(shape(a), scale(b)=1.0, size=None),这个地方的Gamma分布及其参数选取需要进一步讨论
    # Sti_nogo[Sti_nogo!=0] = np.random.gamma(a,b,len(np.nonzero(Sti_nogo)[0]))
    # W_out[W_out!=0] = np.random.gamma(a,b,len(np.nonzero(W_out)[0]))
    # W_out = np.transpose(W_out) #Size (N_E,1)
    # # Low Rank Connectivity (Rank = 1)
    # conn_LR = W_out*Sti_go/(LRSNN.N_E+LRSNN.N_I) # 为什么除以神经元总数?
    # # Random Connectivity
    # # conn_rand = np.random.gamma(a,b,(LRSNN.N_E+LRSNN.N_I,LRSNN.N_E+LRSNN.N_I)) #这里的Gamma分布取值也需要讨论
    # conn_rand = np.abs(np.random.normal(0,1/(LRSNN.N_E+LRSNN.N_I),(LRSNN.N_E+LRSNN.N_I,LRSNN.N_E+LRSNN.N_I))) #改回和原来一样的形式


    # # Use Folded Gaussian Distribution to generate Stimuli and Readout Vector
    # std_Sti = 2. #Standerd Deviration of Stimuli
    # std_Wout = 2. #Standerd Deviration of readout matrix
    # Sti_go[Sti_go!=0] = np.abs(np.random.normal(0,std_Sti,len(np.nonzero(Sti_go)[0]))) 
    # Sti_nogo[Sti_nogo!=0] = np.abs(np.random.normal(0,std_Sti,len(np.nonzero(Sti_nogo)[0])))
    # W_out[W_out!=0] = np.abs(np.random.normal(0,std_Wout,len(np.nonzero(W_out)[0])))
    # # W_out = np.transpose(W_out) #Size (N,1)

    # conn_LR = W_out*Sti_go/(LRSNN.N_E+LRSNN.N_I) # 为什么除以神经元总数? # Low Rank Connectivity (Rank = 1)
    # conn_rand = np.abs(np.random.normal(0,1/(LRSNN.N_E+LRSNN.N_I),(LRSNN.N_E+LRSNN.N_I,LRSNN.N_E+LRSNN.N_I))) # Random Connectivity

    m = W_out #m = Wout
    n = Sti_go #n = Stigo

    # Assemble the Network
    LRSNN.add_lowrank(conn_LR, W_out)
    LRSNN.add_random(conn_rand)

    LRSNN.conn[LRSNN.conn>1] = 1
    # Show the Network information before simulaiton
    # LRSNN.show_conn()

    dt = 0.01 #(ms/step)
    T_pre = 5 # length of time before sti (ms)
    T_sti = 10 # length of time for sti (ms)
    T_after = 15 # length of time after sti (ms)
    T = T_pre+T_sti+T_after # length of Period time (ms): 30ms

    # Input_go = np.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt))) #size:(N,time)
    # Input_go[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = Sti_go.T
    # Input_nogo = np.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt)))
    # Input_nogo[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = Sti_nogo.T


    Input_go = torch.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt))) #size:(N,time)
    Input_go[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = IS*Sti_go.T
    Input_nogo = torch.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt)))
    Input_nogo[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = IS*Sti_nogo.T
    #将模型及相应属性移动到GPU
    device = torch.device('cuda:0')
    LRSNN = LRSNN.to(device)
    Input_go = Input_go.to(device)
    Input_nogo = Input_nogo.to(device)

    # Simulation
    Out_go, V_go, g_go, spk_go = LRSNN(dt,Input_go*IS)
    Out_nogo, V_nogo, g_nogo, spk_nogo = LRSNN(dt,Input_nogo*IS)

    # # Simulation
    # Out_go, V_go, g_go, spk_go = LRSNN.simulate(dt,Input_go*IS)
    # Out_nogo, V_nogo, g_nogo, spk_nogo = LRSNN.simulate(dt,Input_nogo*IS)
    
    Out_go = Out_go.cpu().numpy()
    Out_nogo = Out_nogo.cpu().numpy()
    Input_go = Input_go.cpu().numpy()
    Input_nogo = Input_nogo.cpu().numpy()
    g_go = g_go.cpu().numpy()
    g_nogo = g_nogo.cpu().numpy()
    V_go = V_go.cpu().numpy()
    V_nogo = V_nogo.cpu().numpy()
    spk_go = spk_go.cpu().numpy()
    spk_nogo = spk_nogo.cpu().numpy()

    Out_go = np.dot(np.tanh(g_go.T),W_out)/(LRSNN.N_E.cpu().numpy()+LRSNN.N_I.cpu().numpy())
    Out_nogo = np.dot(np.tanh(g_nogo.T),W_out)/(LRSNN.N_E.cpu().numpy()+LRSNN.N_I.cpu().numpy())

    prop = max(Out_go)/max(Out_nogo)
    print('Performance: ', prop[0])

    # Color data
    # color_Go = '#1C63A9'
    # color_Nogo = '#009999'
    # fig,ax = plt.subplots()
    # lowrankSNN.Draw_Output(ax,Out_go,'Output_{Go}',dt,Input_go*IS,color_data = color_Go)
    # lowrankSNN.Draw_Output(ax,Out_nogo,'Output_{Nogo}',dt,Input_nogo*IS,color_data=color_Nogo)
    # # Monitor the Average Conductance
    # fig, ax = plt.subplots()
    # lowrankSNN.Draw_Conductance(ax,g_go,color_Go,"Average Conductance_{Go}",dt,Input_go)
    # lowrankSNN.Draw_Conductance(ax,g_nogo,color_Nogo,"Average Conductance_{Nogo}",dt,Input_nogo)
    return prop[0]

mu_all = range(1,101)
RandomS_all = range(11)
sigma_all = range(1,71)

perf_mu = []
perf_RS = []
perf_sigma = []

# 测试mu
for mu in mu_all:
    perf = test(mu,0.5,10)
    perf_mu.append(perf)

with open(path+'prop_change_mu.csv','a+',newline='') as f:
    csv.writer(f).writerow(mu_all)
    csv.writer(f).writerow(perf_mu)
# 测试RS
for RandomS in RandomS_all:
    perf = test(50,RandomS,10)
    perf_RS.append(perf)
with open(path+'prop_change_RS.csv','a+',newline='') as f:
    csv.writer(f).writerow(RandomS_all)
    csv.writer(f).writerow(perf_RS)
# 测试sigma
for sigma in sigma_all:
    perf = test(50,0.5,sigma)
    perf_sigma.append(perf)
with open(path+'prop_change_sigma.csv','a+',newline='') as f:
    csv.writer(f).writerow(sigma_all)
    csv.writer(f).writerow(perf_sigma)

























