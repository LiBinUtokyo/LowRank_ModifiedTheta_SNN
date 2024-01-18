'''
用于重现精密工学会发表Figure1的图
改变Gamma分布的mu,random strength和sigma看performance的变化
N_E = 5000

mu从1到100一步一测 即range(1,101) RS = 0.1, sigma = 10
random Strength从 0 到 10, 即range(11) mu = 50, sigma = 10
sigma从1到70 range(71) mu = 50, RS = 0.1

得到的结果存到名为prop_change_mu.csv,prop_change_RS.csv,prop_change_sigma.csv的文件中, 第一行是自变量, 第二行是performance
'''
import lowrankSNN
import numpy as np
import csv

def test(mu,RandomS,sigma):
    # Initialiazation
    LRSNN = lowrankSNN.LowRankSNN(N_E=5000,N_I=0,RS=RandomS,taud_E=2,taud_I=5)
    #low rank文献的N=5000
    # Go_NoGo Task
    # Prepare the Stimuli and Readout Vector
    temp = np.random.rand(1,LRSNN.N_E+LRSNN.N_I) #Size (1,N_E) for Sti_go and nogo #这里我想试试把Low Rank加到整个网络上
    Sti_go = temp.copy()
    Sti_nogo = temp.copy()
    W_out = temp.copy()
    Sti_go[Sti_go>1/3] = 0
    Sti_nogo[Sti_nogo<1/3] = 0
    Sti_nogo[Sti_nogo>2/3] = 0
    W_out[W_out<2/3] = 0

    # Use Gamma Distribution to generate Stimuli and Readout Vector
    # mean and std of Gamma Distribution(Deside Sti_go,Sti_nogo,W_out,conn_rand)
    # mu = 1
    si = sigma
    b = si**2/mu
    a = mu/b

    Sti_go[Sti_go!=0] = np.random.gamma(a,b,len(np.nonzero(Sti_go)[0])) #random.gamma(shape(a), scale(b)=1.0, size=None),这个地方的Gamma分布及其参数选取需要进一步讨论
    Sti_nogo[Sti_nogo!=0] = np.random.gamma(a,b,len(np.nonzero(Sti_nogo)[0]))
    W_out[W_out!=0] = np.random.gamma(a,b,len(np.nonzero(W_out)[0]))
    W_out = np.transpose(W_out) #Size (N_E,1)
    # Low Rank Connectivity (Rank = 1)
    conn_LR = W_out*Sti_go/(LRSNN.N_E+LRSNN.N_I) # 为什么除以神经元总数?
    # Random Connectivity
    conn_rand = np.random.gamma(a,b,(LRSNN.N_E+LRSNN.N_I,LRSNN.N_E+LRSNN.N_I)) #这里的Gamma分布取值也需要讨论

    # # Use Folded Gaussian Distribution to generate Stimuli and Readout Vector
    # std_Sti = 2. #Standerd Deviration of Stimuli
    # std_Wout = 2. #Standerd Deviration of readout matrix
    # Sti_go[Sti_go!=0] = np.abs(np.random.normal(0,std_Sti,len(np.nonzero(Sti_go)[0]))) 
    # Sti_nogo[Sti_nogo!=0] = np.abs(np.random.normal(0,std_Sti,len(np.nonzero(Sti_nogo)[0])))
    # W_out[W_out!=0] = np.abs(np.random.normal(0,std_Wout,len(np.nonzero(W_out)[0])))
    # W_out = np.transpose(W_out) #Size (N,1)

    # conn_LR = W_out*Sti_go/(LRSNN.N_E+LRSNN.N_I) # 为什么除以神经元总数? # Low Rank Connectivity (Rank = 1)
    # conn_rand = np.abs(np.random.normal(0,1/(LRSNN.N_E+LRSNN.N_I),(LRSNN.N_E+LRSNN.N_I,LRSNN.N_E+LRSNN.N_I))) # Random Connectivity

    m = W_out #m = Wout
    n = Sti_go #n = Stigo

    # Assemble the Network
    LRSNN.add_lowrank(conn_LR, W_out)
    LRSNN.add_random(conn_rand)
    # Show the Network information before simulaiton
    # LRSNN.show_conn()
    dt = 0.01 #(ms/step)
    T_pre = 5 # length of time before sti (ms)
    T_sti = 10 # length of time for sti (ms)
    T_after = 15 # length of time after sti (ms)
    T = T_pre+T_sti+T_after # length of Period time (ms): 30ms

    Input_go = np.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt))) #size:(N,time)
    Input_go[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = Sti_go.T
    Input_nogo = np.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt)))
    Input_nogo[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = Sti_nogo.T

    # Simulation
    Out_go, V_go, g_go, spk_go = LRSNN.simulate(dt,Input_go)
    Out_nogo, V_nogo, g_nogo, spk_nogo = LRSNN.simulate(dt,Input_nogo)

    prop = max(Out_go)/max(Out_nogo)
    print('Performance: ', prop[0])
    return prop[0]

mu_all = range(1,101)
RandomS_all = range(11)
sigma_all = range(1,71)

perf_mu = []
perf_RS = []
perf_sigma = []

# 测试mu
for mu in mu_all:
    perf = test(mu,0.1,10)
    perf_mu.append(perf)

with open('prop_change_mu.csv','a+',newline='') as f:
    csv.writer(f).writerow(mu_all)
    csv.writer(f).writerow(perf_mu)
# 测试RS
for RandomS in RandomS_all:
    perf = test(50,RandomS,10)
    perf_RS.append(perf)
with open('prop_change_RS.csv','a+',newline='') as f:
    csv.writer(f).writerow(RandomS_all)
    csv.writer(f).writerow(perf_RS)
# 测试sigma
for sigma in sigma_all:
    perf = test(50,0.1,sigma)
    perf_sigma.append(perf)
with open('prop_change_sigma.csv','a+',newline='') as f:
    csv.writer(f).writerow(sigma_all)
    csv.writer(f).writerow(perf_sigma)

























