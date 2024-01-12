'''
对比只使用兴奋性神经元和既用exc又用inh的结果 看结果
可以通过循环创建多个SNN
看performance 如何变化
performance存在prop_origin_onlyexc.csv和prop_origin.csv中
'''
import lowrankSNN
import numpy as np
import csv
# 存储Propotion of go to nogo 到csv文件 
prop_origin=[]
prop_onlyexc = []

# 开始循环测试
for rep in range(40):
    # Initialiazation
    # LRSNN = lowrankSNN.LowRankSNN(N_E=4000,N_I=1000,RS= 1,IS=100,taud_E=2,taud_I=5)
    # 偶次循环只用exc，奇次循环用exc+inh
    if rep % 2 == 0:
        LRSNN = lowrankSNN.LowRankSNN(N_E=250,N_I=0,RS= 1,IS=100,taud_E=2,taud_I=5)
    else:
        LRSNN = lowrankSNN.LowRankSNN(N_E=200,N_I=50,RS= 1,IS=100,taud_E=2,taud_I=5)

    # Go_NoGo Task
    # Prepare the Stimuli and Readout Vector
    temp = np.random.rand(1,LRSNN.N_E+LRSNN.N_I) #Size (1,N_E) for Sti_go and nogo #把Low Rank加到整个网络上
    Sti_go = temp.copy()
    Sti_nogo = temp.copy()
    W_out = temp.copy()
    Sti_go[Sti_go>1/3] = 0
    Sti_nogo[Sti_nogo<1/3] = 0
    Sti_nogo[Sti_nogo>2/3] = 0
    W_out[W_out<2/3] = 0

    # Use Folded Gaussian Distribution to generate Stimuli and Readout Vector
    std_Sti = 2. #Standerd Deviration of Stimuli
    std_Wout = 2. #Standerd Deviration of readout matrix
    Sti_go[Sti_go!=0] = np.abs(np.random.normal(0,std_Sti,len(np.nonzero(Sti_go)[0]))) 
    Sti_nogo[Sti_nogo!=0] = np.abs(np.random.normal(0,std_Sti,len(np.nonzero(Sti_nogo)[0])))
    W_out[W_out!=0] = np.abs(np.random.normal(0,std_Wout,len(np.nonzero(W_out)[0])))
    W_out = np.transpose(W_out) #Size (N,1)

    conn_LR = W_out*Sti_go/(LRSNN.N_E+LRSNN.N_I) # 为什么除以神经元总数? # Low Rank Connectivity (Rank = 1)
    conn_rand = np.abs(np.random.normal(0,1/(LRSNN.N_E+LRSNN.N_I),(LRSNN.N_E+LRSNN.N_I,LRSNN.N_E+LRSNN.N_I))) # Random Connectivity

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

    # 以Out_go和Out_nogo的最大值的比例作为Performance
    prop = max(Out_go)/max(Out_nogo)
    if rep % 2 == 0:
        prop_onlyexc.append(prop[0])
    else: 
        prop_origin.append(prop[0])

    print('Performance_origin: ', prop)
    # print(type(prop))

with open('prop_origin_onlyexc.csv','a+',newline='') as f:
    csv.writer(f).writerow(prop_onlyexc)

with open('prop_origin.csv','a+',newline='') as f:
    csv.writer(f).writerow(prop_origin)














