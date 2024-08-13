'''
3/7 改变来自同一个神经元群体的连接，比如来自E的所有连接和来自I的所有连接
3/1 mu改为1，random取gamma分布，重新跑 上午9:24开始
2/29 增加了保存模型的代码 下午2:11开始跑
2/28 给连接添加权重，通过改变权重来改变连接强度，做参数探索

2/27: 改为使用Gamma 分布和使用GPU的代码重新测试
RS = 1 , IS = 1, mu = 1, sigma = 10
取消了输出的激活函数

改变神经元间的连接强度(-30%,0,30%)
I to I
I to E
E to I
E to E
通过循环创建多个SNN
看他们的performance 如何变化
并存到prop_change_EE_70,prop_change_EE_130,
        prop_change_EI_70,prop_change_EI_130,
        prop_change_IE_70,prop_change_IE_130,
        prop_change_II_70,prop_change_II_130, prop_origin.csv中
'''
import lowrankSNN_GPU as lowrankSNN
import numpy as np
import csv
import torch.distributions as dist
import torch
# from datetime import datetime
import functions


#定义一个函数，传入SNN，各个连接矩阵的附加权重，自动运行并返回任务表现
def test(SNN,sti_go,sti_nogo,dt,kee=1,kei=1,kie=1,kii=1):
    #改变连接强度后测试

    # change EE
    conn_ee = SNN.conn[:SNN.N_E,:SNN.N_E].clone()
    SNN.conn[:SNN.N_E,:SNN.N_E] = kee*conn_ee.clone()
    # change EI
    conn_ei = SNN.conn[:SNN.N_E,SNN.N_E:].clone()
    SNN.conn[:SNN.N_E,SNN.N_E:] = kei*conn_ei.clone()
    # change IE
    conn_ie = SNN.conn[SNN.N_E:,:SNN.N_E].clone()
    SNN.conn[SNN.N_E:,:SNN.N_E] = kie*conn_ie.clone()
    # change II
    conn_ii = SNN.conn[SNN.N_E:,SNN.N_E:].clone()
    SNN.conn[SNN.N_E:,SNN.N_E:] = kii*conn_ii.clone()

    Out_go, _, _, _ = LRSNN(dt,sti_go)
    Out_nogo, _, _, _ = LRSNN(dt,sti_nogo)
    prop = torch.max(Out_go)/torch.max(Out_nogo)

    # with open(file_path,'a+',newline='') as f:
    #     csv.writer(f).writerow([prop.item()])

    #测试后恢复连接强度

    # change EE
    SNN.conn[:SNN.N_E,:SNN.N_E] = conn_ee.clone()
    # change EI
    SNN.conn[:SNN.N_E,SNN.N_E:] = conn_ei.clone()
    # change IE
    SNN.conn[SNN.N_E:,:SNN.N_E] = conn_ie.clone()
    # change II
    SNN.conn[SNN.N_E:,SNN.N_E:] = conn_ii.clone()
    return prop.item()

for rep in range(50):
    # Initialiazation
    LRSNN = lowrankSNN.LowRankSNN(N_E=4000,N_I=1000,RS= 1,taud_E=2,taud_I=5)
    # LRSNN = lowrankSNN.LowRankSNN(N_E=200,N_I=50,RS= 1,taud_E=2,taud_I=5)
    # Go_NoGo Task
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
    # mu = 0.1
    # mu = 50
    mu = 1
    # mu = 2
    # 创建Gamma分布

    si = 10
    # si = np.sqrt(1/LRSNN.N_E+LRSNN.N_I)
    # si = 2

    b = mu/si**2
    a = mu*b
    gamma_dist = dist.gamma.Gamma(a, b)

    Sti_go[Sti_go!=0] = gamma_dist.sample((len(torch.nonzero(Sti_go)),)) #random.gamma(shape(a), scale(b)=1.0, size=None),这个地方的Gamma分布及其参数选取需要进一步讨论
    Sti_nogo[Sti_nogo!=0] = gamma_dist.sample((len(torch.nonzero(Sti_nogo)),))
    W_out[W_out!=0] = gamma_dist.sample((len(torch.nonzero(W_out)),))
    W_out = np.transpose(W_out) #Size (N_E,1)
    # Low Rank Connectivity (Rank = 1)
    conn_LR = W_out*Sti_go/(LRSNN.N_E+LRSNN.N_I) # 为什么除以神经元总数?
    # conn_LR[conn_LR>1] = 1
    # # Random Connectivity
    # # conn_rand = gamma_dist.sample(((LRSNN.N_E+LRSNN.N_I,LRSNN.N_E+LRSNN.N_I)))

    # conn_rand = np.abs(np.random.normal(0,np.sqrt(1/(LRSNN.N_E+LRSNN.N_I)) ,(LRSNN.N_E+LRSNN.N_I,LRSNN.N_E+LRSNN.N_I))) #改回和原来一样的形式
    # conn_rand = torch.from_numpy(conn_rand)
    # # conn_rand[conn_rand>1] = 1

    mu_rand = 1/(LRSNN.N_E+LRSNN.N_I)
    si_rand = 100/(LRSNN.N_E+LRSNN.N_I)
    b_rand = mu_rand/si_rand**2
    a_rand = mu_rand*b_rand
    gamma_dist_rand = dist.gamma.Gamma(a_rand, b_rand)
    # Random Connectivity
    conn_rand = gamma_dist_rand.sample(((LRSNN.N_E+LRSNN.N_I,LRSNN.N_E+LRSNN.N_I))) #这里的Gamma分布取值也需要讨论


    m = W_out #m = Wout
    n = Sti_go #n = Stigo

    # Assemble the Network
    LRSNN.add_lowrank(conn_LR, W_out)
    LRSNN.add_random(conn_rand)
    LRSNN.conn[LRSNN.conn>1] = 1
    # Show the Network information before simulaiton
    # LRSNN.show_conn()

    # print(LRSNN.conn_lowrank.shape, LRSNN.conn_random.shape)

    dt = 0.01 #(ms/step)
    T_pre = 5 # length of time before sti (ms)
    T_sti = 10 # length of time for sti (ms)
    T_after = 15 # length of time after sti (ms)
    T = T_pre+T_sti+T_after # length of Period time (ms): 30ms

    # IS = 3 #Input Strength
    IS = 1 #Input Strength

    Input_go = torch.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt))) #size:(N,time)
    Input_go[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = IS*Sti_go.T
    Input_nogo = torch.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt)))
    Input_nogo[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = IS*Sti_nogo.T

    #将模型及相应属性移动到GPU
    device = torch.device('cuda:0')
    LRSNN = LRSNN.to(device)
    Input_go = Input_go.to(device)
    Input_nogo = Input_nogo.to(device)
    #后面会除以10的 范围在0到2之间，每0.1测试一次
    kes = range(0,21)
    kis = range(0,21)
    path_ke = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/prop_change_ke.csv'
    path_ki = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/prop_change_ki.csv'

    profs=[]
    for ke in kes:
        profs.append(test(LRSNN,Input_go,Input_nogo,dt,kee=ke/10,kie=ke/10))
    with open(path_ke,'a+',newline='') as f:
        csv.writer(f).writerow(profs)

    profs=[]
    for ki in kis:
        profs.append(test(LRSNN,Input_go,Input_nogo,dt,kei=ki/10,kii=ki/10))
    with open(path_ki,'a+',newline='') as f:
        csv.writer(f).writerow(profs)

    #存储模型
    functions.save_model(LRSNN,dt,Sti_go,Sti_nogo,Input_go,Input_nogo,IS,m,n)

    # #后面会除以10的 范围在0到2之间，每0.1测试一次
    # kees = range(0,21)
    # keis = range(0,21)
    # kies = range(0,21)
    # kiis = range(0,21)

    # path_kee = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/prop_change_kee.csv'
    # path_kei = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/prop_change_kei.csv'
    # path_kie = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/prop_change_kie.csv'
    # path_kii = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/prop_change_kii.csv'
    # path_kii_kei = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/prop_change_kii_kei.csv'
    # path_kii_kie = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/prop_change_kii_kie.csv'
    # path_kei_kie = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/prop_change_kei_kie.csv'

    # profs=[]
    # for kee in kees:
    #     profs.append(test(LRSNN,Input_go,Input_nogo,dt,kee=kee/10))
    # with open(path_kee,'a+',newline='') as f:
    #     csv.writer(f).writerow(profs)

    # profs=[]
    # for kei in keis:
    #     profs.append(test(LRSNN,Input_go,Input_nogo,dt,kei=kei/10))
    # with open(path_kei,'a+',newline='') as f:
    #     csv.writer(f).writerow(profs)

    # profs=[]
    # for kie in kies:
    #     profs.append(test(LRSNN,Input_go,Input_nogo,dt,kie=kie/10))
    # with open(path_kie,'a+',newline='') as f:
    #     csv.writer(f).writerow(profs)

    # profs=[]
    # for kii in kiis:
    #     profs.append(test(LRSNN,Input_go,Input_nogo,dt,kii=kii/10))
    # with open(path_kii,'a+',newline='') as f:
    #     csv.writer(f).writerow(profs)

    # for kii in kiis:
    #     profs = []
    #     for kei in keis:
    #         profs.append(test(LRSNN,Input_go,Input_nogo,dt,kei=kei/10,kii=kii/10))
    #     with open(path_kii_kei,'a+',newline='') as f:
    #         csv.writer(f).writerow(profs)

    # for kii in kiis:
    #     profs = []
    #     for kie in kies:
    #         profs.append(test(LRSNN,Input_go,Input_nogo,dt,kie=kie/10,kii=kii/10))
    #     with open(path_kii_kie,'a+',newline='') as f:
    #         csv.writer(f).writerow(profs)

    # for kei in keis:
    #     profs = []
    #     for kie in kies:
    #         profs.append(test(LRSNN,Input_go,Input_nogo,dt,kei=kei/10,kie=kie/10))
    #     with open(path_kei_kie,'a+',newline='') as f:
    #         csv.writer(f).writerow(profs)

    # now = datetime.now()
    # formatted_now = now.strftime("%Y_%m_%d_%H_%M")
    # torch.save({
    #     'model':LRSNN,
    #     'Input_go':Input_go,
    #     'Input_nogo':Input_nogo,
    #     'dt':dt,
    #     'IS':IS,
    #     'Sti_go':Sti_go,
    #     'Sti_nogo':Sti_nogo
    # },f'/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/models/{formatted_now}.pth')






# # 存储Propotion of go to nogo (II,EI,IE,EE)到csv文件 
# prop_change_EE_70=[]
# prop_change_EE_130=[]

# prop_change_EI_70=[]
# prop_change_EI_130=[]

# prop_change_IE_70=[]
# prop_change_IE_130=[]

# prop_change_II_70=[]
# prop_change_II_130=[]

# prop_origin=[]


# # 开始循环测试
# for rep in range(40):
#     # Initialiazation
#     LRSNN = lowrankSNN.LowRankSNN(N_E=4000,N_I=1000,RS= 1,taud_E=2,taud_I=5)
#     # LRSNN = lowrankSNN.LowRankSNN(N_E=200,N_I=50,RS= 1,taud_E=2,taud_I=5)
#     # Go_NoGo Task
#     # Prepare the Stimuli and Readout Vector
#     temp = torch.rand(1,LRSNN.N_E+LRSNN.N_I) #Size (1,N_E) for Sti_go and nogo #这里我想试试把Low Rank加到整个网络上
#     Sti_go = temp.clone()
#     Sti_nogo = temp.clone()
#     W_out = temp.clone()

#     Sti_go[Sti_go>1/3] = 0
#     Sti_nogo[Sti_nogo<1/3] = 0
#     Sti_nogo[Sti_nogo>2/3] = 0
#     W_out[W_out<2/3] = 0

#     # Use Gamma Distribution to generate Stimuli and Readout Vector
#     # mean and std of Gamma Distribution(Deside Sti_go,Sti_nogo,W_out,conn_rand)
#     # mu = 0.1
#     # mu = 50
#     mu = 1
#     # mu = 2
#     # 创建Gamma分布

#     si = 10
#     # si = np.sqrt(1/LRSNN.N_E+LRSNN.N_I)
#     # si = 2

#     b = mu/si**2
#     a = mu*b
#     gamma_dist = dist.gamma.Gamma(a, b)

#     Sti_go[Sti_go!=0] = gamma_dist.sample((len(torch.nonzero(Sti_go)),)) #random.gamma(shape(a), scale(b)=1.0, size=None),这个地方的Gamma分布及其参数选取需要进一步讨论
#     Sti_nogo[Sti_nogo!=0] = gamma_dist.sample((len(torch.nonzero(Sti_nogo)),))
#     W_out[W_out!=0] = gamma_dist.sample((len(torch.nonzero(W_out)),))
#     W_out = np.transpose(W_out) #Size (N_E,1)
#     # Low Rank Connectivity (Rank = 1)
#     conn_LR = W_out*Sti_go/(LRSNN.N_E+LRSNN.N_I) # 为什么除以神经元总数?
#     # conn_LR[conn_LR>1] = 1
#     # Random Connectivity
#     # conn_rand = gamma_dist.sample(((LRSNN.N_E+LRSNN.N_I,LRSNN.N_E+LRSNN.N_I)))

#     conn_rand = np.abs(np.random.normal(0,np.sqrt(1/(LRSNN.N_E+LRSNN.N_I)) ,(LRSNN.N_E+LRSNN.N_I,LRSNN.N_E+LRSNN.N_I))) #改回和原来一样的形式
#     conn_rand = torch.from_numpy(conn_rand)
#     # conn_rand[conn_rand>1] = 1

#     m = W_out #m = Wout
#     n = Sti_go #n = Stigo

#     # Assemble the Network
#     LRSNN.add_lowrank(conn_LR, W_out)
#     LRSNN.add_random(conn_rand)
#     LRSNN.conn[LRSNN.conn>1] = 1
#     # Show the Network information before simulaiton
#     # LRSNN.show_conn()

#     dt = 0.01 #(ms/step)
#     T_pre = 5 # length of time before sti (ms)
#     T_sti = 10 # length of time for sti (ms)
#     T_after = 15 # length of time after sti (ms)
#     T = T_pre+T_sti+T_after # length of Period time (ms): 30ms

#     # IS = 3 #Input Strength
#     IS = 1 #Input Strength

#     Input_go = torch.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt))) #size:(N,time)
#     Input_go[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = IS*Sti_go.T
#     Input_nogo = torch.zeros((LRSNN.N_E+LRSNN.N_I,int(T/dt)))
#     Input_nogo[:,int(T_pre/dt):int((T_pre+T_sti)/dt)] = IS*Sti_nogo.T

#     #将模型及相应属性移动到GPU
#     device = torch.device('cuda:0')
#     LRSNN = LRSNN.to(device)
#     Input_go = Input_go.to(device)
#     Input_nogo = Input_nogo.to(device)

#     # Start Simulation
#     Out_go, V_go, g_go, spk_go = LRSNN(dt,Input_go)
#     Out_nogo, V_nogo, g_nogo, spk_nogo = LRSNN(dt,Input_nogo)

#     # 以Out_go和Out_nogo的最大值的比例作为Performance
#     prop = torch.max(Out_go)/torch.max(Out_nogo)
#     print('Performance(Origin): ', prop.item())
#     # print(type(prop))
#     prop_origin.append(prop.item())

#     # 改变连接强度后测试

#     conn_EE = LRSNN.conn[:LRSNN.N_E,:LRSNN.N_E].clone()
#     # change EE
#     LRSNN.conn[:LRSNN.N_E,:LRSNN.N_E] = conn_EE.clone()*0.7
#     Out_go, V_go, g_go, spk_go = LRSNN(dt,Input_go)
#     Out_nogo, V_nogo, g_nogo, spk_nogo = LRSNN(dt,Input_nogo)
#     prop = torch.max(Out_go)/torch.max(Out_nogo)
#     prop_change_EE_70.append(prop.item())
#     print('Performance_70%_EtoE: ', prop.item())

#     LRSNN.conn[:LRSNN.N_E,:LRSNN.N_E] = conn_EE.clone()*1.3
#     Out_go, V_go, g_go, spk_go = LRSNN(dt,Input_go)
#     Out_nogo, V_nogo, g_nogo, spk_nogo = LRSNN(dt,Input_nogo)
#     prop = torch.max(Out_go)/torch.max(Out_nogo)
#     prop_change_EE_130.append(prop.item())
#     print('Performance_130%_EtoE: ', prop.item())
#     LRSNN.conn[:LRSNN.N_E,:LRSNN.N_E] = conn_EE.clone()

#     # change IE
#     conn_IE = LRSNN.conn[LRSNN.N_E:,:LRSNN.N_E].clone()
#     LRSNN.conn[LRSNN.N_E:,:LRSNN.N_E] = conn_IE.clone()*0.7
#     Out_go, V_go, g_go, spk_go = LRSNN(dt,Input_go)
#     Out_nogo, V_nogo, g_nogo, spk_nogo = LRSNN(dt,Input_nogo)
#     prop = torch.max(Out_go)/torch.max(Out_nogo)
#     prop_change_IE_70.append(prop.item())
#     print('Performance_70%_EtoI: ', prop.item())

#     LRSNN.conn[LRSNN.N_E:,:LRSNN.N_E] = conn_IE.clone()*1.3
#     Out_go, V_go, g_go, spk_go = LRSNN(dt,Input_go)
#     Out_nogo, V_nogo, g_nogo, spk_nogo = LRSNN(dt,Input_nogo)
#     prop = torch.max(Out_go)/torch.max(Out_nogo)
#     prop_change_IE_130.append(prop.item())
#     print('Performance_130%_EtoI: ', prop.item())
#     LRSNN.conn[LRSNN.N_E:,:LRSNN.N_E] = conn_IE.clone()

#     # Change EI
#     conn_EI = LRSNN.conn[:LRSNN.N_E,LRSNN.N_E:].clone()
#     LRSNN.conn[:LRSNN.N_E,LRSNN.N_E:] = conn_EI.clone()*0.7
#     Out_go, V_go, g_go, spk_go = LRSNN(dt,Input_go)
#     Out_nogo, V_nogo, g_nogo, spk_nogo = LRSNN(dt,Input_nogo)
#     prop = torch.max(Out_go)/torch.max(Out_nogo)
#     prop_change_EI_70.append(prop.item())
#     print('Performance_70%_ItoE: ', prop.item())

#     LRSNN.conn[:LRSNN.N_E,LRSNN.N_E:] = conn_EI.clone()*1.3
#     Out_go, V_go, g_go, spk_go = LRSNN(dt,Input_go)
#     Out_nogo, V_nogo, g_nogo, spk_nogo = LRSNN(dt,Input_nogo)
#     prop = torch.max(Out_go)/torch.max(Out_nogo)
#     prop_change_EI_130.append(prop.item())
#     print('Performance_130%_ItoE: ', prop.item())
#     LRSNN.conn[:LRSNN.N_E,LRSNN.N_E:] = conn_EI.clone()

#     # Change II
#     conn_II = LRSNN.conn[LRSNN.N_E:,LRSNN.N_E:].clone()
#     LRSNN.conn[LRSNN.N_E:,LRSNN.N_E:] = conn_II.clone()*0.7
#     Out_go, V_go, g_go, spk_go = LRSNN(dt,Input_go)
#     Out_nogo, V_nogo, g_nogo, spk_nogo = LRSNN(dt,Input_nogo)
#     prop = torch.max(Out_go)/torch.max(Out_nogo)
#     prop_change_II_70.append(prop.item())
#     print('Performance_70%_ItoI: ', prop.item())

#     LRSNN.conn[LRSNN.N_E:,LRSNN.N_E:] = conn_II.clone()*1.3
#     Out_go, V_go, g_go, spk_go = LRSNN(dt,Input_go)
#     Out_nogo, V_nogo, g_nogo, spk_nogo = LRSNN(dt,Input_nogo)
#     prop = torch.max(Out_go)/torch.max(Out_nogo)
#     prop_change_II_130.append(prop.item())
#     print('Performance_130%_ItoI: ', prop.item())
#     LRSNN.conn[LRSNN.N_E:,LRSNN.N_E:] = conn_II.clone()

# k = 0
# data = [prop_change_EE_70,prop_change_EE_130,
#         prop_change_EI_70,prop_change_EI_130,
#         prop_change_IE_70,prop_change_IE_130,
#         prop_change_II_70,prop_change_II_130]
# for i in ['EE','EI','IE','II']:
#     for j in ['70','130']:
#         # 存到csv文件中
#         with open(f'prop_change_{i}_{j}.csv','a+',newline='') as f:
#             csv.writer(f).writerow(data[k])
#             k+=1

# with open('prop_origin.csv','a+',newline='') as f:
#     csv.writer(f).writerow(prop_origin)














