
'''
构建基于Pytorch的lowrankSNN
11/14: 在g的模拟中给来自其他神经元的影响除以了dt，从而让不同的dt下发放的影响一致
11/08: 增加指定初始值的功能,做了一些功能的优化
10/30: introduced synaptic current
10/23: 把g的位置的输出改为了[g, g_EE,g_IE,g_EI,g_II]
'''

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams.update({'font.size': 30})  #设置所有字体大小

class LowRankSNN(nn.Module):
    # CONSTANTS
    G_L_E = torch.tensor(0.08)
    G_L_I = torch.tensor(0.1)
    G_P= torch.tensor([0.004069, 0.02672, 0.003276, 0.02138]) #g_peak:[E←E, E←I, I←E, I←I]
    V_T = torch.tensor(-55)
    V_R = torch.tensor(-62)
    REV_E = torch.tensor(0)
    REV_I = torch.tensor(-70)
    BIAS = V_T - V_R #Bias current
    C0 = 2/(V_T-V_R)
    C1 = (2*REV_E-V_R-V_T)/(V_T-V_R)
    C2 = (2*REV_I-V_R-V_T)/(V_T-V_R)


    def __init__(self,N_E=1000,N_I=200,RS= 1,taud_E=2,taud_I=5) -> None:
        super().__init__() #调用了父类的方法
        self.N_E = torch.tensor(N_E)
        self.N_I = torch.tensor(N_I)
        self.RS = torch.tensor(RS)
        self.taud_E = torch.tensor(taud_E)
        self.taud_I = torch.tensor(taud_I)
        # self.conn = np.zeros((N_E+N_I,N_E+N_I))
        self.conn = torch.zeros(N_E+N_I,N_E+N_I)
        self.added_lowrank = False
        self.added_random = False
        self.loaded_init = False

    def load_init(self,g,g_EE,g_EI,g_IE,g_II,V,phase,I_syn,I_syn_EE,I_syn_EI,I_syn_IE,I_syn_II,spk):
        # size of all the inputs should be N
        self.g_init = g
        self.g_EE_init = g_EE
        self.g_EI_init = g_EI
        self.g_IE_init = g_IE
        self.g_II_init = g_II
        self.V_init = V
        self.phase_init = phase
        self.I_syn_init = I_syn
        self.I_syn_EE_init = I_syn_EE
        self.I_syn_EI_init = I_syn_EI
        self.I_syn_IE_init = I_syn_IE
        self.I_syn_II_init = I_syn_II
        self.spk_init = spk
        self.loaded_init = True
        print('Initial values have been loaded.')
        

    def show(self):
        print('Network Settings')
        print('==========================================')
        print('Number of Neurons: ', self.N_E+self.N_I)
        print('Number of Excitatory Units: ', self.N_E)
        print('Number of Inhibitory Units: ', self.N_I)
        print('Random Strength: ', self.RS)
        print('Excitatory Synaptic Time Constant: ', self.taud_E)
        print('Inhibitory Synaptic Time Constant: ', self.taud_I)
        print('==========================================')

    def show_conn(self):
        W_conn = self.conn.cpu().clone().detach()
        W_rank1 = self.conn_lowrank.cpu().clone().detach()
        W_random = self.conn_random.cpu().clone().detach()

        #draw the rank-1 matrix
        plt.figure()
        plt.imshow(W_rank1,interpolation='nearest')
        plt.colorbar()
        plt.title('Rank-1 matrix')
        plt.show()
        # 展示各部分的平均值
        print("Rank-1 matrix average value_EtoE:", torch.mean(W_rank1[:self.N_E, :self.N_E]))
        print("Rank-1 matrix average value_EtoI:", torch.mean(W_rank1[:self.N_E, self.N_E:]))
        print("Rank-1 matrix average value_ItoE:", torch.mean(W_rank1[self.N_E:, :self.N_E]))
        print("Rank-1 matrix average value_ItoI:", torch.mean(W_rank1[self.N_E:, self.N_E:]))
        #draw the random matrix
        plt.figure()
        plt.imshow(W_random,interpolation='nearest')
        plt.colorbar()
        plt.title('Full Rank matrix')
        plt.show()
        # 展示各部分的平均值
        print("Full Rank matrix average value_EtoE:", torch.mean(W_random[:self.N_E, :self.N_E]))
        print("Full Rank matrix average value_EtoI:", torch.mean(W_random[:self.N_E, self.N_E:]))
        print("Full Rank matrix average value_ItoE:", torch.mean(W_random[self.N_E:, :self.N_E]))
        print("Full Rank matrix average value_ItoI:", torch.mean(W_random[self.N_E:, self.N_E:]))
        #draw the full connectivity
        plt.figure()
        plt.imshow(W_conn,interpolation='nearest')
        plt.colorbar()
        plt.title('Connectivity matrix')
        plt.show()
        # 展示各部分的平均值
        print("Connectivity matrix average value_EtoE:", torch.mean(W_conn[:self.N_E, :self.N_E]))
        print("Connectivity matrix average value_EtoI:", torch.mean(W_conn[:self.N_E, self.N_E:]))
        print("Connectivity matrix average value_ItoE:", torch.mean(W_conn[self.N_E:, :self.N_E]))
        print("Connectivity matrix average value_ItoI:", torch.mean(W_conn[self.N_E:, self.N_E:]))
        return W_conn, W_rank1, W_random

    def add_random(self,conn_rand):
        if self.added_random:
            print('Random connection has been added.')
            return
        self.conn_random = conn_rand
        self.conn += conn_rand*self.RS
        self.added_random = True
        print('Random connection has been added.')
        return

    def remove_random(self):
        if not self.added_random:
            print('There is no random connection added.')
            return
        self.conn -= self.conn_random
        del self.conn_random
        self.added_random = False
        print('Random connection has been removed.')
        return

    def add_lowrank(self,conn_LR, W_out):
        if self.added_lowrank:
            print('Low Rank Connectivity has been added.')
            return
        self.conn_lowrank = conn_LR
        self.conn += conn_LR
        self.W_out = W_out
        self.added_lowrank = True
        print('Low Rank connection and readout vector have been added.')
        return

    def remove_lowrank(self):
        if not self.added_lowrank:
            print('There is no Low Rank Connectivity added.')
            return
        self.conn -= self.conn_lowrank
        del self.conn_lowrank
        del self.W_out
        self.added_lowrank = False
        print('Low Rank connection and readout vector have been removed.')

    def V2theta(self,V):
        V_R = LowRankSNN.V_R
        V_T = LowRankSNN.V_T
        return 2*torch.arctan((V-(V_R+V_T)/2)*2/(V_T-V_R))

    def theta2V(self,theta):
        V_R = LowRankSNN.V_R
        V_T = LowRankSNN.V_T
        return (V_T+V_R)/2+(V_T-V_R)/2*torch.tan(theta/2)
    
    def to(self, device):
        super().to(device)
        #将有必要的属性转为tensor并移动到指定的device
        self.G_P = self.G_P.to(device)
        self.G_L_E = self.G_L_E.to(device)
        self.G_L_I = self.G_L_I.to(device)
        self.C0 = self.C0.to(device)
        self.C1 = self.C1.to(device)
        self.C2 = self.C2.to(device)
        self.conn = self.conn.to(device)
        self.N_E = self.N_E.to(device)
        self.N_I = self.N_I.to(device)
        self.taud_E = self.taud_E.to(device)
        self.taud_I = self.taud_I.to(device)
        self.W_out = self.W_out.to(device)

        return self
    
    def forward(self,dt,Input):
        print('Start Simulation')
        # Input size:(N，time)

        # if self.conn_lowrank.shape == self.conn_random.shape:
        #     print('Low-rank connectivity is added to all the connections')
        dt = torch.tensor(dt).to(Input.device)
        G_P = self.G_P
        G_L_E = self.G_L_E
        G_L_I = self.G_L_I
        C0 = self.C0
        C1 = self.C1
        C2 = self.C2
        conn_EE = self.conn[:self.N_E,:self.N_E]
        conn_IE = self.conn[self.N_E:self.N_E+self.N_I,:self.N_E]
        conn_EI = self.conn[:self.N_E,self.N_E:self.N_E+self.N_I]
        conn_II = self.conn[self.N_E:self.N_E+self.N_I,self.N_E:self.N_E+self.N_I]
        V = torch.zeros_like(Input).to(Input.device)
        phase = torch.zeros_like(Input).to(Input.device)

        # Synaptic Conductance
        g = torch.zeros_like(Input).to(Input.device)
        g_EE = g[:self.N_E,:].clone().to(Input.device)
        g_IE = g[:self.N_I,:].clone().to(Input.device)
        g_EI = g[:self.N_E,:].clone().to(Input.device)
        g_II = g[:self.N_I,:].clone().to(Input.device)

        # Synaptic Current
        I_syn = torch.zeros_like(Input).to(Input.device)
        I_syn_EE = I_syn[:self.N_E,:].clone().to(Input.device)
        I_syn_IE = I_syn[:self.N_I,:].clone().to(Input.device)
        I_syn_EI = I_syn[:self.N_E,:].clone().to(Input.device)
        I_syn_II = I_syn[:self.N_I,:].clone().to(Input.device)

        # Spike record
        spk = torch.zeros_like(Input).to(Input.device)

        spk_step = []
        spk_ind = []

        if self.loaded_init:
            g[:,0] = self.g_init
            g_EE[:,0] = self.g_EE_init
            g_IE[:,0] = self.g_IE_init
            g_EI[:,0] = self.g_EI_init
            g_II[:,0] = self.g_II_init
            V[:,0] = self.V_init
            phase[:,0] = self.phase_init
            I_syn[:,0] = self.I_syn_init
            I_syn_EE[:,0] = self.I_syn_EE_init
            I_syn_IE[:,0] = self.I_syn_IE_init
            I_syn_EI[:,0] = self.I_syn_EI_init
            I_syn_II[:,0] = self.I_syn_II_init
            spk[:,0] = self.spk_init
            print('Using loaded initial values')

        for step, inputs in enumerate(Input.T): #for every time step
            if step == 0: continue

            # Calculate Synaptic Conductance (Single Exponential filter)
            # print(g_EE.device,self.taud_E.device,G_P[0].device,conn_EE.device,spk.device,dt.device)
            g_EE[:,step] = g_EE[:,step-1] + \
                (-g_EE[:,step-1]/self.taud_E+ \
                G_P[0]*conn_EE@spk[:self.N_E,step-1]/dt)*dt
            g_EI[:,step] = g_EI[:,step-1] + \
                (-g_EI[:,step-1]/self.taud_I+ \
                G_P[1]*conn_EI@spk[self.N_E:self.N_E+self.N_I,step-1]/dt)*dt
            g_IE[:,step] = g_IE[:,step-1] + \
                (-g_IE[:,step-1]/self.taud_E+ \
                G_P[2]*conn_IE@spk[:self.N_E,step-1]/dt)*dt
            g_II[:,step] = g_II[:,step-1] + \
                (-g_II[:,step-1]/self.taud_I+ \
                G_P[3]*conn_II@spk[self.N_E:self.N_E+self.N_I,step-1]/dt)*dt
            # # *(maybe wrong in decay time) Calculate the comprehensive Synaptic Conductance (Single Exponential filter)
                # For Excitatory Neurons
            g[:self.N_E,step] = g[0:self.N_E,step-1] + \
                (-g[0:self.N_E,step-1]/self.taud_E+ \
                G_P[0]*conn_EE@spk[:self.N_E,step-1]+ \
                G_P[1]*conn_EI@spk[self.N_E:self.N_E+self.N_I,step-1]/dt)*dt
                # For Inhibitory Neurons
            g[self.N_E:self.N_E+self.N_I,step] = g[self.N_E:self.N_E+self.N_I,step-1] + \
                (-g[self.N_E:self.N_E+self.N_I,step-1]/self.taud_I+ \
                G_P[2]*conn_IE@spk[:self.N_E,step-1]+ \
                G_P[3]*conn_II@spk[self.N_E:self.N_E+self.N_I,step-1]/dt)*dt
            
            # Calculate the Synaptic Current
            V[:,step-1] = self.theta2V(phase[:,step-1])
            I_syn_EE[:,step] = -g_EE[:,step]*(V[:self.N_E,step-1]-LowRankSNN.REV_E)
            I_syn_EI[:,step] = -g_EI[:,step]*(V[:self.N_E,step-1]-LowRankSNN.REV_I)
            I_syn_IE[:,step] = -g_IE[:,step]*(V[self.N_E:self.N_E+self.N_I,step-1]-LowRankSNN.REV_E)
            I_syn_II[:,step] = -g_II[:,step]*(V[self.N_E:self.N_E+self.N_I,step-1]-LowRankSNN.REV_I)
            I_syn[:self.N_E,step] = I_syn_EE[:,step]+I_syn_EI[:,step]
            I_syn[self.N_E:self.N_E+self.N_I,step] = I_syn_IE[:,step]+I_syn_II[:,step]
            if torch.any(g<0):
                print('got it')

            # Calculate Membrane Voltage (Phase)
            # For Excitatory Neurons
            phase_pre_E = phase[:self.N_E,step-1]
            phase[:self.N_E,step] = phase_pre_E + (-G_L_E*torch.cos(phase_pre_E)+C0*(1+torch.cos(phase_pre_E))*inputs[:self.N_E]+g_EE[:,step]*(C1*(1+torch.cos(phase_pre_E))-\
                torch.sin(phase_pre_E))+g_EI[:,step]*(C2*(1+torch.cos(phase_pre_E))-torch.sin(phase_pre_E)))*dt
            # For Inhibitory Neurons
            phase_pre_I = phase[self.N_E:self.N_E+self.N_I,step-1]
            phase[self.N_E:self.N_E+self.N_I,step] = phase_pre_I + (-G_L_I*torch.cos(phase_pre_I)+C0*(1+torch.cos(phase_pre_I))*inputs[self.N_E:self.N_E+self.N_I]+g_IE[:,step]*(C1*(1+torch.cos(phase_pre_I))-\
                torch.sin(phase_pre_I))+g_II[:,step]*(C2*(1+torch.cos(phase_pre_I))-torch.sin(phase_pre_I)))*dt

            # Store the firing time
            spk[:,step] = (phase[:,step] >= torch.pi).int()

            # Store the indeces of firing neurons
            ind_list = torch.nonzero(spk[:,step]).flatten().tolist()
            spk_ind.extend(ind_list)
            # Store the firing time
            spk_step.extend([step for _ in range(len(ind_list))])

            phase[:,step][phase[:,step] >= torch.pi] -= 2*np.pi
            
        # print(g.T.dtype,self.W_out.dtype,self.N_E.dtype,self.N_I.dtype)
        # print(type(torch.mm(g.T,self.W_out)))
        # print(type(torch.mm(g.T,self.W_out)),type((self.N_E+self.N_I)))
        # Out = torch.mm(g.T,self.W_out)/(self.N_E+self.N_I) #Size of g:(N,time), Size of W_out: (N,1)
        # Out = torch.mm(torch.tanh(I_syn_EE).T,self.W_out[:self.N_E])/(self.N_E+self.N_I)
        Out = torch.tanh(I_syn_EE).T@self.W_out[:self.N_E] # size(T,1)

        V[:,-1] = self.theta2V(phase[:,-1])
        print('Simulation Finished')

        return Out, V, [g,g_EE,g_EI,g_IE,g_II],[I_syn,I_syn_EE,I_syn_EI,I_syn_IE,I_syn_II], spk_step, spk_ind, spk, phase

