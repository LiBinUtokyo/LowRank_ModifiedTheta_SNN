'''
用于构建基于Pytorch的lowrankSNN

'''

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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
        

    def show(self):
        print('Network Settings')
        print('==========================================')
        print('Number of Neurons: ', self.N_E+self.N_I)
        print('Number of Excitatory Units: ', self.N_E)
        print('Number of Inhibitory Units: ', self.N_I)
        # full_w = torch.mm(self.W, self.mask) #包含兴奋和抑制性信息的连接矩阵
        # full_w = self.W * self.mask
        zero_w = (self.conn == 0).sum().item()
        # pos_w = (self.conn > 0).sum().item()
        # neg_w = (full_w < 0 ).sum().item()
        print('Zero Weights occupy: %2.2f %%'%(zero_w/((self.N_E+self.N_I)**2)*100))
        # print('Positive Weights occupy: %2.2f %%'%(pos_w/(self.N**2)*100))
        # print('Negative Weights occupy: %2.2f %%'%(neg_w/(self.N**2)*100))

    def show_conn(self, maxvalue = 0.001):
        full_w = self.conn.cpu().clone().detach().numpy() #包含兴奋和抑制性信息的连接矩阵
        # let the weight from Inhibitory be negative value
        full_w[:,self.N_E:self.N_E+self.N_I] = -full_w[:,self.N_E:self.N_E+self.N_I]
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # 蓝 -> 白 -> 红
        cmap_name = 'gradient_div_cmap'
        gradient_cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=100)  # N=100 使渐变更加平滑
        plt.imshow(full_w,cmap=gradient_cm,vmax = maxvalue,vmin = -maxvalue)
        plt.colorbar()
        plt.title('Connectivity Matrix')
        plt.xlabel('From')
        plt.xticks(np.arange(0,len(full_w)+1,500))
        plt.ylabel('To')
        plt.yticks(np.arange(0,len(full_w)+1,500))
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')
        plt.show()
        

    def add_random(self,conn_rand):
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
        # Input size:(N，time)

        if self.conn_lowrank.shape == self.conn_random.shape:
            print('Low-rank connectivity is added to all the connections')
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
        g = torch.zeros_like(Input).to(Input.device)
        # g_EE = np.delete(np.zeros_like(Input),range(self.N_I),axis=0)
        # g_IE = np.delete(np.zeros_like(Input),range(self.N_E),axis=0)
        # g_EI = np.delete(np.zeros_like(Input),range(self.N_I),axis=0)
        # g_II = np.delete(np.zeros_like(Input),range(self.N_E),axis=0)

        g_EE = g[:self.N_E,:].clone().to(Input.device)
        g_IE = g[:self.N_I,:].clone().to(Input.device)
        g_EI = g[:self.N_E,:].clone().to(Input.device)
        g_II = g[:self.N_I,:].clone().to(Input.device)


        spk = torch.zeros_like(Input).to(Input.device)

        for step, inputs in enumerate(Input.T): #for every time step
            if step == 0: continue

            # Calculate Synaptic Conductance (Single Exponential filter)
            # print(g_EE.device,self.taud_E.device,G_P[0].device,conn_EE.device,spk.device,dt.device)
            g_EE[:,step] = g_EE[:,step-1] + \
                (-g_EE[:,step-1]/self.taud_E+ \
                G_P[0]*conn_EE@spk[:self.N_E,step-1])*dt
            g_EI[:,step] = g_EI[:,step-1] + \
                (-g_EI[:,step-1]/self.taud_E+ \
                G_P[1]*conn_EI@spk[self.N_E:self.N_E+self.N_I,step-1])*dt
            g_IE[:,step] = g_IE[:,step-1] + \
                (-g_IE[:,step-1]/self.taud_I+ \
                G_P[2]*conn_IE@spk[:self.N_E,step-1])*dt
            g_II[:,step] = g_II[:,step-1] + \
                (-g_II[:,step-1]/self.taud_I+ \
                G_P[3]*conn_II@spk[self.N_E:self.N_E+self.N_I,step-1])*dt
            # Calculate the comprehensive Synaptic Conductance (Single Exponential filter)
                # For Excitatory Neurons
            g[:self.N_E,step] = g[0:self.N_E,step-1] + \
                (-g[0:self.N_E,step-1]/self.taud_E+ \
                G_P[0]*conn_EE@spk[:self.N_E,step-1]+ \
                G_P[1]*conn_EI@spk[self.N_E:self.N_E+self.N_I,step-1])*dt
                # For Inhibitory Neurons
            g[self.N_E:self.N_E+self.N_I,step] = g[self.N_E:self.N_E+self.N_I,step-1] + \
                (-g[self.N_E:self.N_E+self.N_I,step-1]/self.taud_I+ \
                G_P[2]*conn_IE@spk[:self.N_E,step-1]+ \
                G_P[3]*conn_II@spk[self.N_E:self.N_E+self.N_I,step-1])*dt

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
            phase[:,step][phase[:,step] >= torch.pi] -= 2*np.pi
            

        if self.conn_lowrank.shape == self.conn_random.shape:
            # print(g.T.dtype,self.W_out.dtype,self.N_E.dtype,self.N_I.dtype)
            # print(type(torch.mm(g.T,self.W_out)))
            # print(type(torch.mm(g.T,self.W_out)),type((self.N_E+self.N_I)))
            Out = torch.mm(g.T,self.W_out)/(self.N_E+self.N_I) #Size of g:(N,time), Size of W_out: (N,1)
        V = self.theta2V(phase)
        return Out, V, g, spk


