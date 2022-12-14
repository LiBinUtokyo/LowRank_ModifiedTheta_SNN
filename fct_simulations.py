import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import pickle
from functools import partial
import math

### Functions for building CONNECTIVITY MATRIX

def GetBulk (N):
    chi = np.random.normal( 0, np.sqrt(1./(N)), (N,N) )
    return chi

def GetGaussianVector (mean, std, N):

	if std>0:
		return np.random.normal (mean, std, N )
	else:
		return mean*np.ones(N)

### Functions for SIMULATION of modified theta model

def Integrate (X, t, J, Sti):
    dXdT = -X + np.dot( J, np.tanh(X) ) + Sti
    return dXdT

def V2Theta(V,V_T=-55,V_R=-62): #theta范围为-pi到pi
    return 2*np.array(list(map(math.atan, (V-(V_R+V_T)/2)*2/(V_T-V_R))))
    
    
def Theta2V(theta,V_T=-55,V_R=-62):
    ttheta = np.array(list(map(math.tan, theta/2)))
    return (V_R+V_T)/2+(V_T-V_R)/2*ttheta

def SimulateActivity (t, g0, V0, J, Sti):#t为持续时间的array，g0, V0为初始条件，J为连接矩阵，Sti为外界输入,返回g的后续变化和V的后续变化（and theta）还有点火记录
    print(' ** Simulating... **')
    #根据输入，初始化必须的变量
    N = len(g0)
    dt = t[1]-t[0] #单位ms
    T = t[-1] #单位ms
    V_T = -55 #点火阈值mV
    V_R = -62 #静息电位mV
    V_RE = -70#反转电位mV
    c1 = 2/(V_T-V_R) #和点火阈值和静息电位有关的常数
    c2 = (2*V_RE-V_R-V_T)/(V_T-V_R) #和反转电位有关的常数
    theta0 = V2Theta(V0)#初始相位
    gL = 0.1 #漏电导 (mS/cm^2)
    C = 1  #膜电容（uF/cm^2）
    td = 5 #synaptic decay time衰减常数（ms）
    tr = 2 #synaptic rising time(ms) here assumed to be zero 
#    gp = 0.0214 #peak conductance for GABA on interneuron(mS/cm^2)
    gp = 0.138 #peak conductance for GABA on interneuron(mS/cm^2)
    g = np.zeros([N,len(t)])
    g[:, 0] = g0
    
    theta = np.zeros([N,len(t)])
    theta[:, 0] = theta0
    
    v = np.zeros([N,len(t)])
    v[:, 0] = V0
    
    A = np.zeros([N,len(t)])
    
    #根据初始条件，开始模拟ModifiedThetaModel
    for i, ti in enumerate(t):
        if i == 0: continue
        #更新synaptic conductance g
        g[:, i] = g[:, i-1] + (-g[:, i-1]/td + gp/td*np.dot(J,A[:,i-1]))*dt
        #更新膜电位theta
        theta[:, i] = theta[:, i-1] + (-gL*np.cos(theta[:, i-1]) \
                                       +c1*(1+np.cos(theta[:, i-1]))*Sti \
                                       +g[:,i]*(c2*(1+np.cos(theta[:, i-1]))-np.sin(theta[:, i-1])))*dt
        #保存膜电位V(可改进：将点火时的电位值统一起来)
        v[:, i] = Theta2V(theta[:,i])
        #保存点火信息
        A[:,i] = (theta[:,i]>=np.pi).astype(int)
        #重置点火了的膜电位
        theta[theta[:,i]>=np.pi] -= 2*np.pi 
    
    return v, theta, g, A
    
    
    
    
    
    
    
    
    
    
    
    
