# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:00:34 2022
Low-Rank-Sturcture-Modified-Theta-Model
先是不区分抑制性和兴奋性的版本
@author: li
"""
import matplotlib.pyplot as plt
import numpy as np
import os

import fct_simulations as sim
import fct_facilities as fac
#%% Initialization 初始化
path_local = os.getcwd()

RS = 0.1 #Random Strength

std_Sti = 2. #Standerd Deviration of Stimuli
std_Wout = 2. #Standerd Deviration of readout matrix

N = 2500 #Number of neurons

T0 = 20 #Time for initial transient(discard)(ms)
T1 = 5 #Time for resting state(ms)
T2 = 10 #Time for stimulus presentation(ms)
#T2 = 20
T3 = 15 #Time for decay to rest(ms)
T3 = 30

deltat = 0.1 #time step(ms)

t0 = np.arange(0, T0, deltat)
t1 = np.arange(0, T1, deltat)
t2 = np.arange(0, T2, deltat)
t3 = np.arange(0, T3, deltat)

t = np.concatenate([t1,T1+t2,T1+T2+t3]) #总时间,t0不考虑

OutPut = np.zeros([2, len(t)])
g_rec = np.zeros([2, N, len(t)])
spk_rec = np.zeros([2, N, len(t)])

Wout = sim.GetGaussianVector(0,std_Wout,N) #输出矩阵
Sti_go = sim.GetGaussianVector(0,std_Sti,N)#Go输入信号
Sti_nogo = sim.GetGaussianVector(0,std_Sti,N) #nogo signal

# Sti_go = np.absolute(Sti_go)#为保证生物学可解释性？使输入恒为正
# Sti_nogo = np.absolute(Sti_nogo)

structure = np.array([Wout,Sti_go,Sti_nogo]) #决定该网络结构的参数

P = np.outer(Wout, Sti_go)/N #Let the connectiviey vector n and m equal to Wout and Sti_go 结构矩阵
X = sim.GetBulk(N) #随机矩阵
J = RS*X + P #connectivity matrix

#%%Simulation 开始模拟
# Step 0: simulate and discard the transient

v, theta, g, spk = sim.SimulateActivity(t0, sim.GetGaussianVector(0,0.1,N), np.array([-62]*N).T, J, Sti = 0) 

# Step 1: no inputs

v1, theta1, g, spk = sim.SimulateActivity(t1, g[:,-1], v[:,-1], J, Sti = 0) 

g_rec[0, :, 0:len(t1)] = g
g_rec[1, :, 0:len(t1)] = g

spk_rec[0, :, 0:len(t1)] = spk
spk_rec[1, :, 0:len(t1)] = spk

OutPut[0, 0:len(t1)] = np.dot(np.tanh(g).T, Wout) / N
OutPut[1, 0:len(t1)] = np.dot(np.tanh(g).T, Wout) / N

# Step 2: Go and Nogo Stimulus

v2go, theta2go, ggo, spk2go = sim.SimulateActivity(t2, g[:,-1], v1[:,-1], J, Sti = Sti_go) 
v2no, theta2no, gno, spk2no = sim.SimulateActivity(t2, g[:,-1], v1[:,-1], J, Sti = Sti_nogo) 

g_rec[0,:,len(t1):len(t1)+len(t2)] = ggo
g_rec[1,:,len(t1):len(t1)+len(t2)] = gno

spk_rec[0, :, len(t1):len(t1)+len(t2)] = spk2go
spk_rec[1, :, len(t1):len(t1)+len(t2)] = spk2no

OutPut[0, len(t1):len(t1)+len(t2)] = np.dot(np.tanh(ggo).T, Wout) / N
OutPut[1, len(t1):len(t1)+len(t2)] = np.dot(np.tanh(gno).T, Wout) / N


# Step 3: no input

v3go, theta3go, ggo, spk3go = sim.SimulateActivity(t3, ggo[:,-1], v2go[:,-1], J, Sti = 0) 
v3no, theta3no, gno, spk3no = sim.SimulateActivity(t3, gno[:,-1], v2no[:,-1], J, Sti = 0) 

g_rec[0,:, len(t1)+len(t2):] = ggo
g_rec[1,:, len(t1)+len(t2):] = gno

spk_rec[0, :, len(t1)+len(t2):] = spk3go
spk_rec[1, :, len(t1)+len(t2):] = spk3no

OutPut[0, len(t1)+len(t2):] = np.dot(np.tanh(ggo).T, Wout) / N
OutPut[1, len(t1)+len(t2):] = np.dot(np.tanh(gno).T, Wout) / N

#%%Drawing Graphs 绘图
m = structure[0] #m = Wout
n = structure[1] #n = Stigo

color_Go = '#1C63A9'
color_Nogo = '#009999'


#检查synaptic conductance g(平均值和取几个看)
fig, ax = plt.subplots()
ax.plot(t,np.mean(g_rec[0],axis=0),color = color_Go, label = '$g_{go}$')
ax.plot(t,np.mean(g_rec[1],axis=0),color = color_Nogo, label = '$g_{nogo}$')
ax.set_xlabel('time (ms)')
ax.set_ylabel('g $(mS/cm^2)$')
ax.set_title('Average Synaptic Conductance')

ax.legend(loc = 1, prop={'size':10})

# s = 4
# fig, ax = plt.subplots()
# shift = 0
# for i in range(s):
#     ax.plot(t,g_rec[0,i,:]+shift,color = color_Go)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     shift += 0.03

# s = 4
# fig, ax = plt.subplots()
# for i in range(s):
#     ax.plot(t,g_rec[1,i,:],color = color_Nogo)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

s = 7
fig, axs = plt.subplots(s,1,sharex = True)
for i in range(s):
    axs[i].plot(t,g_rec[0,i,:],color = color_Go)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    
#fig, axs = plt.subplots(s,1,sharex = True)
for i in range(s):
    axs[i].plot(t,g_rec[1,i,:],color = color_Nogo)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
axs[s-1].set_xlabel('time (ms)')
    
#检查点火率

#检查连接矩阵（随机加结构）

#检查电压

#检查点火情况

fig, ax = plt.subplots()
x, y = np.where(spk_rec[0] == 1)
ax.scatter(y*deltat, x, s=0.1, color = color_Go, label = '$Spikes_{go}$')
ax.set_xlim([0, t[-1]])
ax.set_xlabel('time (ms)')
ax.set_ylabel('Neuron')
ax.set_title('Record of Firings in a Go Task')

fig, ax = plt.subplots()
x, y = np.where(spk_rec[1] == 1)
ax.scatter(y*deltat, x, s=0.1, color = color_Nogo, label = '$Spikes_{nogo}$')
ax.set_xlim([0, t[-1]])
ax.set_xlabel('time (ms)')
ax.set_ylabel('Neuron')
ax.set_title('Record of Firings in a No-go Task')

#检查对应输入的输出结果
fig, ax = plt.subplots()
ax.plot(t, OutPut[0],color = color_Go, label = '$O_{go}$')
ax.plot(t, OutPut[1],color = color_Nogo, label = '$O_{nogo}$')

ax.set_xlabel('time (ms)')
ax.set_ylabel('Read Out')

ax.set_xlim([0, t[-1]])
ax.set_ylim([-0.0002, 0.004])
ax.fill_between([T1,T1+T2],-2,1,alpha = 0.1)
ax.legend(loc = 1, prop={'size':10})


# Project the Go trials on the m-IA plane

fg = plt.figure()
ax0 = plt.axes()


on_m = np.dot( g_rec.transpose(0,2,1), m ) / N
on_IA = np.dot( g_rec.transpose(0,2,1), Sti_go ) / N
on_IB = np.dot( g_rec.transpose(0,2,1), Sti_nogo ) / N

plt.plot(on_IA[0,:], on_m[0,:], color = color_Go)

plt.xlabel(r'$\delta I$')
plt.ylabel(r'$m$')

plt.xlim(-0.00001, 0.00007)
plt.ylim(-0.001, 0.004)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.show()

# Project the Nogo trials on the m-IB plane

fg = plt.figure()
ax0 = plt.axes()

plt.plot(on_IB[1,:], on_m[1,:], color = color_Nogo)

plt.xlabel(r'$\delta I$')
plt.ylabel(r'$m$')

plt.xlim(-0.00001, 0.00007)
plt.ylim(-0.001, 0.004)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.show()


