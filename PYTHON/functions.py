# Functions
import torch
from datetime import datetime
import numpy as np
import yaml
import matplotlib.pyplot as plt
import torch.distributions as dist

# Functions for drawing
def Draw_Output(ax,data,label_data,dt,input_data,color_data='#1C63A9'):
    # tt = np.linspace(0,len(data)-1)*dt
    tt = np.array(range(len(data)))*dt
    ax.plot(tt,data,color = color_data, label = '$'+label_data+'$')

    ax.set_xlabel('time (ms)')
    ax.set_ylabel('Read Out')

    ax.set_xlim([0, tt[-1]])
    ax.set_ylim([np.min([0,np.min(data),ax.get_ylim()[0]]), np.max([0.0000001,np.max(data),ax.get_ylim()[1]])])

    non_zero_columns = np.any(input_data!=0, axis=0)
    non_zero_columns = np.where(non_zero_columns)[0]
    start_sti = non_zero_columns[0]*dt
    end_sti = non_zero_columns[-1]*dt
    ax.fill_between([start_sti,end_sti],-2,1,alpha = 0.1)
    ax.legend(loc = 1, prop={'size':10})

def Draw_Conductance(ax,data,color_data,label_data,dt,input_data,ylim=None,title=None):
    if type(label_data) == list:
        tt = np.array(range(len(data[0][0])))*dt
        for i in range(len(data)):
            ax.plot(tt,np.mean(data[i],axis=0),color = color_data[i], label = '$'+label_data[i]+'$')
        if np.max(data[0]) == 0: 
            print('g is all zero')
            return
    else:
        tt = np.array(range(len(data[0])))*dt
        ax.plot(tt,np.mean(data,axis=0),color = color_data, label = '$'+label_data+'$')
        if np.max(data) == 0: 
            print('g is all zero')
            return

    ax.set_xlabel('time (ms)')
    ax.set_ylabel('g (mS/cm^2)')

    ax.set_xlim([0, tt[-1]])

    # ax.set_ylim([0, np.max([0.00000001,np.max(data),ax.get_ylim()[1]])])
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([0,np.max(data)*1.1])
    # print(np.max(data),ax.get_ylim()[1])
    non_zero_columns = np.any(input_data!=0, axis=0)
    non_zero_columns = np.where(non_zero_columns)[0]
    start_sti = non_zero_columns[0]*dt
    end_sti = non_zero_columns[-1]*dt
    # ax.fill_between([start_sti,end_sti],0,ax.get_ylim()[1],alpha = 0.1)
    ax.fill_between([start_sti,end_sti],-2,1,alpha = 0.1)
    ax.legend(loc = 1, prop={'size':10})
    if title:
        ax.set_title(title)


def Draw_RasterPlot(ax, spk_step, spk_ind, title_name, dt, input_data, N_E, N_I):

    # ax.scatter(spk_step, spk_ind, color = 'red',s=5)
    # change the color of the Inhibitory neurons
    # for i in range(len(spk_step)):
    #     if spk_ind[i] >= N_E:
    #         ax.scatter(spk_step[i]*dt, spk_ind[i], color = 'blue',s=5)
    #         print(i)
    #     else:
    #         ax.scatter(spk_step[i]*dt, spk_ind[i], color = 'red',s=5)
    #         print(i)
    # 预先计算需要绘制的点和颜色
    x_values = np.array(spk_step) * dt
    colors = ['blue' if ind >= N_E else 'red' for ind in spk_ind]

    # 一次性绘制所有点
    ax.scatter(x_values, spk_ind, c=colors, s=1)

    # # 如果仍然需要打印 i，可以使用一个简单的 for 循环
    # for i in range(len(spk_step)):
    #     print(i)

    ax.set_xlabel('time (ms)')
    ax.set_ylabel('Neuron Index')

    ax.set_xlim([0, len(input_data[0])*dt])
    ax.set_ylim([-1, N_E+N_I])

    non_zero_columns = np.any(input_data!=0, axis=0)
    non_zero_columns = np.where(non_zero_columns)[0]
    start_sti = non_zero_columns[0]*dt
    end_sti = non_zero_columns[-1]*dt
    ax.fill_between([start_sti,end_sti],-1,N_E+N_I,alpha = 0.1)
    # ax.legend(loc = 1, prop={'size':10})
    ax.set_title(title_name)



def Draw_Voltage(ax,data,color_data,label_data,dt,input_data):
    # print(len(data) == 0)
    if len(data) == 0: return
    tt = np.array(range(len(data[0])))*dt
    
    if type(label_data) == list:
        for i in range(len(data)):
            ax.plot(tt,data[i],color = color_data, label = '$'+label_data+'$')
    else:
        ax.plot(tt,data[0],color = color_data, label = '$'+label_data+'$')
        for i in range(1,len(data)):
            ax.plot(tt,data[i],color = color_data)

    ax.set_xlabel('time (ms)')
    ax.set_ylabel('Voltage (mV)')

    ax.set_xlim([0, tt[-1]])
    ax.set_ylim([-100, 10])

    non_zero_columns = np.any(input_data!=0, axis=0)
    non_zero_columns = np.where(non_zero_columns)[0]
    start_sti = non_zero_columns[0]*dt
    end_sti = non_zero_columns[-1]*dt
    ax.fill_between([start_sti,end_sti],-100,100,alpha = 0.1)
    ax.legend(loc = 1, prop={'size':10})
    # ax.legend()

def Draw_Projection(ax,activity,direction1,direction2,title_name='Projection',color_line = '#1C63A9',xlabel = 'Activity along Direction1',ylabel = 'Activity along Direction2',ylim = None,xlim=None):
    # Calculate teh projection （using @ to calculate inner multiply）
    # activity: numpy ndarray(N,T), direction1,2: numpy ndarray(N,1)
    act_on_dir1 = activity.T@direction1 # size(T,1)
    act_on_dir2 = activity.T@direction2
    # Draw the graph
    ax.plot(act_on_dir1,act_on_dir2,color = color_line)
    ax.set_title(title_name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    #return the ylim and xlim
    return ax.get_ylim(),ax.get_xlim()

def show_mn(N,N_E, N_I, m,n,Sti_nogo,factor_mn):

    # draw the vectors m, n, Sti_nogo in heatmap
    plt.figure()
    plt.imshow(torch.cat((m, n, Sti_nogo), 1), aspect='auto',interpolation='nearest')
    plt.colorbar()
    plt.title('Vectors m, n, Sti_nogo')
    plt.show()
    print("m norm:", torch.norm(m))
    print("n norm:", torch.norm(n))
    print("Sti_nogo norm:", torch.norm(Sti_nogo))

    # W_rank1 = factor_mn*(torch.ger(m.squeeze(), n.squeeze()))
    # W_rank1 = factor_mn*torch.abs(torch.ger(m.squeeze(), n.squeeze()))
    W_rank1 = factor_mn*torch.ger(m.squeeze(), n.squeeze())
    #draw the rank-1 matrix
    plt.figure()
    plt.imshow(W_rank1,interpolation='nearest')
    plt.colorbar()
    plt.title('Rank-1 matrix')
    plt.show()
    # 展示各部分的平均值
    print("Rank-1 matrix average value_EtoE:", torch.mean(W_rank1[:N_E, :N_E]))
    print("Rank-1 matrix average value_EtoI:", torch.mean(W_rank1[:N_E, N_E:]))
    print("Rank-1 matrix average value_ItoE:", torch.mean(W_rank1[N_E:, :N_E]))
    print("Rank-1 matrix average value_ItoI:", torch.mean(W_rank1[N_E:, N_E:]))
    return W_rank1

def show_conn(N,N_E, N_I, P_EE, P_EI, P_IE, P_II,W_rank1,RS,W_random=None):
    if W_random is None:
        W_random = Generate_RandomMatrix(N_E, N_I, P_EE, P_EI, P_IE, P_II, W_rank1)
    # rank = np.linalg.matrix_rank(W_random)
    # print("矩阵的秩:", rank)
    # print("非零元素的比例:", np.count_nonzero(W_random) / (N * N))

    plt.figure()
    plt.imshow(W_random,interpolation='nearest')
    plt.colorbar()
    plt.title('Full Rank matrix')
    plt.show()
    # 展示各部分的平均值
    print("Full Rank matrix average value_EtoE:", torch.mean(W_random[:N_E, :N_E]))
    print("Full Rank matrix average value_EtoI:", torch.mean(W_random[:N_E, N_E:]))
    print("Full Rank matrix average value_ItoE:", torch.mean(W_random[N_E:, :N_E]))
    print("Full Rank matrix average value_ItoI:", torch.mean(W_random[N_E:, N_E:]))

    W_conn = W_rank1 + RS * W_random
    W_conn[W_conn > 1] = 1
    W_conn[W_conn < 0] = 0
    plt.figure()
    plt.imshow(W_conn,interpolation='nearest')
    plt.colorbar()
    plt.title('Connectivity matrix')
    plt.show()
    # 展示各部分的平均值
    print("Connectivity matrix average value_EtoE:", torch.mean(W_conn[:N_E, :N_E]))
    print("Connectivity matrix average value_EtoI:", torch.mean(W_conn[:N_E, N_E:]))
    print("Connectivity matrix average value_ItoE:", torch.mean(W_conn[N_E:, :N_E]))
    print("Connectivity matrix average value_ItoI:", torch.mean(W_conn[N_E:, N_E:]))
    return W_conn



# Functions for generating matrices
def Generate_Vectors(N, mu=0, sigma=0.1,seed=None):
    # seed: random seed
    # m,n,sti_nogo 从 gaussian 分布中采样
    if seed:
        m = torch.normal(mu, sigma, (N,1), generator=seed)
        n = torch.normal(mu, sigma, (N,1), generator=seed)
        sti_nogo = torch.normal(mu, sigma, (N,1), generator=seed)
    else:
        m = torch.normal(mu, sigma, (N,1))
        n = torch.normal(mu, sigma, (N,1))
        sti_nogo = torch.normal(mu, sigma, (N,1))
    return m, n, sti_nogo

def ab_gamma(mu, sigma):
    # mu: mean of gamma distribution
    # sigma: standard deviation of gamma distribution
    # return a, b parameters of gamma distribution
    # a = mu^2/sigma^2, b = mu/sigma^2
    a = mu ** 2 / sigma ** 2
    b = mu / sigma ** 2
    return a, b

def Generate_RandomMatrix(N_E, N_I, P_EE, P_EI, P_IE, P_II, W_rank1, sigma=0.1):
    # Construct random weight matrix
    # use beta distribution to generate random matrix
    # W_rank1: low rank matrix
    # deside the average value of the beta distribution according to the rank-1 matrix to make sure
    # the average value of the sum of the rank-1 matrix and the random matrix is according to the P_EE, P_EI, P_IE, P_II
    N = N_E + N_I
    W = torch.zeros(N, N)

    mu_EE = P_EE - torch.sum(W_rank1[:N_E, :N_E]) / (N_E * N_E)
    mu_EI = P_EI - torch.sum(W_rank1[:N_E, N_E:]) / (N_E * N_I)
    mu_IE = P_IE - torch.sum(W_rank1[N_E:, :N_E]) / (N_I * N_E)
    mu_II = P_II - torch.sum(W_rank1[N_E:, N_E:]) / (N_I * N_I)

    # 生成a,b参数
    # 打印出来看看
    a_EE, b_EE = ab_gamma(mu_EE, sigma)
    a_EI, b_EI = ab_gamma(mu_EI, sigma)
    a_IE, b_IE = ab_gamma(mu_IE, sigma)
    a_II, b_II = ab_gamma(mu_II, sigma)

    # print("a_EE, b_EE, average value:", a_EE, b_EE, mu_EE)
    # print("a_EI, b_EI, average value:", a_EI, b_EI, mu_EI)
    # print("a_IE, b_IE, average value:", a_IE, b_IE, mu_IE)
    # print("a_II, b_II, average value:", a_II, b_II, mu_II)

    # 生成连接矩阵
    W[:N_E, :N_E] = dist.Gamma(a_EE,b_EE).sample((N_E,N_E))
    W[:N_E, N_E:] = dist.Gamma(a_EI, b_EI).sample((N_E, N_I))
    W[N_E:, :N_E] = dist.Gamma(a_IE, b_IE).sample((N_I, N_E))
    W[N_E:, N_E:] = dist.Gamma(a_II, b_II).sample((N_I, N_I))
    return W


# functions for saving and loading parameters
def save_model(LRSNN,dt,Sti_go,Sti_nogo,Input_go,Input_nogo,IS,m,n,path='/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/models/'):
    now = datetime.now()
    formatted_now = now.strftime("%Y_%m_%d_%H_%M")
    torch.save({
        'model':LRSNN,
        'Input_go':Input_go,
        'Input_nogo':Input_nogo,
        'dt':dt,
        'IS':IS,
        'Sti_go':Sti_go,
        'Sti_nogo':Sti_nogo,
        'm':m,
        'n':n,
    },f'{path}{formatted_now}.pth')

def load_config_yaml(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_init(LRSNN, T_pre, dt, g_ref, g_ref_EE, g_ref_EI, g_ref_IE, g_ref_II, V_ref, phase_ref, I_ref_syn, I_ref_syn_EE, I_ref_syn_EI, I_ref_syn_IE, I_ref_syn_II, spk_ref):
    # load the predefined values into the model
    step_init = int(T_pre/dt)
    g_init = g_ref[:,step_init].clone().detach()
    g_init_EE = g_ref_EE[:,step_init].clone().detach()
    g_init_EI = g_ref_EI[:,step_init].clone().detach()
    g_init_IE = g_ref_IE[:,step_init].clone().detach()
    g_init_II = g_ref_II[:,step_init].clone().detach()
    V_init = V_ref[:,step_init].clone().detach()
    phase_init = phase_ref[:,step_init].clone().detach()
    I_syn_init = I_ref_syn[:,step_init].clone().detach()
    I_syn_init_EE = I_ref_syn_EE[:,step_init].clone().detach()
    I_syn_init_EI = I_ref_syn_EI[:,step_init].clone().detach()
    I_syn_init_IE = I_ref_syn_IE[:,step_init].clone().detach()
    I_syn_init_II = I_ref_syn_II[:,step_init].clone().detach()
    spk_init = spk_ref[:,step_init]
    LRSNN.load_init(g_init, g_init_EE, g_init_EI, g_init_IE, g_init_II, V_init, phase_init, I_syn_init, I_syn_init_EE, I_syn_init_EI, I_syn_init_IE, I_syn_init_II, spk_init)
    return LRSNN