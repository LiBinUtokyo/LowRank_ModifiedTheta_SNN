# Functions
import torch
from datetime import datetime
import numpy as np
import yaml
import matplotlib.pyplot as plt

# Functions
def Draw_Output(ax,data,label_data,dt,input_data,color_data='#1C63A9'):
    # tt = np.linspace(0,len(data)-1)*dt
    tt = np.array(range(len(data)))*dt
    ax.plot(tt,data,color = color_data, label = '$'+label_data+'$')

    ax.set_xlabel('time (ms)')
    ax.set_ylabel('Read Out')

    ax.set_xlim([0, tt[-1]])
    ax.set_ylim([0, np.max([0.0000001,np.max(data),ax.get_ylim()[1]])])

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