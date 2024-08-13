# Functions
import torch
from datetime import datetime
import numpy as np

def Draw_Output(ax,data,label_data,dt,input_data,color_data='#1C63A9'):
    # tt = np.linspace(0,len(data)-1)*dt
    tt = np.array(range(len(data)))*dt
    ax.plot(tt,data,color = color_data, label = '$'+label_data+'$')

    ax.set_xlabel('time (ms)')
    ax.set_ylabel('Read Out')

    ax.set_xlim([0, tt[-1]])
    ax.set_ylim([0, np.max([0.00001,np.max(data),ax.get_ylim()[1]])])

    non_zero_columns = np.any(input_data!=0, axis=0)
    non_zero_columns = np.where(non_zero_columns)[0]
    start_sti = non_zero_columns[0]*dt
    end_sti = non_zero_columns[-1]*dt
    ax.fill_between([start_sti,end_sti],-2,1,alpha = 0.1)
    ax.legend(loc = 1, prop={'size':10})

def Draw_Conductance(ax,data,color_data,label_data,dt,input_data):
    tt = np.array(range(len(data[0])))*dt
    if type(label_data) == list:
        for i in range(len(data)):
            ax.plot(tt,data[i],color = color_data, label = '$'+label_data+'$')
    else:
        ax.plot(tt,np.mean(data,axis=0),color = color_data, label = '$'+label_data+'$')

    ax.set_xlabel('time (ms)')
    ax.set_ylabel('Synaptic Conductance (mS/cm^2)')

    ax.set_xlim([0, tt[-1]])
    ax.set_ylim([0, np.max([0.000001,np.max(data),ax.get_ylim()[1]])])

    non_zero_columns = np.any(input_data!=0, axis=0)
    non_zero_columns = np.where(non_zero_columns)[0]
    start_sti = non_zero_columns[0]*dt
    end_sti = non_zero_columns[-1]*dt
    ax.fill_between([start_sti,end_sti],-2,1,alpha = 0.1)
    ax.legend(loc = 1, prop={'size':10})


def Draw_RasterPlot(ax,data,title_name,dt,input_data,N_E,N_I):
    tt = np.array(range(len(data[0])))*dt
    #Exc
    for row in range(data.shape[0]-N_I):
        spike_times = np.where(data[row]==1)[0]
        # ax.scatter(spike_times*dt, np.ones_like(spike_times) * row, marker='|',color = 'red')
        ax.scatter(spike_times*dt, np.ones_like(spike_times) * row, color = 'red',s=5)

    #Inh
    for row in range(N_E,data.shape[0]):
        spike_times = np.where(data[row]==1)[0]
        # ax.scatter(spike_times*dt, np.ones_like(spike_times) * row, marker='|',color = 'blue')
        ax.scatter(spike_times*dt, np.ones_like(spike_times) * row, color = 'blue',s=5)

    ax.set_xlabel('time (ms)')
    ax.set_ylabel('Neuron Index')

    ax.set_xlim([0, tt[-1]])
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