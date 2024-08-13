%练习重写杉野的Modified Theta Model的simulation
%杉野注：此为E-I groups的QIF Modified Theta Model， 其中各神经的连接为各自设定，而不是平均设定，输入电流服从洛伦兹分布
clear

% Setting Parameters

N_E = 400; %Number of Exitatory Neurons
N_I = 400; %Number of Inhibitory Neurons
N = N_E + N_I; %总神经元数

dt = 0.01; % step size,步长(ms)
T = 500; % 总模拟时间（ms）
all_steps = T/dt; %总步长数

theta = 2*pi*rand(N,1); %位相初始值,这里和杉野不一样，根据野山的解释，各神经元位相初始值只要随机选取即可，
% 范围从0到2pi,杉野是取全部一样的值（理由不明）

g_E = 0.08; %兴奋性突触电导(mS)
g_I = 0.1; %抑制性突触电导(mS)
g_p = [0.004069, 0.02672, 0.003276, 0.02138]; %根据杉野定义，此为兴奋性神经元群与抑制性神经元群的相互作用强度权重，例如E->E, E->I, I->I, I->E，


V_peak = -55; %Votage peak (mV)
V_rest = -62; %静息电位(mV)
V_E = 0; %兴奋性神经元反转电位
V_I = -70; %抑制性神经元反转电位

h = 0; %不知道是什么所以不理它
q = []; %同上

tau_d = 10; %synaptic decay time(ms)这里和杉野不一样

tau_dE = 0; %synaptic decay time(ms) for exitatory neurons 没理它
tau_dI = 0; %synaptic decay time(ms) for inhibitory neurons 没理它

cp = [0.1, 0.1, 0.1, 0.1]; %connect probability (EtoE,EtoI,ItoE,ItoI)
conn_EE = rand(N_E,N_E)<cp(1);%兴奋性对兴奋性神经元的连接矩阵
conn_EI = rand(N_E,N_I)<cp(2);%兴奋性对抑制性神经元的连接矩阵
conn_IE = rand(N_I,N_E)<cp(3);%抑制性对兴奋性神经元的连接矩阵
conn_II = rand(N_I,N_I)<cp(4);%抑制性对抑制性神经元的连接矩阵

Eta = [1, 2];%输入电流的平均值（？）
Delta = [0.05, 0.05]; %输入电流的半峰宽度（？）

rec_V = zeros(N,all_steps);%记录膜电位变化
spk = zeros(N,all_steps);%点火记录

I_E = Eta(1) + Delta(1)*tan(pi*((1:N_E)'/(N_E+1)-1/2)); %兴奋性神经元的输入电流(?)
I_I = Eta(2) + Delta(2)*tan(pi*((1:N_I)'/(N_I+1)-1/2)); %抑制性神经元的输入电流(?)



%Simulation
for step = 2:all_steps
    %计算不同类型神经元间连接的电导
    g_EE(:,step) = g_EE(:,step-1) + dt*(-g_EE(:,step-1)/tau_dE+g_p(1)*conn_EE*spk(1:N_E,step-1));
    g_EI(:,step) = g_EI(:,step-1) + dt*(-g_EI(:,step-1)/tau_dI+g_p(2)*conn_EI*spk(1+N_E:N,step-1));
    g_IE(:,step) = g_IE(:,step-1) + dt*(-g_IE(:,step-1)/tau_dE+g_p(3)*conn_IE*spk(1:N_E,step-1));
    g_II(:,step) = g_II(:,step-1) + dt*(-g_II(:,step-1)/tau_dI+g_p(4)*conn_II*spk(1+N_E:N,step-1));%这些式子从哪来的



    
end



%Result Drawing







































