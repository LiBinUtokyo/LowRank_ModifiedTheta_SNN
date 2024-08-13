% 分成E-I两组地QIF modified theta model
% 连接是一对一生成的（非平均场）
% Go_noGo Task

clear

%% Setting
N_E = 2000; % 神经元数(E)
N_I = 500; % 神经元数(I)
N = N_E+N_I; % 神经元数(Total)
dt = 0.01; % 运算步长(ms/step)
T = 30; % 模拟时间总长度(ms)
tt = T/dt; % 计算步数
theta = ones(N,1)*V2theta(-70); % 位相初始值
gLE = 0.08; % 漏电导（E）
gLI = 0.1; % 漏电导（I）
g_p = [0.004069, 0.02672, 0.003276, 0.02138]; %突触电导的权重(E←E, E←I, I←E, I←I)
V_T = -55; % 点火阈值
V_R = -62; % 静息电位
V_E = 0; % 兴奋性突触反转电位
V_I = -70; % 抑制性突触反转电位

h = 2/(V_T-V_R);
q = [(2*V_E-V_R-V_T)/(V_T-V_R), (2*V_I-V_R-V_T)/(V_T-V_R)];
tau_dE = 2; % 衰减时间常数（E）(ms)
tau_dI = 5; % 衰减时间常数（I）(ms)
cp = [0.1, 0.1, 0.05, 0.2]; % 连接概率(E←E, E←I, I←E, I←I)
conn_EE = 2*randn(N_E,1); % 连接矩阵(E←E),标准差为2的高斯分布
conn_EI = 2*randn(N_E,N_I); % 连接矩阵(E←I)
conn_IE = 2*randn(N_I,N_E); % 连接矩阵(I←E)
conn_II = 2*randn(N_I,N_I); % 连接矩阵(I←I)

RS = 0.1; %RandomStrength
Sti_go = 2*rand(1,N_E); %取自标准差为2的高斯分布
Sti_nogo = 2*rand(1,N_E); %取自标准差为2的高斯分布
W_out = conn_EE;
conn_EE = RS*randn(N_E,N_E)*(1/N_E) + W_out*Sti_go; %连接矩阵(E←E),随机加lowRank, con_EE = RS*X+P, P=W_out*Sti_go

rec_V = zeros(N,tt); % 记录膜电位
A = zeros(N,1); % 点火的记录
g_EE = zeros(N_E,tt); % 突触电导(E←E)
g_EI = zeros(N_E,tt); % 突触电导(E←I)
g_IE = zeros(N_I,tt); % 突触电导(I←E)
g_II = zeros(N_I,tt); % 突触电导(I←I)

firings_inh = []; % 点火的记录(最后画图用)
firings_exc = [];

I_E = Sti_go'; % 输入电流(E)
I_I = 0; % 输入电流(I)


%% Main
for t = 2:tt
        I_E = 0;
    if 5 < t/100 &&  t/100 <15
        I_E = Sti_nogo';
    end

    if mod(t,tt/100)==0
        disp([num2str(t/100),'ms'])
    end

    % 计算突触电导
    g_EE(:,t) = g_EE(:,t-1) + (-g_EE(:,t-1)/tau_dE+g_p(1)*conn_EE*A(1:N_E))*dt;
    g_EI(:,t) = g_EI(:,t-1) + (-g_EI(:,t-1)/tau_dI+g_p(2)*conn_EI*A(N_E+1:N))*dt;
    g_IE(:,t) = g_IE(:,t-1) + (-g_IE(:,t-1)/tau_dE+g_p(3)*conn_IE*A(1:N_E))*dt;
    g_II(:,t) = g_II(:,t-1) + (-g_II(:,t-1)/tau_dI+g_p(4)*conn_II*A(N_E+1:N))*dt;
    % 计算位相(膜电位)
    pre_theta = theta;
    % E
    tmp_theta = pre_theta(1:N_E);
    theta(1:N_E) = tmp_theta + (-gLE*cos(tmp_theta)+h*(1+cos(tmp_theta)).*I_E+(g_EE(t)+g_EE(t))*(q(1)*(1+cos(tmp_theta))- ...
        sin(tmp_theta))+(g_EI(t)+g_EI(t))*(q(2)*(1+cos(tmp_theta))-sin(tmp_theta)))*dt;
    % I
    tmp_theta = pre_theta(N_E+1:N);
    theta(N_E+1:N) = tmp_theta + (-gLI*cos(tmp_theta)+h*(1+cos(tmp_theta)).*I_I+(g_IE(t)+g_IE(t))*(q(1)*(1+cos(tmp_theta))- ...
        sin(tmp_theta))+(g_II(t)+g_II(t))*(q(2)*(1+cos(tmp_theta))-sin(tmp_theta)))*dt;
    % 保存膜电位
    rec_V(:,t) = (V_T+V_R)/2+(V_T-V_R)/2*tan(theta/2);
    % 处理点火了的神经元
    A = (theta >= pi);
    theta(theta >= pi) = theta(theta >= pi)-2*pi;
    % 保存点火情报
    firings_inh = [firings_inh;t+0*find(A(N_E+1:end)),find(A(N_E+1:end))];
    firings_exc = [firings_exc;t+0*find(A(1:N_E)),find(A(1:N_E))];
end

%输出
out = tanh(g_EE')*W_out/N_E;

%% Figure
xaxis = dt:dt:T;
figure
% 発火情報描画
subplot(3,1,1)
scatter(firings_exc(:,1)*dt,firings_exc(:,2),2,'red','filled')%兴奋性神经元点火情况
hold on 
scatter(firings_inh(:,1)*dt,firings_inh(:,2),2,'blue','filled')%抑制性神经元点火情况
xlim([0 T])
title("spikes")

% 膜電位描画(一部分)
subplot(3,1,2)
plot(xaxis,rec_V(1:floor(N_E*0.1)+1,:),'r')
hold on
plot(xaxis,rec_V(N_E+1:floor(N_E+N_I*0.1)+1,:),'b')
hold off
xlim([0 T])
ylim([-200 100])
title("v")

% 突触电导描画
subplot(3,1,3)
plot(xaxis, mean(g_EE))
hold on
plot(xaxis, mean(g_EI))
plot(xaxis, mean(g_IE))
plot(xaxis, mean(g_II))
hold off
xlim([0 T])
title("g")
legend('g_{EE}','g_{EI}','g_{IE}','g_{II}', 'Location','West')

% 连接矩阵描画
figure
imagesc([[conn_EE,conn_EI];[conn_IE,conn_II]])
xlabel('From')
ylabel('To')
colormap('gray');
pbaspect([1,1,1])
title('Connectivity')

%观察输出
figure(3)
hold on
limy = 3;
x = [5 5 15 15];
y = [-limy limy limy -limy];
plot(xaxis, out)
hold on
patch(x,y,'blue','FaceAlpha',0.2,'EdgeColor','none')
title('Readout from Excitatory neurons')
ylim([min(out)*1.1 max(out)*1.1])


%% function

function V=theta2V(theta)
    V_T = -55; % 点火阈值
    V_R = -62; % 静息电位
    V = (V_T+V_R)/2+(V_T-V_R)/2*tan(theta/2);
end

function theta = V2theta(V)
    V_T = -55; % 点火阈值
    V_R = -62; % 静息电位
    theta = 2*atan((V-(V_R+V_T)/2)*2/(V_T-V_R));
end

















