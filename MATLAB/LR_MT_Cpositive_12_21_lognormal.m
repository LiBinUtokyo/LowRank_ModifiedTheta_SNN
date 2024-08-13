% 分成E-I两组地QIF modified theta model
% 连接是一对一生成的（非平均场）
% Go_noGo Task
% 同时观察两个输入
%用对数正态分布
%笔记：
%高斯分布随机数 r = normrnd(mu,sigma,sz)
%gamma分布（指数分布随机数）r = exprnd(mu,sz1,...,szN) mu为均值
%对数正态分布 r = lognrnd(mu,sigma,sz1,...,szN)
clear

%% Setting
N_E = 2000; % 神经元数(E)
N_I = 500; % 神经元数(I)
N = N_E+N_I; % 神经元数(Total)
dt = 0.01; % 运算步长(ms/step)
T = 30; % 模拟时间总长度(ms)
tt = T/dt; % 计算步数
theta = ones(N,2)*V2theta(-70); % 位相初始值
gLE = 0.08; % 漏电导（E）
gLI = 0.1; % 漏电导（I）
g_p = [0.004069, 0.02672, 0.003276, 0.02138]; %突触电导的权重(E←E, E←I, I←E, I←I)
V_T = -55; % 点火阈值
V_R = -62; % 静息电位
V_E = 0; % 兴奋性突触反转电位
V_I = -70; % 抑制性突触反转电位
RS = 0.1; %RandomStrength

h = 2/(V_T-V_R);
q = [(2*V_E-V_R-V_T)/(V_T-V_R), (2*V_I-V_R-V_T)/(V_T-V_R)];
tau_dE = 2; % 衰减时间常数（E）(ms)
tau_dI = 5; % 衰减时间常数（I）(ms)
cp = [0.1, 0.1, 0.05, 0.2]; % 连接概率(E←E, E←I, I←E, I←I)

% conn_EE = 2*randn(N_E,N_E); % 连接矩阵(E←E)
conn_EE = 2*randn(N_E,1); % 连接矩阵(E←E),标准差为2的高斯分布
conn_EE = 1./(1+exp(-conn_EE))*2; %施加Sigmoid并乘二

conn_EI = RS*lognrnd(0,1,N_E,N_I)*(1/N_E); % 连接矩阵(E←I)
% conn_EI = 2*randn(N_E,1); % 连接矩阵(E←I),标准差为2的高斯分布
% conn_EI = 1./(1+exp(-conn_EI))*2; %施加Sigmoid并乘二
% conn_EI = abs(conn_EI);
% conn_EI = RS*exprnd(1,N_E,N_I)*(1/N_E); % 连接矩阵(E←I)

conn_IE = RS*lognrnd(0,1,N_I,N_E)*(1/N_I); % 连接矩阵(I←E)
% conn_IE = 2*randn(N_I,1); % 连接矩阵(I←E),标准差为2的高斯分布
% conn_IE = 1./(1+exp(-conn_IE))*2; %施加Sigmoid并乘二
% conn_IE = abs(conn_IE);
% conn_IE = RS*exprnd(1,N_I,N_E)*(1/N_I); % 连接矩阵(I←E)

conn_II = RS*lognrnd(0,1,N_I,N_I)*(1/N_I); % 连接矩阵(I←I)
% conn_II = 2*randn(N_I,1); % 连接矩阵(I←I),标准差为2的高斯分布
% conn_II = 1./(1+exp(-conn_II))*2; %施加Sigmoid并乘二
% conn_II = abs(conn_II);
% conn_II = RS*exprnd(1,N_I,N_I)*(1/N_I); % 连接矩阵(I←I)


% lowrank给EE，读EE
Sti_go = lognrnd(0,1,1,N_E); %取自标准差为2的对数高斯分布
% Sti_go = 1./(1+exp(-Sti_go))*2; %施加Sigmoid并乘二
Sti_nogo = lognrnd(0,2,1,N_E); %取自标准差为2的对数高斯分布
% Sti_nogo = 1./(1+exp(-Sti_nogo))*2; %施加Sigmoid并乘二

W_out = lognrnd(0,1,N_E,1);

% Sti_go = abs(Sti_go);
% Sti_nogo = abs(Sti_nogo);
% W_out = abs(W_out);


W = RS*lognrnd(0,1,N_E,N_E)*(1/N_E);
P = (W_out*Sti_go)/N_E;

conn = W + P; %连接矩阵(N*N),随机加lowRank, con = RS*X+P, P=W_out*Sti_go/N
conn_EE = conn;

% lowrank给IE，读IE
% Sti_go = 2*rand(1,N_E); %取自标准差为2的高斯分布
% Sti_go = 1./(1+exp(-Sti_go))*2; %施加Sigmoid并乘二
% Sti_nogo = 2*rand(1,N_E); %取自标准差为2的高斯分布
% 
% 
% W_out = conn_IE;
% 
% conn_IE = RS*randn(N_I,N_E)*(1/N_I) + W_out*Sti_go; %连接矩阵(E←E),随机加lowRank, con_EE = RS*X+P, P=W_out*Sti_go

% lowrank给EI，读EI
% Sti_go = 2*rand(1,N_I); %取自标准差为2的高斯分布
% Sti_go = 1./(1+exp(-Sti_go))*2; %施加Sigmoid并乘二
% Sti_nogo = 2*rand(1,N_I); %取自标准差为2的高斯分布
% 
% 
% W_out = conn_EI;
% 
% conn_EI = RS*randn(N_E,N_I)*(1/N_E) + W_out*Sti_go; %连接矩阵(E←E),随机加lowRank, con_EE = RS*X+P, P=W_out*Sti_go

% lowrank给II，读II
% Sti_go = 2*rand(1,N_I); %取自标准差为2的高斯分布
% Sti_go = 1./(1+exp(-Sti_go))*2; %施加Sigmoid并乘二
% Sti_nogo = 2*rand(1,N_I); %取自标准差为2的高斯分布
% 
% 
% W_out = conn_II;
% 
% conn_II = RS*randn(N_I,N_I)*(1/N_I) + W_out*Sti_go; %连接矩阵(E←E),随机加lowRank, con_EE = RS*X+P, P=W_out*Sti_go


rec_V = zeros(N,tt,2); % 记录膜电位
A = zeros(N,1,2); % 点火的记录

% g_EE = randn(N_E,tt,2)/10; % 突触电导(E←E)
g_EE = zeros(N_E,tt,2); % 突触电导(E←I)
g_EI = zeros(N_E,tt,2); % 突触电导(E←I)
g_IE = zeros(N_I,tt,2); % 突触电导(I←E)
g_II = zeros(N_I,tt,2); % 突触电导(I←I)

% g_EE = abs(rand(N_E,tt,2))/10; % 突触电导(E←E)
% g_EI = abs(rand(N_E,tt,2))/10; % 突触电导(E←I)
% g_IE = abs(rand(N_I,tt,2))/10; % 突触电导(I←E)
% g_II = abs(rand(N_I,tt,2))/10; % 突触电导(I←I)

 % 点火的记录(最后画图用)

    firings_inh_1 = [];
    firings_exc_1 = [];

    firings_inh_2 = [];
    firings_exc_2 = [];


I_E = [Sti_go(1:N_E)' Sti_nogo(1:N_E)']; % 输入电流(E)
I_I = [0 0]; % 输入电流(I)


%% Main
for t = 2:tt
        I_E = [0 0];
    if 5 < t/100 &&  t/100 <15
        I_E = [Sti_go(1:N_E)' Sti_nogo(1:N_E)']; % 输入电流(E)
    end

    if mod(t,tt/100)==0
        disp([num2str(t/100),'ms'])
    end

    % 计算突触电导
    g_EE(:,t,1) = g_EE(:,t-1,1) + (-g_EE(:,t-1,1)/tau_dE+g_p(1)*conn_EE*A(1:N_E,1))*dt;
    g_EI(:,t,1) = g_EI(:,t-1,1) + (-g_EI(:,t-1,1)/tau_dI+g_p(2)*conn_EI*A(N_E+1:N,1))*dt;
    g_IE(:,t,1) = g_IE(:,t-1,1) + (-g_IE(:,t-1,1)/tau_dE+g_p(3)*conn_IE*A(1:N_E,1))*dt;
    g_II(:,t,1) = g_II(:,t-1,1) + (-g_II(:,t-1,1)/tau_dI+g_p(4)*conn_II*A(N_E+1:N,1))*dt;

    g_EE(:,t,2) = g_EE(:,t-1,2) + (-g_EE(:,t-1,2)/tau_dE+g_p(1)*conn_EE*A(1:N_E,2))*dt;
    g_EI(:,t,2) = g_EI(:,t-1,2) + (-g_EI(:,t-1,2)/tau_dI+g_p(2)*conn_EI*A(N_E+1:N,2))*dt;
    g_IE(:,t,2) = g_IE(:,t-1,2) + (-g_IE(:,t-1,2)/tau_dE+g_p(3)*conn_IE*A(1:N_E,2))*dt;
    g_II(:,t,2) = g_II(:,t-1,2) + (-g_II(:,t-1,2)/tau_dI+g_p(4)*conn_II*A(N_E+1:N,2))*dt;

%     g_EE(g_EE<0) = 0;

    % 计算位相(膜电位)

    pre_theta = theta;
    % E
    tmp_theta = pre_theta(1:N_E,1);
    theta(1:N_E,1) = tmp_theta + (-gLE*cos(tmp_theta)+h*(1+cos(tmp_theta)).*I_E(:,1)+g_EE(:,t,1).*(q(1)*(1+cos(tmp_theta))- ...
        sin(tmp_theta))+g_EI(:,t,1).*(q(2)*(1+cos(tmp_theta))-sin(tmp_theta)))*dt;
    % I
    tmp_theta = pre_theta(N_E+1:N,1);
    theta(N_E+1:N,1) = tmp_theta + (-gLI*cos(tmp_theta)+h*(1+cos(tmp_theta)).*I_I(:,1)+g_IE(:,t,1).*(q(1)*(1+cos(tmp_theta))- ...
        sin(tmp_theta))+g_II(:,t,1).*(q(2)*(1+cos(tmp_theta))-sin(tmp_theta)))*dt;

    % E
    tmp_theta = pre_theta(1:N_E,2);
    theta(1:N_E,2) = tmp_theta + (-gLE*cos(tmp_theta)+h*(1+cos(tmp_theta)).*I_E(:,2)+g_EE(:,t,2).*(q(1)*(1+cos(tmp_theta))- ...
        sin(tmp_theta))+g_EI(:,t,2).*(q(2)*(1+cos(tmp_theta))-sin(tmp_theta)))*dt;
    % I
    tmp_theta = pre_theta(N_E+1:N,2);
    theta(N_E+1:N,2) = tmp_theta + (-gLI*cos(tmp_theta)+h*(1+cos(tmp_theta)).*I_I(:,2)+g_IE(:,t,2).*(q(1)*(1+cos(tmp_theta))- ...
        sin(tmp_theta))+g_II(:,t,2).*(q(2)*(1+cos(tmp_theta))-sin(tmp_theta)))*dt;

    % 保存膜电位
    rec_V(:,t,:) = (V_T+V_R)/2+(V_T-V_R)/2*tan(theta/2);

    % 处理点火了的神经元
    A = (theta >= pi);
    theta(theta >= pi) = theta(theta >= pi)-2*pi;
    % 保存点火情报
    firings_inh_1 = [firings_inh_1;t+0*find(A(N_E+1:end,1)),find(A(N_E+1:end,1))];
    firings_exc_1 = [firings_exc_1;t+0*find(A(1:N_E,1)),find(A(1:N_E,1))];

    firings_inh_2 = [firings_inh_2;t+0*find(A(N_E+1:end,2)),find(A(N_E+1:end,2))];
    firings_exc_2 = [firings_exc_2;t+0*find(A(1:N_E,2)),find(A(1:N_E,2))];

end
g = [g_EE + g_EI ; g_IE + g_II];
%输出 读EE
out = [tanh(g_EE(:,:,1)')*W_out/N_E tanh(g_EE(:,:,2)')*W_out/N_E];

%% Figure
xaxis = dt:dt:T;

% 连接矩阵描画
figure
imagesc([[conn_EE,conn_EI];[conn_IE,conn_II]])
xlabel('From')
ylabel('To')
colormap('gray');
colorbar
pbaspect([1,1,1])
title('Connectivity')

o = 1;
while o <= 2
figure
% 発火情報描画

subplot(3,1,1)
if o == 1
    scatter(firings_exc_1(:,1)*dt,firings_exc_1(:,2),2,'red','filled')%兴奋性神经元点火情况
    hold on 
    scatter(firings_inh_1(:,1)*dt,firings_inh_1(:,2),2,'blue','filled')%抑制性神经元点火情况
else
    scatter(firings_exc_2(:,1)*dt,firings_exc_2(:,2),2,'red','filled')%兴奋性神经元点火情况
    hold on 
    scatter(firings_inh_2(:,1)*dt,firings_inh_2(:,2),2,'blue','filled')%抑制性神经元点火情况
end
xlim([0 T])
title("spikes " + num2str(o))

% 膜電位描画(一部分)
subplot(3,1,2)
plot(xaxis,rec_V(1:floor(N_E*0.1)+1,:,o),'r')
hold on
% plot(xaxis,rec_V(N_E+1:floor(N_E+N_I*0.1)+1,:,o),'b')
hold off
xlim([0 T])
ylim([-200 100])
title("v " + num2str(o))

% 突触电导描画
subplot(3,1,3)
plot(xaxis, mean(g_EE(:,:,o)))
hold on
plot(xaxis, mean(g_EI(:,:,o)))
plot(xaxis, mean(g_IE(:,:,o)))
plot(xaxis, mean(g_II(:,:,o)))
plot(xaxis, mean(g(:,:,o)))
hold off
xlim([0 T])
title("g " + num2str(o))
legend('g_{EE}','g_{EI}','g_{IE}','g_{II}', 'g', 'Location','West')

figure
for i = 1:5
plot(xaxis, g_EE(i,:,o))
hold on
end

o = o + 1;
end

%观察输出
figure
hold on
limy = 3;
x = [5 5 15 15];
y = [-limy limy limy -limy];
plot(xaxis, out(:,1))
hold on
plot(xaxis, out(:,2))
patch(x,y,'blue','FaceAlpha',0.2,'EdgeColor','none')
title('Readout from Excitatory neurons')
ylim([min(min(out))*1.1 max(max(out))*1.1])
legend('O_{go}', 'O_{nogo}')


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