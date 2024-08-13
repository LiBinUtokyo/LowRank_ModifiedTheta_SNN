% 分成E-I两组地QIF modified theta model
% 连接是一对一生成的（非平均场）
% Go_noGo Task
% 同时观察两个输入
%将兴奋性神经元分成Sensory和Moto两部分
%将抑制性神经元分成Sensory和Moto两部分
%输入给sensory部分，读moto部分

clear
tic
%% Setting
N_E = 1000; % 神经元数(E)
N_I = 200; % 神经元数(I)
N_Es = N_E/2;
N_Em = N_E/2;
N_Is = N_I/2;
N_Im = N_I/2;

N = N_E+N_I; % 神经元数(Total)
dt = 0.01; % 运算步长(ms/step)
T = 60; % 模拟时间总长度(ms)
tt = T/dt; % 计算步数
theta = ones(N,2)*V2theta(-70); % 位相初始值
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

RS = 3; %RandomStrength
IS = 3;% Input Strength,if it is 1, n=IS*Sti_go 

%n,m的参数
mu_n = 50;
si_n = 10;

b_n = si_n^2/mu_n;%gamma分布
a_n = mu_n/b_n;

% normrnd(mu_n,si_n)
% gamrnd(a_n,b_n)

mu_m = 50;
si_m = 10;

b_m = si_m^2/mu_m;%gamma分布
a_m = mu_m/b_m;


% conn_EsEs = RS*normrnd(0,sqrt(1/N_E),N_E/2,N_E/2); % 连接矩阵(Es←Es)
conn_EsEs = RS*normrnd(0,sqrt(1/N_E),N_E/2,N_E/2); % 连接矩阵(Es←Es),标准差为2的高斯分布
conn_EsEs = abs(-conn_EsEs); %施加Sigmoid并乘二

conn_EmEs = RS*normrnd(0,sqrt(1/N_E),N_E/2,N_E/2); % 连接矩阵(Em←Es)
% conn_EmEs = 2*randn(N_E/2,1); % 连接矩阵(Em←Es),标准差为2的高斯分布
conn_EmEs = abs(-conn_EmEs); %施加Sigmoid并乘二


conn_EsEm = RS*normrnd(0,sqrt(1/N_E),N_E/2,N_E/2); % 连接矩阵(Es←Em)
% conn_EsEm = 2*randn(N_E/2,1); % 连接矩阵(Es←Em),标准差为2的高斯分布
conn_EsEm = abs(-conn_EsEm); %施加Sigmoid并乘二


conn_EmEm = RS*normrnd(0,sqrt(1/N_E),N_E/2,N_E/2); % 连接矩阵(Em←Em)
% conn_EmEm = 2*randn(N_E/2,1); % 连接矩阵(Em←Em),标准差为2的高斯分布
conn_EmEm = abs(-conn_EmEm); %施加Sigmoid并乘二



conn_EsIs = RS*normrnd(0,sqrt(1/N_I),N_E/2,N_I/2); % 连接矩阵(Es←Is)
% conn_EsIs = 2*randn(N_E/2,1); % 连接矩阵(E←I),标准差为2的高斯分布
conn_EsIs = abs(-conn_EsIs); %施加Sigmoid并乘二

conn_EmIs = RS*normrnd(0,sqrt(1/N_I),N_E/2,N_I/2); % 连接矩阵(Em←Is)
% conn_EmIs = 2*randn(N_E/2,1); % 连接矩阵(E←I),标准差为2的高斯分布
conn_EmIs = abs(-conn_EmIs); %施加Sigmoid并乘二

conn_EsIm = RS*normrnd(0,sqrt(1/N_I),N_E/2,N_I/2); % 连接矩阵(Es←Im)
% conn_EsIm = 2*randn(N_E/2,1); % 连接矩阵(E←I),标准差为2的高斯分布
conn_EsIm = abs(-conn_EsIm); %施加Sigmoid并乘二

conn_EmIm = RS*normrnd(0,sqrt(1/N_I),N_E/2,N_I/2); % 连接矩阵(Em←Im)
% conn_EmIm = 2*randn(N_E/2,1); % 连接矩阵(E←I),标准差为2的高斯分布
conn_EmIm = abs(-conn_EmIm); %施加Sigmoid并乘二


conn_IsEs = RS*normrnd(0,sqrt(1/N_E),N_I/2,N_E/2); % 连接矩阵(Is←Es)
% conn_IsEs = 2*randn(N_I/2,1); % 连接矩阵(Is←Es),标准差为2的高斯分布
conn_IsEs = abs(-conn_IsEs); %施加Sigmoid并乘二

conn_ImEs = RS*normrnd(0,sqrt(1/N_E),N_I/2,N_E/2); % 连接矩阵(Im←Es)
% conn_ImEs = 2*randn(N_I/2,1); % 连接矩阵(Im←Es),标准差为2的高斯分布
conn_ImEs = abs(-conn_ImEs); %施加Sigmoid并乘二

conn_IsEm = RS*normrnd(0,sqrt(1/N_E),N_I/2,N_E/2); % 连接矩阵(Is←Em)
% conn_IsEm = 2*randn(N_I/2,1); % 连接矩阵(Is←Em),标准差为2的高斯分布
conn_IsEm = abs(-conn_IsEm); %施加Sigmoid并乘二

conn_ImEm = RS*normrnd(0,sqrt(1/N_E),N_I/2,N_E/2); % 连接矩阵(Im←Em)
% conn_ImEm = 2*randn(N_I/2,1); % 连接矩阵(Is←Es),标准差为2的高斯分布
conn_ImEm = abs(-conn_ImEm); %施加Sigmoid并乘二


conn_IsIs = RS*normrnd(0,sqrt(1/N_I),N_I/2,N_I/2); % 连接矩阵(Is←Is)
% conn_IsIs = 2*randn(N_I/2,1); % 连接矩阵(Is←Is),标准差为2的高斯分布
conn_IsIs = abs(-conn_IsIs); %施加Sigmoid并乘二

conn_ImIs = RS*normrnd(0,sqrt(1/N_I),N_I/2,N_I/2); % 连接矩阵(Im←Is)
% conn_ImIs = 2*randn(N_I/2,1); % 连接矩阵(Im←Is),标准差为2的高斯分布
conn_ImIs = abs(-conn_ImIs); %施加Sigmoid并乘二

conn_IsIm = RS*normrnd(0,sqrt(1/N_I),N_I/2,N_I/2); % 连接矩阵(Is←Im)
% conn_IsIm = 2*randn(N_I/2,1); % 连接矩阵(Is←Im),标准差为2的高斯分布
conn_IsIm = abs(-conn_IsIm); %施加Sigmoid并乘二

conn_ImIm = RS*normrnd(0,sqrt(1/N_I),N_I/2,N_I/2); % 连接矩阵(Im←Im)
% conn_ImIm = 2*randn(N_I/2,1); % 连接矩阵(Im←Im),标准差为2的高斯分布
conn_ImIm = abs(-conn_ImIm); %施加Sigmoid并乘二


% lowrank给EsEs, EmEs, EsEm, EmEm，读EmEm
temp = rand([1,N_E/2]);


Sti_go = temp;
Sti_go(Sti_go>=1/30) = 0;%n = Sti_go
% Sti_go(Sti_go~=0) = normrnd(mu_n,si_n,1,length(find(Sti_go~=0)));
Sti_go(Sti_go~=0) = gamrnd(a_n,b_n,1,length(find(Sti_go~=0)));
Sti_go(Sti_go<0) = 0;

Sti_nogo = temp;
Sti_nogo(Sti_nogo>=15/30 | Sti_nogo<14/30) = 0;
% Sti_nogo(Sti_nogo~=0) = normrnd(mu_n,si_n,1,length(find(Sti_nogo~=0)));
Sti_nogo(Sti_nogo~=0) = gamrnd(a_n,b_n,1,length(find(Sti_nogo~=0)));
Sti_nogo(Sti_nogo<0) = 0;

W_out = temp;
W_out(W_out<29/30) = 0;%m = Wout,可以通过调整范围决定两个向量重合的程度
% W_out(W_out~=0) = normrnd(mu_m,si_m,1,length(find(W_out~=0)));
W_out(W_out~=0) = gamrnd(a_m,b_m,1,length(find(W_out~=0)));
W_out = W_out';
W_out(W_out<0) = 0;

W = RS*normrnd(0,sqrt(1/N_E),N_E/2,N_E/2); 
W = abs(W);

P = (W_out*Sti_go)/N_E;

conn = W+P;
conn(conn>1) = 1;

[conn_EsEs, conn_EmEs, conn_EsEm, conn_EmEm] = deal(conn); %连接矩阵(E←E),随机加lowRank, con_EE = RS*X+P, P=W_out*Sti_go


% conn_EsEs = RS*randn(N_E/2,N_E/2)*(1/N_E) + W_out*Sti_go;
% conn_IsIs = RS*randn(N_I/2,N_I/2)*(1/N_I) + W_out*Sti_go;

% lowrank给IE，读IE
% Sti_go = 2*rand(1,N_E); %取自标准差为2的高斯分布
% Sti_go = abs(-Sti_go); %施加Sigmoid并乘二
% Sti_nogo = 2*rand(1,N_E); %取自标准差为2的高斯分布
% 
% 
% W_out = conn_IE;
% 
% conn_IE = RS*randn(N_I,N_E)*(1/N_I) + W_out*Sti_go; %连接矩阵(E←E),随机加lowRank, con_EE = RS*X+P, P=W_out*Sti_go

% lowrank给EI，读EI
% Sti_go = 2*rand(1,N_I); %取自标准差为2的高斯分布
% Sti_go = abs(-Sti_go); %施加Sigmoid并乘二
% Sti_nogo = 2*rand(1,N_I); %取自标准差为2的高斯分布
% 
% 
% W_out = conn_EI;
% 
% conn_EI = RS*randn(N_E,N_I)*(1/N_E) + W_out*Sti_go; %连接矩阵(E←E),随机加lowRank, con_EE = RS*X+P, P=W_out*Sti_go

% lowrank给II，读II
% Sti_go = 2*rand(1,N_I); %取自标准差为2的高斯分布
% Sti_go = abs(-Sti_go); %施加Sigmoid并乘二
% Sti_nogo = 2*rand(1,N_I); %取自标准差为2的高斯分布
% 
% 
% W_out = conn_II;
% 
% conn_II = RS*randn(N_I,N_I)*(1/N_I) + W_out*Sti_go; %连接矩阵(E←E),随机加lowRank, con_EE = RS*X+P, P=W_out*Sti_go


rec_V = zeros(N,tt,2); % 记录膜电位
A = zeros(N,1,2); % 点火的记录

[g_EsEs, g_EmEs, g_EsEm, g_EmEm] = deal(zeros(N_E/2,tt,2)); % 突触电导(Es←Es, Em←Es, Es←Em, Em←Em)
[g_EsIs, g_EmIs, g_EsIm, g_EmIm] = deal(zeros(N_E/2,tt,2)); % 突触电导(Es←Is, Em←Is, Es←Im, Em←Im)
[g_IsEs, g_ImEs, g_IsEm, g_ImEm] = deal(zeros(N_I/2,tt,2)); % 突触电导(Is←Es, Im←Es, Is←Em, Im←Em)
[g_IsIs, g_ImIs, g_IsIm, g_ImIm] = deal(zeros(N_I/2,tt,2)); % 突触电导(Is←Is, Im←Is, Is←Im, Im←Im)

% g_EE = abs(rand(N_E,tt,2))/10; % 突触电导(E←E)
% g_EI = abs(rand(N_E,tt,2))/10; % 突触电导(E←I)
% g_IE = abs(rand(N_I,tt,2))/10; % 突触电导(I←E)
% g_II = abs(rand(N_I,tt,2))/10; % 突触电导(I←I)

 % 点火的记录(最后画图用)

    firings_inh_1 = [];
    firings_exc_1 = [];

    firings_inh_2 = [];
    firings_exc_2 = [];


I_E = [Sti_go' Sti_nogo';0*Sti_go' 0*Sti_nogo']; % 输入电流(E)
I_I = [0 0]; % 输入电流(I)


%% Main
for t = 2:tt
        I_E = [0 0];
    if 5 < t/100 &&  t/100 <15
        I_E = [Sti_go' Sti_nogo';0*Sti_go' 0*Sti_nogo'];
    end

    if mod(t,tt/100)==0
        disp([num2str(t/100),'ms'])
    end

    % 计算突触电导
    for i = [1 2]
    g_EsEs(:,t,i) = g_EsEs(:,t-1,i) + (-g_EsEs(:,t-1,i)/tau_dE+g_p(1)*conn_EsEs*A(1:N_Es,i))*dt;
    g_EmEs(:,t,i) = g_EmEs(:,t-1,i) + (-g_EmEs(:,t-1,i)/tau_dE+g_p(1)*conn_EmEs*A(1:N_Es,i))*dt;
    g_EsEm(:,t,i) = g_EsEs(:,t-1,i) + (-g_EsEs(:,t-1,i)/tau_dE+g_p(1)*conn_EsEs*A(N_Es+1:N_E,i))*dt;
    g_EmEm(:,t,i) = g_EsEs(:,t-1,i) + (-g_EsEs(:,t-1,i)/tau_dE+g_p(1)*conn_EsEs*A(N_Es+1:N_E,i))*dt;

    g_EsIs(:,t,i) = g_EsIs(:,t-1,i) + (-g_EsIs(:,t-1,i)/tau_dI+g_p(2)*conn_EsIs*A(N_E+1:N_E+N_Is,i))*dt;
    g_EmIs(:,t,i) = g_EmIs(:,t-1,i) + (-g_EmIs(:,t-1,i)/tau_dI+g_p(2)*conn_EmIs*A(N_E+1:N_E+N_Is,i))*dt;
    g_EsIm(:,t,i) = g_EsIm(:,t-1,i) + (-g_EsIm(:,t-1,i)/tau_dI+g_p(2)*conn_EsIm*A(N_E+N_Is+1:N,i))*dt;
    g_EmIm(:,t,i) = g_EmIm(:,t-1,i) + (-g_EmIm(:,t-1,i)/tau_dI+g_p(2)*conn_EmIm*A(N_E+N_Is+1:N,i))*dt;

    g_IsEs(:,t,i) = g_IsEs(:,t-1,i) + (-g_IsEs(:,t-1,i)/tau_dE+g_p(3)*conn_IsEs*A(1:N_Es,i))*dt;
    g_ImEs(:,t,i) = g_ImEs(:,t-1,i) + (-g_ImEs(:,t-1,i)/tau_dE+g_p(3)*conn_ImEs*A(1:N_Es,i))*dt;
    g_IsEm(:,t,i) = g_IsEm(:,t-1,i) + (-g_IsEm(:,t-1,i)/tau_dE+g_p(3)*conn_IsEm*A(N_Es+1:N_E,i))*dt;
    g_ImEm(:,t,i) = g_ImEm(:,t-1,i) + (-g_ImEm(:,t-1,i)/tau_dE+g_p(3)*conn_ImEm*A(N_Es+1:N_E,i))*dt;

    g_IsIs(:,t,i) = g_IsIs(:,t-1,i) + (-g_IsIs(:,t-1,i)/tau_dI+g_p(4)*conn_IsIs*A(N_E+1:N_E+N_Is,i))*dt;
    g_ImIs(:,t,i) = g_ImIs(:,t-1,i) + (-g_ImIs(:,t-1,i)/tau_dI+g_p(4)*conn_ImIs*A(N_E+1:N_E+N_Is,i))*dt;
    g_IsIm(:,t,i) = g_IsIm(:,t-1,i) + (-g_IsIm(:,t-1,i)/tau_dI+g_p(4)*conn_IsIm*A(N_E+N_Is+1:N,i))*dt;
    g_ImIm(:,t,i) = g_ImIm(:,t-1,i) + (-g_ImIm(:,t-1,i)/tau_dI+g_p(4)*conn_ImIm*A(N_E+N_Is+1:N,i))*dt;

    end

%     g_EE(:,t,2) = g_EE(:,t-1,2) + (-g_EE(:,t-1,2)/tau_dE+g_p(1)*conn_EE*A(1:N_E,2))*dt;
%     g_EI(:,t,2) = g_EI(:,t-1,2) + (-g_EI(:,t-1,2)/tau_dI+g_p(2)*conn_EI*A(N_E+1:N,2))*dt;
%     g_IE(:,t,2) = g_IE(:,t-1,2) + (-g_IE(:,t-1,2)/tau_dE+g_p(3)*conn_IE*A(1:N_E,2))*dt;
%     g_II(:,t,2) = g_II(:,t-1,2) + (-g_II(:,t-1,2)/tau_dI+g_p(4)*conn_II*A(N_E+1:N,2))*dt;

    % 计算位相(膜电位)

    pre_theta = theta;
    % E
    tmp_theta = pre_theta(1:N_E,1);
    theta(1:N_E,1) = tmp_theta + (-gLE*cos(tmp_theta)+h*(1+cos(tmp_theta)).*I_E(:,1)+([g_EsEs(:,t,1);g_EmEs(:,t,1)]+[g_EsEm(:,t,1);g_EmEm(:,t,1)]).*(q(1)*(1+cos(tmp_theta))- ...
        sin(tmp_theta))+([g_EsIs(:,t,1);g_EmIs(:,t,1)]+[g_EsIm(:,t,1);g_EmIm(:,t,1)]).*(q(2)*(1+cos(tmp_theta))-sin(tmp_theta)))*dt;

%     tmp_theta = pre_theta(N_Es+1:N_E,1);
%     theta(N_Es+1:N_E,1) = tmp_theta + (-gLE*cos(tmp_theta)+h*(1+cos(tmp_theta)).*0+g_EE(:,t,1).*(q(1)*(1+cos(tmp_theta))- ...
%         sin(tmp_theta))+g_EI(:,t,1).*(q(2)*(1+cos(tmp_theta))-sin(tmp_theta)))*dt;

    % I
    tmp_theta = pre_theta(N_E+1:N,1);
    theta(N_E+1:N,1) = tmp_theta + (-gLI*cos(tmp_theta)+h*(1+cos(tmp_theta)).*I_I(:,1)+([g_IsEs(:,t,1);g_ImEs(:,t,1)]+[g_IsEm(:,t,1);g_ImEm(:,t,1)]).*(q(1)*(1+cos(tmp_theta))- ...
        sin(tmp_theta))+([g_IsIs(:,t,1);g_ImIs(:,t,1)]+[g_IsIm(:,t,1);g_ImIm(:,t,1)]).*(q(2)*(1+cos(tmp_theta))-sin(tmp_theta)))*dt;



    % E
    tmp_theta = pre_theta(1:N_E,2);
    theta(1:N_E,2) = tmp_theta + (-gLE*cos(tmp_theta)+h*(1+cos(tmp_theta)).*I_E(:,2)+([g_EsEs(:,t,2);g_EmEs(:,t,2)]+[g_EsEm(:,t,2);g_EmEm(:,t,2)]).*(q(1)*(1+cos(tmp_theta))- ...
        sin(tmp_theta))+([g_EsIs(:,t,2);g_EmIs(:,t,2)]+[g_EsIm(:,t,2);g_EmIm(:,t,2)]).*(q(2)*(1+cos(tmp_theta))-sin(tmp_theta)))*dt;

    % I
    tmp_theta = pre_theta(N_E+1:N,2);
    theta(N_E+1:N,2) = tmp_theta + (-gLI*cos(tmp_theta)+h*(1+cos(tmp_theta)).*I_I(:,2)+([g_IsEs(:,t,2);g_ImEs(:,t,2)]+[g_IsEm(:,t,2);g_ImEm(:,t,2)]).*(q(1)*(1+cos(tmp_theta))- ...
        sin(tmp_theta))+([g_IsIs(:,t,2);g_ImIs(:,t,2)]+[g_IsIm(:,t,2);g_ImIm(:,t,2)]).*(q(2)*(1+cos(tmp_theta))-sin(tmp_theta)))*dt;

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

%输出 读EmEm
out = [tanh(g_EmEm(:,:,1)')*W_out/N_E tanh(g_EmEm(:,:,2)')*W_out/N_E];

%输出 读ImEm
% out = [tanh(g_ImEm(:,:,1)')*W_out/N_I tanh(g_ImEm(:,:,2)')*W_out/N_I];

%输出 读EmIm
% out = [tanh(g_EmIm(:,:,1)')*W_out/N_E tanh(g_EmIm(:,:,2)')*W_out/N_E];

%输出 读ImIm
% out = [tanh(g_ImIm(:,:,1)')*W_out/N_I tanh(g_ImIm(:,:,2)')*W_out/N_I];

%% Figure
xaxis = dt:dt:T;
    path = '/Users/libin/Pictures/SensoryAndMotor_yes/';
    mkdir(path)
    mkdir([path 'connectivity/'])
    mkdir([path 'svg_go/'])
    mkdir([path 'svg_nogo/'])
    mkdir([path 'output/'])
    mkdir([path 'activity_go/'])
    mkdir([path 'activity_nogo/'])
    mkdir([path 'Data/'])
% 连接矩阵描画
figure
imagesc([[conn_EsEs,conn_EsEm,conn_EsIs,conn_EsIm]; ...
        [conn_EmEs,conn_EmEm,conn_EmIs,conn_EmIm]; ...
        [conn_IsEs,conn_IsEm,conn_IsIs,conn_IsIm]; ...
        [conn_ImEs,conn_ImEm,conn_ImIs,conn_ImIm]])
xlabel('From')
ylabel('To')
colormap('gray');
colorbar
% pbaspect([1,1,1])
% colorbar
title('Connectivity')

    exportgraphics(gcf,[path,'connectivity/',num2str(RS),'_', ...
        num2str(IS),'_' ...
        ,num2str(mu_n),'_', ...
        num2str(si_n),'_', ...
        num2str(mu_m),'_', ...
        num2str(si_m), ...
        '.png'],'Resolution',300)

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
plot(xaxis,rec_V(N_E+1:floor(N_E+N_I*0.1)+1,:,o),'b')
hold off
xlim([0 T])
ylim([-200 100])
title("v " + num2str(o))

% 突触电导描画
g_all = {g_EsEs, g_EsEm, g_EmEs, g_EmEm, g_IsEs, g_IsEm, g_ImEs, g_ImEm, g_EsIs, g_EsIm, g_EmIs, g_EmIm, g_IsIs, g_IsIm, g_ImIs, g_ImIm};
g_name = ["g_{EsEs}","g_{EsEm}","g_{EmEs}","g_{EmEm}","g_{IsEs}","g_{IsEm}","g_{ImEs}","g_{ImEm}","g_{EsIs}","g_{EsIm}","g_{EmIs}","g_{EmIm}","g_{IsIs}","g_{IsIm}","g_{ImIs}","g_{ImIm}"];
subplot(3,1,3)
leg = [];
g_draw = {};
for ind = 1:length(g_all)
    if find(g_all{ind})
        plot(xaxis, mean(g_all{ind}(:,:,o)))
        hold on
        leg = [leg,g_name(ind)];
        legend(leg, 'Location','West')
        g_draw{end+1} = g_all{ind};
    end
end
xlim([0 T])
title("g " + num2str(o))

figure 
for i =1:length(g_draw)
subplot(floor(length(g_draw)/2),2,i)
plot(xaxis, mean(g_draw{i}(:,:,o)))
title(leg(i) + num2str(o))
end

o = o + 1;
        if o == 2
            exportgraphics(gcf,[path,'svg_go/',num2str(RS),'_', ...
                num2str(IS),'_' ...
                ,num2str(mu_n),'_', ...
                num2str(si_n),'_', ...
                num2str(mu_m),'_', ...
                num2str(si_m),'_', ...
                num2str(mu_n),'.png'],'Resolution',300)
        else
            exportgraphics(gcf,[path,'svg_nogo/',num2str(RS),'_', ...
                num2str(IS),'_' ...
                ,num2str(mu_n),'_', ...
                num2str(si_n),'_', ...
                num2str(mu_m),'_', ...
                num2str(si_m),'_', ...
                num2str(mu_n),'.png'],'Resolution',300)
        end
end


%观察输出
figure
hold on
limy = 3;
x = [5 5 15 15];
y = [-limy limy limy -limy];
op(1)=plot(xaxis, out(:,1));
hold on
op(2)=plot(xaxis, out(:,2));
patch(x,y,'blue','FaceAlpha',0.2,'EdgeColor','none')
title('Readout from Excitatory neurons')
ylim([min(min(out))*1.1 max(max(out))*1.1])
legend('O_{go}', 'O_{nogo}')

op(1).LineWidth = 2;
op(1).Color = [0.24 0.35 0.67]; %钴色
op(2).LineWidth = 2;
op(2).Color = [0.01 0.66 0.62];% 锰蓝

    exportgraphics(gcf,[path,'output/',num2str(RS),'_', ...
        num2str(IS),'_', ...
        num2str(mu_n),'_', ...
        num2str(si_n),'_', ...
        num2str(mu_m),'_', ...
        num2str(si_m),...
        '.png'],'Resolution',300)

toc
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
