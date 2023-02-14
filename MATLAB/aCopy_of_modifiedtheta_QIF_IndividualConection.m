% E-I集団のQIF modified thetaモデル
% 結合は一対一で形成（平均場ではない）
% 流入電流はローレンツ分布に従う


clear

%% Setting
N_E = 400; % ニューロン数(E)
N_I = 100; % ニューロン数(I)
N = N_E+N_I; % ニューロン数(Total)
dt = 0.01; % 刻み幅(ms/step)
length = 500; % 計算時間(ms)
tt = length/dt; % 計算ステップ数
theta = ones(N,1)*atan((-70-117/2)*2/7)*2; % 位相初期値
gLE = 0.08; % 漏れコンダクタンス
gLI = 0.1; % 漏れコンダクタンス
g_p = [0.004069, 0.02672, 0.003276, 0.02138]; %シナプスコンダクタンスの強さ(E←E, E←I, I←E, I←I)
V_T = -55; % 発火閾値
V_R = -62; % 静止膜電位
V_E = 0; % 興奮性シナプス逆電位
V_I = -70; % 抑制性シナプス逆電位
h = 2/(V_T-V_R);
q = [(2*V_E-V_R-V_T)/(V_T-V_R), (2*V_I-V_R-V_T)/(V_T-V_R)];
tau_dE = 2; % 立ち下がり時定数
tau_dI = 5; % 立ち下がり時定数
cp = [0.1, 0.1, 0.05, 0.2]; % 結合確率(E←E, E←I, I←E, I←I)
conn_EE = rand(N_E,N_E)<cp(1); % 結合の重み行列(E←E)
conn_EI = rand(N_E,N_I)<cp(2); % 結合の重み行列(E←I)
conn_IE = rand(N_I,N_E)<cp(3); % 結合の重み行列(I←E)
conn_II = rand(N_I,N_I)<cp(4); % 結合の重み行列(I←I)

Eta = [0, 10]; % 定常電流の平均値
Delta = [0, 0.05]; % 定常電流のHMHW

rec_V = zeros(N,tt); % 電位を記録
A = zeros(N,1); % 発火を記録
g_EE = zeros(N_E,tt); % シナプスコンダクタンス(E←E)
g_EI = zeros(N_E,tt); % シナプスコンダクタンス(E←I)
g_IE = zeros(N_I,tt); % シナプスコンダクタンス(I←E)
g_II = zeros(N_I,tt); % シナプスコンダクタンス(I←I)

firings = []; % 発火を記録(ラスタープロット用)

I_E = Eta(1)+Delta(1)*tan(pi*((1:N_E)'/(N_E+1)-1/2)); % 流入電流(E)
I_I = Eta(2)+Delta(2)*tan(pi*((1:N_I)'/(N_I+1)-1/2)); % 流入電流(I)


%% Main
for t = 2:tt
%     if mod(t,tt/100)==0
%         disp([num2str(t/100),'ms'])
%     end
    % コンダクタンス計算
    g_EE(:,t) = g_EE(:,t-1) + (-g_EE(:,t-1)/tau_dE+g_p(1)*conn_EE*A(1:N_E))*dt;
    g_EI(:,t) = g_EI(:,t-1) + (-g_EI(:,t-1)/tau_dI+g_p(2)*conn_EI*A(N_E+1:N))*dt;
    g_IE(:,t) = g_IE(:,t-1) + (-g_IE(:,t-1)/tau_dE+g_p(3)*conn_IE*A(1:N_E))*dt;
    g_II(:,t) = g_II(:,t-1) + (-g_II(:,t-1)/tau_dI+g_p(4)*conn_II*A(N_E+1:N))*dt;
    % 位相(膜電位)計算
    pre_theta = theta;
    % E
    tmp_theta = pre_theta(1:N_E);
    theta(1:N_E) = tmp_theta + (-gLI*cos(tmp_theta)+h*(1+cos(tmp_theta)).*I_E+(g_EE(t)+g_EE(t))*(q(1)*(1+cos(tmp_theta))- ...
        sin(tmp_theta))+(g_EI(t)+g_EI(t))*(q(2)*(1+cos(tmp_theta))-sin(tmp_theta)))*dt;
    % I
    tmp_theta = pre_theta(N_E+1:N);
    theta(N_E+1:N) = tmp_theta + (-gLI*cos(tmp_theta)+h*(1+cos(tmp_theta)).*I_I+(g_IE(t)+g_IE(t))*(q(1)*(1+cos(tmp_theta))- ...
        sin(tmp_theta))+(g_II(t)+g_II(t))*(q(2)*(1+cos(tmp_theta))-sin(tmp_theta)))*dt;
    % 膜電位保存
    rec_V(:,t) = (V_T+V_R)/2+(V_T-V_R)/2*tan(theta/2);
    % 発火したニューロンの処理
    A = (theta >= pi);
    theta(theta >= pi) = theta(theta >= pi)-2*pi;
    % 発火情報保存
    firings = [firings;t+0*find(A),find(A)];
end

%% Figure
xaxis = dt:dt:length;
figure(1)
% 発火情報描画
subplot(3,1,1)
scatter(firings(:,1)*dt,firings(:,2),2,'filled')
xlim([0 length])
title("r")

% 膜電位描画(一部を描画)
subplot(3,1,2)
plot(xaxis,rec_V(1:floor(N_E*0.1)+1,:),'r')
hold on
plot(xaxis,rec_V(N_E+1:floor(N_E+N_I*0.1)+1,:),'b')
hold off
xlim([0 length])
ylim([-200 100])
title("v")

% コンダクタンス描画
subplot(3,1,3)
plot(xaxis, mean(g_EE))
hold on
plot(xaxis, mean(g_EI))
plot(xaxis, mean(g_IE))
plot(xaxis, mean(g_II))
hold off
xlim([0 length])
title("g")

% 結合の重み行列描画
figure(2)
imagesc([[conn_EE,conn_EI];[conn_IE,conn_II]])
xlabel('From')
ylabel('To')
colormap('gray');
pbaspect([1,1,1])