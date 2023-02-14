% 分成E-I两组地QIF modified theta model
%只考虑兴奋性连接
%改变inputstreng看结果
%IS可以保证connectivity strength和random strength不变的情况下变化
%将连接概率尽可能增大
% 连接是一对一生成的（非平均场）
% Go_noGo Task
% 同时观察两个输入
%用两个垂直的01向量构筑lowrank connectivity
%理解：如果把go输入看成一个pattern，那么越接近go输入，就越能引起神经网络反应，就像hopfield联想记忆模型一样
%这个时候即使全部为正，也可以定义垂直，即在go的pattern的位置上全是零就是完全与go输入一点也不相关
%把连接矩阵看成是连接的概率，在这里本来值就相当小了
%笔记：
%高斯分布随机数 r = normrnd(mu,sigma,sz)
%gamma分布（指数分布随机数）（gamma（1,λ）=exp(λ)）r = exprnd(mu,sz1,...,szN) mu为均值
%真正的gamma分布 r = gamrnd(a,b,sz1,...,szN)
%对数正态分布 r = lognrnd(mu,sigma,sz1,...,szN)
tic
clear

%% Setting
N_E = 1000; % 神经元数(E)
N_I = 0; % 神经元数(I)
N = N_E+N_I; % 神经元数(Total)
dt = 0.01; % 运算步长(ms/step)
T = 30; % 模拟时间总长度(ms)
tt = T/dt; % 计算步数
theta = ones(N,2)*V2theta(-70); % 位相初始值
gLE = 0.08; % 漏电导（E）
gLI = 0.1; % 漏电导（I）
g_p = [0.004069, 0.02672, 0.003276, 0.02138]; %突触电导的权重(E←E, E←I, I←E, I←I)
V_T = -55; % 点火阈值(mV)
V_R = -62; % 静息电位
V_E = 0; % 兴奋性突触反转电位
V_I = -70; % 抑制性突触反转电位

RS = 1; %RandomStrength

IS = 3;% Input Strength,Sti_go=IS*n

h = 2/(V_T-V_R);
q = [(2*V_E-V_R-V_T)/(V_T-V_R), (2*V_I-V_R-V_T)/(V_T-V_R)];
tau_dE = 2; % 衰减时间常数（E）(ms)
tau_dI = 5; % 衰减时间常数（I）(ms)

%n,m的参数
for si = [20:100]
    mu_n = 50;
    si_n = si;

    b_n = si_n^2/mu_n;%gamma分布
    a_n = mu_n/b_n;

    mu_m = 50;
    si_m = si;

    b_m = si_m^2/mu_m;%gamma分布
    a_m = mu_m/b_m;


    % conn_EI = RS*gamrnd(a, b, N_E,N_I)*(1/N_I); % 连接矩阵(E←I)
    conn_EI = RS*rand([N_E,N_I])*(1/N_I); % 连接矩阵(E←I)
    % conn_IE = RS*gamrnd(a, b, N_I,N_E)*(1/N_E); % 连接矩阵(I←E)
    conn_IE = RS*rand([N_I,N_E])*(1/N_E); % 连接矩阵(I←E)
    % conn_II = RS*gamrnd(a, b, N_I,N_I)*(1/N_I); % 连接矩阵(I←I)
    conn_II = RS*rand([N_I,N_I])*(1/N_I); % 连接矩阵(I←I)


    % lowrank给EE，读EE
    %Sti_go,nogo,W_out三个向量要保证互相垂直同时还非负，那么就只能让有值的位置互相不重合
    %随机创建一个N维向量，对其中小于1/3的位置给sti_go，即不小于1/3的全部变成零，其余同理
    %改进点：其中的值要改成高斯分布的，好像也没有必要，只要不在一个位置上有重叠，任何值都无所谓
    %改进点：原文献中sti的数值取自均值为零，标准差为2的高斯分布
    %改进点：P的量级调整应该更加自然一点
    %改进：完全将位置的选择变为随机，以便观察情况？
    temp = rand([1,N_E]);

    Sti_go = temp;
    Sti_go(Sti_go>=1/3) = 0;%n = Sti_go
    % Sti_go(Sti_go~=0) = normrnd(mu_n,si_n,1,length(find(Sti_go~=0)));
    Sti_go(Sti_go~=0) = gamrnd(a_n,b_n,1,length(find(Sti_go~=0)));
    Sti_go(Sti_go<0) = 0;


    Sti_nogo = temp;
    Sti_nogo(Sti_nogo>=2/3 | Sti_nogo<1/3) = 0;
    % Sti_nogo(Sti_nogo~=0) = normrnd(mu_n,si_n,1,length(find(Sti_nogo~=0)));
    Sti_nogo(Sti_nogo~=0) = gamrnd(a_n,b_n,1,length(find(Sti_nogo~=0)));
    Sti_nogo(Sti_nogo<0) = 0;


    W_out = temp;
    W_out(W_out<2/3) = 0;%m = Wout,可以通过调整范围决定两个向量重合的程度
    % W_out(W_out~=0) = normrnd(mu_m,si_m,1,length(find(W_out~=0)));
    W_out(W_out~=0) = gamrnd(a_m,b_m,1,length(find(W_out~=0)));
    W_out = W_out';
    W_out(W_out<0) = 0;

    P = (W_out*Sti_go)/N_E;%Structre connectivity matrix


    %随机矩阵使用高斯分布取绝对值，因为它均值一直是0，标准差为1/根号N，只需要变random strength即可
    % W = RS*rand([N_E,N_E])*(1/N_E);
    W = RS*normrnd(0,sqrt(1/N_E),N_E,N_E);
    W = abs(W);

    conn = W + P; %连接矩阵(N*N),随机加lowRank, con = RS*X+P, P=W_out*Sti_go/N
%将connectivity matrix限制在0到1

conn_EE = conn;
conn_EE(conn_EE>1)=1;

    rec_V = zeros(N,tt,2); % 记录膜电位
    A = zeros(N,1,2); % 点火的记录

    % g_EE = randn(N_E,tt,2)/10; % 突触电导(E←E)
    g_EE = zeros(N_E,tt,2); % 突触电导(E←I)
    g_EI = zeros(N_E,tt,2); % 突触电导(E←I)
    g_IE = zeros(N_I,tt,2); % 突触电导(I←E)
    g_II = zeros(N_I,tt,2); % 突触电导(I←I)

    % g_EE(:,:,1) = abs(normrnd(0,1,N_E,tt)); %g的初始值

    % 点火的记录(最后画图用)

    firings_inh_1 = [];
    firings_exc_1 = [];

    firings_inh_2 = [];
    firings_exc_2 = [];



    Sti_go = Sti_go*IS;
    Sti_nogo = Sti_nogo*IS;

    I_E = [Sti_go(1:N_E)' Sti_nogo(1:N_E)']; % 输入电流(E)（nA）
    I_I = [0 0]; % 输入电流(I)


    %% Main
    for t = 2:tt
        I_E = [0 0];
        if 5 < t/100 &&  t/100 <15
            I_E = [Sti_go(1:N_E)' Sti_nogo(1:N_E)']; % 输入电流(E)
        end

        if mod(t,tt/10)==0
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
    path = '/Users/libin/Pictures/change_si_both/';
    xaxis = dt:dt:T;

    % 连接矩阵描画,因为抑制性神经元少，所以公式得到的抑制性连接强
    figure
    imagesc([[conn_EE,conn_EI];[conn_IE,conn_II]])
    xlabel('From')
    ylabel('To')
    colormap('gray');
    colorbar
    % clim([0 0.001]);
    %pbaspect([1,1,1])
    title('Connectivity')
    exportgraphics(gcf,[path,'connectivity/',num2str(RS),'_', ...
        num2str(IS),'_' ...
        ,num2str(mu_n),'_', ...
        num2str(si_n),'_', ...
        num2str(mu_m),'_', ...
        num2str(si_m), ...
        '.png'],'Resolution',300)

    % disp(['Random Strength: ' num2str(RS)])
    % disp(['Connectivity Strength: ' num2str(Sti_go*W_out/N_E)])

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

        % figure
        % for i = 1:5
        % plot(xaxis, g_EE(i,:,o))
        % hold on
        % end

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

    % %观察输出
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

    % %分开看两个输出
    % figure
    % limy = 3;
    % x = [5 5 15 15];
    % y = [-limy limy limy -limy];
    % op(1)=plot(xaxis, out(:,1));
    % patch(x,y,'blue','FaceAlpha',0.2,'EdgeColor','none')
    % title('Readout from Excitatory neurons')
    % ylim([min(min(out))*1.1 max(max(out))*1.1])
    % y_go=ylim;
    % legend('O_{go}')
    %
    % figure
    % op(2)=plot(xaxis, out(:,2));
    % patch(x,y,'blue','FaceAlpha',0.2,'EdgeColor','none')
    % ylim(y_go)
    % title('Readout from Excitatory neurons')
    %
    % legend('O_{nogo}')

    op(1).LineWidth = 2;
    op(1).Color = [0.24 0.35 0.67]; %钴色
    op(2).LineWidth = 2;
    op(2).Color = [0.01 0.66 0.62];% 锰蓝

    %观察平均活动
    on_m_go = W_out'*g_EE(:,:,1);
    on_m_nogo = W_out'*g_EE(:,:,2);

    on_Igo = Sti_go*g_EE(:,:,1);
    on_Inogo = Sti_nogo*g_EE(:,:,2);

    figure
    Activity(1) = plot(on_Igo,on_m_go);
    xlabel('Sti_{go}')
    ylabel('m')
    Activity(1).LineWidth = 2;
    Activity(1).Color = [0.24 0.35 0.67];%钴色
    yl = ylim;
    xl = xlim;
    exportgraphics(gcf,[path,'activity_go/',num2str(RS),'_', ...
        num2str(IS),'_' ...
        ,num2str(mu_n),'_', ...
        num2str(si_n),'_', ...
        num2str(mu_m),'_', ...
        num2str(si_m), ...
        '.png'],'Resolution',300)

    % hold on
    figure
    Activity(2) = plot(on_Inogo,on_m_nogo);
    xlabel('Sti_{nogo}')
    ylabel('m')
    Activity(2).LineWidth = 2;
    Activity(2).Color = [0.01 0.66 0.62];% 锰蓝
    ylim(yl)
    xlim(xl)
    exportgraphics(gcf,[path,'activity_nogo/',num2str(RS),'_', ...
        num2str(IS),'_' ...
        ,num2str(mu_n),'_', ...
        num2str(si_n),'_', ...
        num2str(mu_m),'_', ...
        num2str(si_m), ...
        '.png'],'Resolution',300)

    close all

    toc
        save([path,'change_si_both/Data/',num2str(RS),'_', ...
        num2str(IS),'_' ...
        ,num2str(mu_n),'_', ...
        num2str(si_n),'_', ...
        num2str(mu_m),'_', ...
        num2str(si_m)])
end
%%
% W2 = gamrnd(a,b,N_E,1)/N_E;
% out2 = [tanh(g_EE(:,:,1)')*W2 tanh(g_EE(:,:,2)')*W2];
% figure
% plot(xaxis, out2(:,1))
% hold on
% plot(xaxis, out2(:,2))

% for o=1
% figure
% for i = 5:15
% plot(xaxis,g_EE(i,:,o))
% hold on
% end
% title("g_{rec}"+num2str(o))
% % xlim([20 30])
% end

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