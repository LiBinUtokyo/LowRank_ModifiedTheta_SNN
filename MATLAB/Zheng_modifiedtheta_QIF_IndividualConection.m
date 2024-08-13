
clc;
clear;
initime = clock;
for repeat=0:0
    %% Parameters
    sss=1; % loop times
    local=1; % no use
    %0 for cut

    Ne_1=3000; %I % 1000
    % Ne_2=10000; %E
    meanrate = [];
    inout=[];
    dt = 0.01; %0.005 0.0025
    %a_master=2 %+ (floor(repeat/10)-5)*0.1;%1212
    %a_I=a_master;
    a_E=2;%2 % no use
    num=200000;%120000 % total time 1/100000s

    gmax = 1.6; % gsyn axis range

    eta_zero_N = 5; %2 % average input current
    del_N = 0.04; %0.1 % larger range
    psyn1000 = 0.1;
    I_V_syn = -53;
    sig = 0; %0  ;%sqrt(2*D);% noise
    V_T = -54; % firing threshold

    psyn = psyn1000/(Ne_1/1000);

    place_filename = strjoin(["C:\zheng\mywork\cumulant\data\xpp\commonly\eta2vsyn\changing_vt\", ...
        "eta",num2str(eta_zero_N),"_vt",num2str(V_T),"_vsyn",num2str(I_V_syn),"_psyn",num2str(psyn1000),"_del",num2str(del_N), ...
        "_sig",num2str(sig),"_N",num2str(Ne_1),"_time",num2str(num)], '');


    threshold=0.15;
    u_end=0;
    theta=pi; % threshold of spike or not spike
    V_R=-62;
    E_V_syn=0;
    c_1=2/(V_T-V_R);

    I_g_L=0.1;
    E_g_L=0.08;%0.05 no use

    II_c_2=(2*I_V_syn-V_T-V_R)/(V_T-V_R); % c2
    IE_c_2=(2*E_V_syn-V_T-V_R)/(V_T-V_R);


    t_1 = 0; %delay step to transmit
    II_t_d= 5; % tau=5ms
    II_t_r=0.5; %tau for inhibit

    II_gbar = 0.138062206; % GABA on interneuron
    II_rate = psyn * Ne_1; %50 % expectation connection number p*N
    II_c_3e=-1/II_t_d; % -1/tao


    a_o_n = (   2.9*10^(-4)  )  *10^6  ;% area of neuron [cm^2 * 10^6]
    gsyn_peak_SN(1,1) = 6.2; % g peak
    II_c_5e = gsyn_peak_SN(1,1)/a_o_n *II_rate; % mu=gpeak*p*N



    %% Input current
    K = -0.0; %-0.3 % no use
    y_temp = transpose( (1:Ne_1)/(Ne_1+1));

    % Normal Lorentz
    eta_temp = del_N * tan (  pi * (y_temp - 1/2)  ) + eta_zero_N ;

    for i=1:Ne_1
        for j=1:Ne_1
            tmp = rand(1);
            if tmp > -5; %0.9;
                S(i,j)=1;
            else S(i,j)=0;
            end
        end
    end

    %% Initial values
    v_1=2*pi*rand(Ne_1,1)-pi;    % Initial values of v % uniform distribution

    %% Container preparation 中间变量
    EE_G= zeros(2,num);
    EI_G= zeros(2,num);
    IE_G= zeros(2,num);
    II_G= zeros(2,num+2);
    A_1 = zeros(num,1);
    out1 = zeros(num,1);

    Nave=500;
    in = zeros(num,1); % no use
    c=0.0; % no use

    I2_1=zeros(num,1); % no use

    kk=1;
    tmp_pre=0;

    %% Simulation
    for k=1:num    % count for all neurons

        I_common= randn(1,1); % only for noisesyn % no use
        I_1= sig*randn(Ne_1,1); % input with noise
        In(k) = 0.0;

        if k>0
            fired_1=find(v_1>=theta);   % indices of spikes
            A_1(k)=size(fired_1,1)/dt / Ne_1;
            v_1(fired_1)=v_1(fired_1)-2*pi;
        end
        if k > t_1
            II_G(1,k+1)= II_G(1,k)+ II_c_3e*dt*II_G(1,k)+II_c_5e*A_1(k-t_1)*dt;%flux no need eta 1st

        end

        v_1=v_1+(-I_g_L * cos(v_1) + c_1 * (1+cos(v_1)).*(eta_temp - c_1*sig*sig/2*sin(v_1)) +II_G(1,k)*(II_c_2 * (1+cos(v_1))-sin(v_1)) +IE_G(1,k)*(IE_c_2 * (1+cos(v_1))-sin(v_1)))*dt + c_1 * (1+cos(v_1)).* I_1 * sqrt(dt);
        % eta

        strage(sss)=v_1(900);
        sss=sss+1;

        if(k==20000) real_threshold = mean(II_G(1,1:20000));
        end
        if(k>=20000)
            if(II_G(1,k)>threshold)&&(II_G(1,k)<=II_G(1,k-1))&&(II_G(1,k+1)<II_G(1,k-1))&&(II_G(1,k-1)>II_G(1,k-2))&&(II_G(1,k-1)>II_G(1,k-3))&&(k>tmp_pre+1000)

                u_end(kk)=k;
                tmp_pre=k;
                kk=kk+1;

            end
        end

    end

    %% Plot
    plot(+II_G(1,:));
    axis([0 num 0 gmax]);
    xlabel('time(steps)');
    ylabel('g_{syn}');

    saveas(gcf, strjoin([place_filename,".png"],''),'png');
    save(strjoin([place_filename,".mat"], ''))
end
%end
fintime = clock;
alltime = etime(fintime, initime)