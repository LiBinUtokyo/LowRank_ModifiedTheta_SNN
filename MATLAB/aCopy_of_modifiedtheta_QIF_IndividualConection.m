% E-I�W�c��QIF modified theta���f��
% �����͈�Έ�Ō`���i���Ϗ�ł͂Ȃ��j
% �����d���̓��[�����c���z�ɏ]��


clear

%% Setting
N_E = 400; % �j���[������(E)
N_I = 100; % �j���[������(I)
N = N_E+N_I; % �j���[������(Total)
dt = 0.01; % ���ݕ�(ms/step)
length = 500; % �v�Z����(ms)
tt = length/dt; % �v�Z�X�e�b�v��
theta = ones(N,1)*atan((-70-117/2)*2/7)*2; % �ʑ������l
gLE = 0.08; % �R��R���_�N�^���X
gLI = 0.1; % �R��R���_�N�^���X
g_p = [0.004069, 0.02672, 0.003276, 0.02138]; %�V�i�v�X�R���_�N�^���X�̋���(E��E, E��I, I��E, I��I)
V_T = -55; % ����臒l
V_R = -62; % �Î~���d��
V_E = 0; % �������V�i�v�X�t�d��
V_I = -70; % �}�����V�i�v�X�t�d��
h = 2/(V_T-V_R);
q = [(2*V_E-V_R-V_T)/(V_T-V_R), (2*V_I-V_R-V_T)/(V_T-V_R)];
tau_dE = 2; % ���������莞�萔
tau_dI = 5; % ���������莞�萔
cp = [0.1, 0.1, 0.05, 0.2]; % �����m��(E��E, E��I, I��E, I��I)
conn_EE = rand(N_E,N_E)<cp(1); % �����̏d�ݍs��(E��E)
conn_EI = rand(N_E,N_I)<cp(2); % �����̏d�ݍs��(E��I)
conn_IE = rand(N_I,N_E)<cp(3); % �����̏d�ݍs��(I��E)
conn_II = rand(N_I,N_I)<cp(4); % �����̏d�ݍs��(I��I)

Eta = [0, 10]; % ���d���̕��ϒl
Delta = [0, 0.05]; % ���d����HMHW

rec_V = zeros(N,tt); % �d�ʂ��L�^
A = zeros(N,1); % ���΂��L�^
g_EE = zeros(N_E,tt); % �V�i�v�X�R���_�N�^���X(E��E)
g_EI = zeros(N_E,tt); % �V�i�v�X�R���_�N�^���X(E��I)
g_IE = zeros(N_I,tt); % �V�i�v�X�R���_�N�^���X(I��E)
g_II = zeros(N_I,tt); % �V�i�v�X�R���_�N�^���X(I��I)

firings = []; % ���΂��L�^(���X�^�[�v���b�g�p)

I_E = Eta(1)+Delta(1)*tan(pi*((1:N_E)'/(N_E+1)-1/2)); % �����d��(E)
I_I = Eta(2)+Delta(2)*tan(pi*((1:N_I)'/(N_I+1)-1/2)); % �����d��(I)


%% Main
for t = 2:tt
%     if mod(t,tt/100)==0
%         disp([num2str(t/100),'ms'])
%     end
    % �R���_�N�^���X�v�Z
    g_EE(:,t) = g_EE(:,t-1) + (-g_EE(:,t-1)/tau_dE+g_p(1)*conn_EE*A(1:N_E))*dt;
    g_EI(:,t) = g_EI(:,t-1) + (-g_EI(:,t-1)/tau_dI+g_p(2)*conn_EI*A(N_E+1:N))*dt;
    g_IE(:,t) = g_IE(:,t-1) + (-g_IE(:,t-1)/tau_dE+g_p(3)*conn_IE*A(1:N_E))*dt;
    g_II(:,t) = g_II(:,t-1) + (-g_II(:,t-1)/tau_dI+g_p(4)*conn_II*A(N_E+1:N))*dt;
    % �ʑ�(���d��)�v�Z
    pre_theta = theta;
    % E
    tmp_theta = pre_theta(1:N_E);
    theta(1:N_E) = tmp_theta + (-gLI*cos(tmp_theta)+h*(1+cos(tmp_theta)).*I_E+(g_EE(t)+g_EE(t))*(q(1)*(1+cos(tmp_theta))- ...
        sin(tmp_theta))+(g_EI(t)+g_EI(t))*(q(2)*(1+cos(tmp_theta))-sin(tmp_theta)))*dt;
    % I
    tmp_theta = pre_theta(N_E+1:N);
    theta(N_E+1:N) = tmp_theta + (-gLI*cos(tmp_theta)+h*(1+cos(tmp_theta)).*I_I+(g_IE(t)+g_IE(t))*(q(1)*(1+cos(tmp_theta))- ...
        sin(tmp_theta))+(g_II(t)+g_II(t))*(q(2)*(1+cos(tmp_theta))-sin(tmp_theta)))*dt;
    % ���d�ʕۑ�
    rec_V(:,t) = (V_T+V_R)/2+(V_T-V_R)/2*tan(theta/2);
    % ���΂����j���[�����̏���
    A = (theta >= pi);
    theta(theta >= pi) = theta(theta >= pi)-2*pi;
    % ���Ώ��ۑ�
    firings = [firings;t+0*find(A),find(A)];
end

%% Figure
xaxis = dt:dt:length;
figure(1)
% ���Ώ��`��
subplot(3,1,1)
scatter(firings(:,1)*dt,firings(:,2),2,'filled')
xlim([0 length])
title("r")

% ���d�ʕ`��(�ꕔ��`��)
subplot(3,1,2)
plot(xaxis,rec_V(1:floor(N_E*0.1)+1,:),'r')
hold on
plot(xaxis,rec_V(N_E+1:floor(N_E+N_I*0.1)+1,:),'b')
hold off
xlim([0 length])
ylim([-200 100])
title("v")

% �R���_�N�^���X�`��
subplot(3,1,3)
plot(xaxis, mean(g_EE))
hold on
plot(xaxis, mean(g_EI))
plot(xaxis, mean(g_IE))
plot(xaxis, mean(g_II))
hold off
xlim([0 length])
title("g")

% �����̏d�ݍs��`��
figure(2)
imagesc([[conn_EE,conn_EI];[conn_IE,conn_II]])
xlabel('From')
ylabel('To')
colormap('gray');
pbaspect([1,1,1])