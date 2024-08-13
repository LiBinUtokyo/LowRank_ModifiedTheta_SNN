%画分辨率对分布期望值mu的依赖
%得到包含所有文件名的结构体
path = '/Users/libin/Desktop/change_mu_both/Data/';
data_all = dir(fullfile(path,'*.mat'));

mu_rec = zeros(1,length(data_all));
go_to_nogo = zeros(1,length(data_all));

%将数据中的output提取出来，
% 将对应的mu存在mu_rec中
% 计算go的最大值除以nogo的最大值的结果，存在go_to_nogo中

for i =1:length(data_all)
data_now = load(fullfile(path,data_all(i).name),'mu_n','out');
mu_rec(i) = data_now.mu_n/1000;
go_to_nogo(i) = max(data_now.out(:,1))/max(data_now.out(:,2)) ;
end

figure
% plot(mu_rec,go_to_nogo)
scatter(mu_rec,go_to_nogo,50,"filled");
xlabel('Mathematic Expectation \mu of Structure Connectivity Pij')
ylabel('Maximun of output_{go} to output_{nogo}')
%%
%画分辨率对分布标准差si的依赖
%得到包含所有文件名的结构体
path = '/Users/libin/Desktop/change_si_both_new/Data/';
data_all = dir(fullfile(path,'*.mat'));

si_rec = zeros(1,length(data_all));
go_to_nogo = zeros(1,length(data_all));

%将数据中的output提取出来，
% 将对应的si存在si_rec中
% 计算go的最大值除以nogo的最大值的结果，存在go_to_nogo中

for i =1:length(data_all)
data_now = load(fullfile(path,data_all(i).name),'si_n','out');
si_rec(i) = data_now.si_n/sqrt(2);
go_to_nogo(i) = max(data_now.out(:,1))/max(data_now.out(:,2)) ;
end

figure
% plot(si_rec,go_to_nogo)
scatter(si_rec,go_to_nogo,50,"filled");
xlabel('Standerd Deviation of Structure Connectivity Pij')
ylabel('Maximun of output_{go} to output_{nogo}')

%%
%画分辨率对分布随机强度RS的依赖
%得到包含所有文件名的结构体
path = '/Users/libin/Pictures/change_random_strength/Data';
data_all = dir(fullfile(path,'*.mat'));

RS_rec = zeros(1,length(data_all));
go_to_nogo = zeros(1,length(data_all));

%将数据中的output提取出来，
% 将对应的si存在si_rec中
% 计算go的最大值除以nogo的最大值的结果，存在go_to_nogo中

for i =1:length(data_all)
data_now = load(fullfile(path,data_all(i).name),'RS','out');
RS_rec(i) = data_now.RS;
go_to_nogo(i) = max(data_now.out(:,1))/max(data_now.out(:,2)) ;
end

figure
% plot(si_rec,go_to_nogo)
scatter(RS_rec,go_to_nogo,50,"filled");
xlabel('Random Strength')
ylabel('Maximun of output_{go} to output_{nogo}')


