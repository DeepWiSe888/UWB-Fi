%% main for post process
% This is the post-processing script.
% This script should be executed after the completion of the SpecTrans (PyTorch model)
% to differentiate between subjects.
clear
clc
close all
%% load data
% you can also change you own path!!!!!!!!
script_path = mfilename('fullpath');
folder_path = fileparts(script_path);
gt_path = [folder_path '/dataset_mat/GroundTruth/gt_xy.mat'];
load(gt_path)
pre_path = [folder_path '/SpecTrans_result/pre_result.mat'];
load(pre_path)
MUSIC_spectrum = squeeze(pre_result);
%%  set parameters
theta = 20:0.5:160;                          % AOA range (degree)
tau = 1e-8:2.5e-10:4e-8;                     % TOF range (second)
exit_fig = 1;                                % plot or not
WLAN_paras.num_antenna = 2;                  % number of antennas
WLAN_paras.num_subcarrier = 57*1;            % number of subcarriers
WLAN_paras.frequency_space = 312.5e3;        % Hz
WLAN_paras.num_path = 3;                     % number of paths
xr = 7;                                      % distance between Tx and Rx
WLAN_paras.speed_light = 299792458;          % ligth speed
%% find peaks
res = [];
for i = 1:size(MUSIC_spectrum,1)
    input_data = squeeze(MUSIC_spectrum(i,:,:))';
    [path_info_outputL,ind,max_N_valueL] = MUSIC_peaks(input_data, ...
        10,theta,tau);
    [~,powL] = sort(max_N_valueL,'descend');
    path_info_outputL = path_info_outputL(powL,:);
    [~,val] = min(ind(:,2));
    MUSIC_spectrum_clean = input_data;
    [~,LOS_theta] = min(abs(theta - 90));
    [~,LOS_tau]=min(abs(tau-1.13e-8));
    MUSIC_spectrum_clean(:,1:LOS_tau) = -inf;
    MUSIC_spectrum_clean(LOS_theta,:) = -inf;
    [path_info_output,ind,max_N_value] = MUSIC_peaks(MUSIC_spectrum_clean(:,:,1),3,theta,tau);
    [~,pow] = sort(max_N_value,'descend');
    path_info_output = path_info_output(pow,:);
    % Result: Each result spans two lines: 
    % the first line represents the AoA, 
    % while the second line indicates the ToF.
    res = [res;[path_info_outputL',path_info_output']];
    path(i,:) = valData(path_info_outputL);
    downstream_data(:,:,i) = downstream(input_data,path_info_output(1,:),tau,theta);  % extract data for downstream task
end
% This is the sbuject AoA and ToF
subject = path(:,3:4);
for i = 1:size(subject,1)
    [x,y]=com_xy(subject(i,1),subject(i,2),xr);
    subject_xy(i,:) = [x,y];
end
set_num = min(round(size(subject_xy,1)/30),size(gt_xy,1));
for i = 1:set_num
    errors((i-1)*30+1:i*30,1) = ((subject_xy((i-1)*30+1:i*30,1)-gt_xy(i,1)).^2 ...
        + (subject_xy((i-1)*30+1:i*30,2)-gt_xy(i,2)).^2).^(0.5);
end
fprintf('<============== The mean location error is %.2fm ==============>\n', mean(errors))
fprintf('<============== The median location error is %.2fm ==============>\n', median(errors))
%% plot the MUSIC spectrum
if exit_fig == 1
    set(0,'defaultfigurecolor','w');
    figure;
    fig = gcf;
    fig.Position = [200 200 800 600];
    imagesc(tau*(1e9),theta,squeeze(MUSIC_spectrum(end,:,:))'/max(max(squeeze(MUSIC_spectrum(end,:,:))')));
    xlabel('ToF [ns]', 'fontsize', 18);
    ylabel('AoA [°]', 'fontsize', 18);
    set(gca,'YDir','normal')
    set(gca,'FontSize',18)
    set(gca,'FontName','Times')
    box on
    colorbar
    % set(gca, 'LooseInset', [0,0.01,0,0.01]);
end
fprintf('Finish===========>\n')
