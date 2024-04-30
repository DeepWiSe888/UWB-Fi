%% generate the MUSIC spectrum
function MUSIC_spectrum = gen_MUSIC_spectum(gt)
%% set the WLAN parameters
    En = 1;                                              % enhance the MUSIC spectrum 
    path_num = 10;                                       % path number
    LOS = [90,2.3349e-08];                               % LOS
    path_info = [LOS;gt];                                % path infomation
    theta = 20:0.5:160;                                  % AOA range (degree)
    tau = 1e-8:2.5e-10:4e-8;                             % TOF range (second)
    WLAN_paras.num_antenna = 2;                          % number of antennas
    WLAN_paras.num_subcarrier = 57*1;                    % number of subcarriers
    WLAN_paras.frequency_space = 312.5 * 10^3;           % Hz
    WLAN_paras.num_path = size(path_info,1);             % number of paths
    WLAN_paras.antenna_space = 0.32;                     % antenna distance (m)
    WLAN_paras.speed_light = 299792458;                  % light speed
    WLAN_paras.has_noise = 0;                            % add noise
    WLAN_paras.SNR = 10;                                 % SNR
    % gain
    WLAN_paras.path_complex_gain = complex(zeros(1,WLAN_paras.num_path));
    WLAN_paras.path_complex_gain(1,1) = 3 + 3i; 
    WLAN_paras.path_complex_gain(1,2) = 3 + 3i;
    for t = 3:WLAN_paras.num_path
        WLAN_paras.path_complex_gain(1,t) = 3 + 3i;
    end
    MUSIC_spectrum = zeros(length(theta),length(tau));
    % center frequency of the ideal CSI
    CF = [2412,2437,2462,5180,5620,5640,5660,5680,5745,...
        5765,5805,5955,5995,6035,6075,6095,6115,6155,6235,...
        6255,6275,6355,6375,6415,6435,6475,6535,6555,6595,...
        6615,6635,6655,6675,6695,6715,6735,6755,6775,6795,...
        6815,6835,6855,6895,6915,6955,6975,6995,7015,7035,7055,7075,7095];
 %% generate   
    for i = 1:1:52
        WLAN_paras.frequency = CF(1,i) * 10^6;
        % packet number of every channel, noly need 1.
        for j = 1:1          
            % generate ideal CSI
            CSI = generate_ideal_CSI_data(WLAN_paras, path_info);
            % generate the MUSIC spectrum
            tmp_spectrum = music_spectrum(CSI,WLAN_paras,theta,tau);
            MUSIC_spectrum = tmp_spectrum + MUSIC_spectrum;
        end
        CSIset(i,:,1) = real(CSI(1,:));
        CSIset(i,:,2) = imag(CSI(1,:));
        CSIset(i,:,3) = real(CSI(2,:));
        CSIset(i,:,4) = imag(CSI(2,:));  
    end
    if En == 1
        if size(gt,1) >1
            MUSIC_spectrum = pro_spe_v2(MUSIC_spectrum./52,path_info,theta,tau,1,0.9);
        else
            MUSIC_spectrum = pro_spe(MUSIC_spectrum./52,path_info,theta,tau,1);
    end
end



