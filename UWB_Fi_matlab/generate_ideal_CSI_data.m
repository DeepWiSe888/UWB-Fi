%% generate ideal CSI data
function CSI = generate_ideal_CSI_data(WLAN_paras, path_info)
    CSI = complex(zeros(WLAN_paras.num_antenna,WLAN_paras.num_subcarrier));
    antenna_space = WLAN_paras.antenna_space;
    for k = 1:size(path_info,1)
        exp_AoA = exp((2 * pi * antenna_space * cos(path_info(k,1)*pi / 180) * WLAN_paras.frequency / WLAN_paras.speed_light) * -1i);
        exp_ToF = exp(2 * pi * WLAN_paras.frequency_space * path_info(k,2) * -1i);
        
        for t = 1:WLAN_paras.num_antenna
            tmp_AoA = exp_AoA.^(t-1);
            for m = 1:WLAN_paras.num_subcarrier
                CSI(t,m) = CSI(t,m) + exp_ToF.^(m - 1) * tmp_AoA * WLAN_paras.path_complex_gain(1,k); 
            end
        end
    end
    
    if WLAN_paras.has_noise == 1
        CSI = awgn(CSI,WLAN_paras.SNR,'measured');
    end
end