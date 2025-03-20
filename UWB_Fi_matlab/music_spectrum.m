%% Compute the 2D MUSIC spectrum (ToF-AoA spectrum).
function samples = music_spectrum(CSI,WLAN_paras,X,Y)
    signal_space = WLAN_paras.num_path;
    row = floor(WLAN_paras.num_subcarrier/2) * 2;
    column=(WLAN_paras.num_subcarrier - row/2 + 1) * (WLAN_paras.num_antenna - 1);
    smoothed_CSI = zeros(row,column,'like',CSI);
    for k = 1:(WLAN_paras.num_antenna-1)
        for t = 1:(WLAN_paras.num_subcarrier - row/2 + 1)
            smoothed_CSI(:,t + (k-1)*(WLAN_paras.num_subcarrier - row/2 + 1)) = [CSI(k,t:(t+row/2-1)),CSI(k+1,t:(t+row/2-1))].';
        end
    end
    
    correlation_matrix = smoothed_CSI * smoothed_CSI';
    [E,D] = eig(correlation_matrix);
    
    [~,indx] = sort(diag(D),'descend');
    eigenvects = E(:,indx);
    noise_eigenvects = eigenvects(:,(signal_space+1):end);
    antenna_space = WLAN_paras.antenna_space; 
    samples = complex(zeros(length(X),length(Y)));
    
    for t = 1:length(X)
        for k = 1:length(Y)
            angleE = exp(-1i * 2 * pi * antenna_space * cos(X(t)*pi/180) * WLAN_paras.frequency / WLAN_paras.speed_light);
            timeE = exp(-1i * 2 * pi * WLAN_paras.frequency_space * Y(k));
            steering_vector = complex(zeros(row,1));
            for n=0:1
                for m=1:row/2
                    steering_vector((n*row/2)+m,1) = angleE.^n * timeE.^(m-1);
                end
            end
            samples(t,k) = 1/sum(abs(noise_eigenvects' * steering_vector).^2,1);
        end
    end
end

