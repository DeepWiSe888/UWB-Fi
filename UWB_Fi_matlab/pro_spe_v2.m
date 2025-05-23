% Processing the MUSIC spectrum v2
function MUSIC_spectrum = pro_spe_v2(MUSIC_spectrum,path_info,theta,tau,near,p)
    for i = 1:size(path_info,1)
        [~,ind_theta]=min(abs(path_info(i,1) - theta));
        [~,ind_tau]=min(abs(path_info(i,2) - tau));
        neighbor_points = nearP(ind_theta,ind_tau,theta,tau);
        if i == 1
            MUSIC_spectrum(ind_theta,ind_tau) = max(max(MUSIC_spectrum)) * 1; 
            if near == 1
            for j = 1: size(neighbor_points,1)
                MUSIC_spectrum(neighbor_points(j,1),neighbor_points(j,2)) = MUSIC_spectrum(neighbor_points(j,1),neighbor_points(j,2))*1.2; %1.2
            end
            end
        else
            MUSIC_spectrum(ind_theta,ind_tau) = MUSIC_spectrum(ind_theta,ind_tau) + max(max(MUSIC_spectrum)) * (p-i*0.01); % 1.2
            if near == 1
            for j = 1: size(neighbor_points,1)
                MUSIC_spectrum(neighbor_points(j,1),neighbor_points(j,2)) = MUSIC_spectrum(neighbor_points(j,1),neighbor_points(j,2))*1.1;    %1.1
            end
            end
        end
    end
end
