%% Find the neighboring points of a given point
function neighbor_points = nearP(ind_theta,ind_tau,theta,tau)
    theta_min = 1;
    theta_max = length(theta);
    tau_min = 1;
    tau_max = length(tau);

    count = 1;
    for delta_theta = -1:1
        for delta_tau = -1:1
            % Excluding the point itself.
            if delta_theta == 0 && delta_tau == 0
                continue;
            end
            
            % Compute the coordinates of the surrounding points.
            theta = ind_theta + delta_theta;
            tau = ind_tau + delta_tau;
            
            % Check whether the surrounding points are within the given range.
            if theta >= theta_min && theta <= theta_max && tau >= tau_min && tau <= tau_max
                neighbor_points(count, :) = [theta, tau];
                count = count + 1;
            end
        end
    end    
end
