function downstream_data = downstream(input_data,path_info_output,tau,theta)
% extract data for downstream task
r = 1; % default
[~,tau_index] = min(abs(path_info_output(1,2)-tau));
[~,theta_index] = min(abs(path_info_output(1,1)-theta));
downstream_data = input_data(theta_index-r:theta_index+r,tau_index-r:tau_index+r);


end
