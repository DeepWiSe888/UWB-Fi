function [path_info_output,ind,max_N_value] = MUSIC_peaks(samples,signal_space,X,Y)
% Define the matrix to store the computed AoA and ToF
% X = theta;
% Y = tau;
% signal_space = WLAN_paras.num_path;
% samples = MUSIC_spectrum;
path_info_output = zeros(signal_space,2);
ind = zeros(signal_space,2);
max_N_value = zeros(1,signal_space);

% Find the top signal_space peak values
for m = 1:length(X)
    for n = 1:length(Y)
        step = [1 0;0 1;-1 0;0 -1];
        scope = [length(X),length(Y)];
        mark = 1;

        % Check if the current point is a peak
        for k = 1:size(step,1)
            temp_x = m + step(k,1);
            if temp_x < 1 || temp_x > scope(1)
                temp_x = m;
            end
            temp_y = n + step(k,2);
            if temp_y < 1 ||temp_y > scope(2)
                temp_y = n;
            end
            if samples(m,n) < samples(temp_x,temp_y)
                mark = 0;
                break;
            end
        end
       
        % If it is a peak, store it
        if mark == 1
            min_index = minI(max_N_value);
            if max_N_value(min_index) < samples(m,n)  % Search for larger values
                max_N_value(min_index) =  samples(m,n);
                path_info_output(min_index,:) = [X(m) Y(n)];
                ind(min_index,:) = [m,n]; %[theta tau]
            end
        end
    end
end

end

%% Find the index of the smallest element in the input array
% input = max_N_value;
function index = minI(input)
    index  = 1;
    for k = 2:length(input)
        if input(k) < input(index)
            index = k;
        end
    end
end
