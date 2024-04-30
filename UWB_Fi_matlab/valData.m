%% 验证正确率
function path = valData(path_info_outputL)
LOS = [90,2.3349e-08];
theta_range = [20,160];
n = 5; %size(res,2);
LOSr = zeros(1,2);
nLOSr = [];
for i = 1:n
    if path_info_outputL(i,1) == LOS(1,1) && abs(path_info_outputL(i,2) - LOS(1,2)) <= 1e-10
        LOSr = path_info_outputL(i,:);
    elseif path_info_outputL(i,1) ~= LOS(1,1) && abs(path_info_outputL(i,2) - LOS(1,2)) > 1e-10
        if path_info_outputL(i,1) <= theta_range(1,2) && path_info_outputL(i,1) >= theta_range(1,1)
            nLOSr = cat(2,nLOSr,path_info_outputL(i,:));
            break
        end
    end
end
if isempty(nLOSr)
    nLOSr = zeros(1,2);
end
path = cat(2,LOSr,nLOSr);
end