function file_names = get_filename(data_path)
files = dir(data_path);
file_names = {}; 
for i = 1:length(files)
    if endsWith(files(i).name, '.mat')
    % if ~files(i).isdir && ~strcmp(files(i).name, '.') && ~strcmp(files(i).name, '..') && ~strcmp(files(i).name, '.DS_Store')
        file_names{end+1} = files(i).name;
    end
end