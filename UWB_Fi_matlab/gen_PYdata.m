%% generate the dataset for pytorch model
function [input, output] = gen_PYdata(pathT, pathS, data_Flag)
% pathT: CSI dataset path
% pathS: spectrum path
    if strcmp(data_Flag, 'train')
        num = [1:50,71:120,141:190];
    elseif strcmp(data_Flag, 'test')
        num = [61:70,131:140,201:210];
    elseif strcmp(data_Flag, 'valid')
        num = [51:60,121:130,191:200];
    end

    file_names = get_filename(pathS);

    pp = '.mat';
    len = length(num);
    input = zeros(281,121,4,len*length(file_names));
    output = zeros(281,121,len*length(file_names));
    for i = 1:length(file_names)
        var = num2str(i);
        dataT = [pathT '/loc' var pp];
        tmp = load(dataT);
        eval(['tmp = tmp.loc' num2str(i) ';']);
    
        speT = [pathS '/spe' var pp];
        spe = load(speT);
        eval(['spe = spe.spe' num2str(i) ';']);
    
        for j = 1:len
            input(:,:,1,(i-1)*len+j) = imresize(squeeze(tmp(:,:,1,num(j))),[281,121]);
            input(:,:,2,(i-1)*len+j) = imresize(squeeze(tmp(:,:,2,num(j))),[281,121]);
            input(:,:,3,(i-1)*len+j) = imresize(squeeze(tmp(:,:,3,num(j))),[281,121]);
            input(:,:,4,(i-1)*len+j) = imresize(squeeze(tmp(:,:,4,num(j))),[281,121]);
    
            output(:,:,(i-1)*len+j) = spe;      
        end
        fprintf('Processing NO.%d data \n',i)
    end
end