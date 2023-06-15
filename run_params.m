clear
clc

% data_path = '/home/xiaosx/data/mvc_datasets/';
% code_path = '/home/xiaosx/codes/clustering_codes/2017_TIP_MVAR_matlab/';
data_path = 'F:\research\Multi-view Clustering\data\datasets\';
code_path = 'E:\05_SemiGNN\Experiments\02_Compared methods\2017_TIP_MVAR_matlab\';
addpath(genpath(data_path))
addpath(genpath(code_path))

% list_dataset = ["100leaves.mat", "3sources.mat", "Animals.mat", "BBC.mat", "BBCSport.mat", "Caltech101-7.mat", "Caltech101-20.mat", "Caltech101-all.mat", "Caltech101-all_v2.mat", "Citeseer.mat", "COIL20.mat", "CUB.mat", "Flower17.mat", "GRZA02.mat", "Hdigit.mat", "Mfeat.mat", "Mfeat_v2.mat", "MITIndoor.mat", "MSRCv1.mat", "MSRCv1_v2.mat", "NGs.mat", "Notting-Hill.mat", "NUS.mat", "NUS_v2.mat", "NUS_v3.mat", "ORL_v2.mat", "ORL_v3.mat", "Out_Scene.mat", "Prokaryotic.mat", "ProteinFold.mat", "Reuters.mat", "Reuters_v2.mat", "Scene15.mat", "UCI-Digits.mat", "UCI-Digits_v2.mat", "UCI-Digits_v3.mat", "WebKB.mat", "WebKB_Cornell.mat", "WebKB_Texas.mat", "WebKB_Washington.mat", "WebKB_Wisconsin.mat", "Yale.mat", "YaleB_Extended.mat", "YaleB_F10.mat"];
list_dataset = ["UCI-Digits.mat",];

ratio = 0.1;

for idx = 1: length(list_dataset)
    % load data
    dataset_name = list_dataset(idx)
    file_path = [data_path, dataset_name];
    load(dataset_name);
    
    % initialze of adjusting parameters
    best_mu = 0;
    
    result_path = [code_path, 'results.txt'];
    fp = fopen(result_path,'A');
    fprintf(fp, 'Dataset: %s\n', dataset_name);

    % adjusting number of anchor
    list_mu = [1e0 1e1 1e2 1e3 1e4 1e5 1e6];
    best_acc = 0.0;
    for i = 1 : length(list_mu)
        repNum = 10;
        result_ACC = zeros(repNum,1);
        
        for j = 1:repNum
            results = fsClassification_MVAR(X, Y', ratio, list_mu(i));
            result_ACC(j) = results(1);
        end
        
        curr_acc = mean(result_ACC)
        if curr_acc > best_acc
            best_acc = curr_acc;
            best_mu = list_mu(i);
        end
    end
    fprintf(fp, 'best mu: %s\n\n', num2str(best_mu));
    fclose(fp);
end