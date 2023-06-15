clear
clc

% data_path = '/home/xiaosx/data/mvc_datasets/';
% code_path = '/home/xiaosx/codes/clustering_codes/2017_TIP_MVAR_matlab/';
data_path = 'C:\Users\hsj\OneDrive\datasets\';
code_path_ = 'C:\Users\hsj\OneDrive\multi_view\对比算法代码\acc_f1_compute\2017_TIP_MVAR_matlab\';
code_path = 'C:\Users\hsj\OneDrive\multi_view\对比算法代码\acc_f1_compute\2017_TIP_MVAR_matlab\ratio\';
addpath(genpath(data_path))
addpath(genpath(code_path_))

% list_dataset = ["100leaves.mat", "3sources.mat", "Animals.mat", "BBC.mat", "BBCSport.mat", "Caltech101-7.mat", "Caltech101-20.mat", "Caltech101-all.mat", "Caltech101-all_v2.mat", "Citeseer.mat", "COIL20.mat", "CUB.mat", "Flower17.mat", "GRZA02.mat", "Hdigit.mat", "Mfeat.mat", "Mfeat_v2.mat", "MITIndoor.mat", "MSRCv1.mat", "MSRCv1_v2.mat", "NGs.mat", "Notting-Hill.mat", "NUS.mat", "NUS_v2.mat", "NUS_v3.mat", "ORL_v2.mat", "ORL_v3.mat", "Out_Scene.mat", "Prokaryotic.mat", "ProteinFold.mat", "Reuters.mat", "Reuters_v2.mat", "Scene15.mat", "UCI-Digits.mat", "UCI-Digits_v2.mat", "UCI-Digits_v3.mat", "WebKB.mat", "WebKB_Cornell.mat", "WebKB_Texas.mat", "WebKB_Washington.mat", "WebKB_Wisconsin.mat", "Yale.mat", "YaleB_Extended.mat", "YaleB_F10.mat"];
list_dataset = ["Youtube.mat"];
list_mu = [1e-4, ]; %5e1

% list_ratio = [0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80];
list_ratio = [0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50];
                                                                                                                                         

for idx = 1: length(list_dataset)
    % load data
    dataset_name = list_dataset(idx)
    file_path = [data_path, dataset_name];
    load(dataset_name);
    Y = Y;
    X = X;
%     Y = truth;
    num_clusters = max(Y);
    num_views = length(X);
    num_samples = length(Y);
    
    for ii = 1:length(list_ratio)
        result_path = [code_path, 'results.txt'];
        fp = fopen(result_path,'A');
        repNum = 5;
        result_ACC = zeros(repNum,1);
        result_micro_P = zeros(repNum,1);
        result_macro_P = zeros(repNum,1);
        result_micro_R = zeros(repNum,1);
        result_macro_R = zeros(repNum,1);
        result_micro_F = zeros(repNum,1);
        result_macro_F = zeros(repNum,1);
        result_time = zeros(repNum,1);

        % Add a waitbar
        steps = repNum;
%         hwait = waitbar(0, 'Please wait ...');

        for i = 1:repNum
            % Start to time
            t1 = clock;

            results = fsClassification_MVAR(X, Y, list_ratio(ii), list_mu(idx));
            result_ACC(i) = results(1)
            result_micro_P(i) = results(2);
            result_macro_P(i) = results(3);
            result_micro_R(i) = results(4);
            result_macro_R(i) = results(5);
            result_micro_F(i) = results(6);
            result_macro_F(i) = results(7);
            t2 = clock;
            result_time(i) = etime(t2,t1);
            
            % Update waitbar
%             waitbar(i/steps, hwait, 'Will complete soon.');
        end

        % Close waitbar
%         close(hwait);

        % Compute the mean and standard devation
        ACC = [mean(result_ACC), std(result_ACC)]
        micro_P = [mean(result_micro_P), std(result_micro_P)];
        macro_P = [mean(result_macro_P), std(result_macro_P)]
        micro_R = [mean(result_micro_R), std(result_micro_R)];
        macro_R = [mean(result_macro_R), std(result_macro_R)]
        micro_F = [mean(result_micro_F), std(result_micro_F)];
        macro_F = [mean(result_macro_F), std(result_macro_F)]
        Time = [mean(result_time), std(result_time)];

        fprintf(fp, 'Dataset: %s\n', dataset_name);
        fprintf(fp, 'Ratio: %10.2f\n', list_ratio(ii));
        fprintf(fp, 'ACC: %10.2f (%6.2f)\n', ACC(1)*100, ACC(2)*100);
        fprintf(fp, 'macro_P: %10.2f (%6.2f)\n', macro_P(1)*100, macro_P(2)*100);
        fprintf(fp, 'macro_R: %10.2f (%6.2f)\n', macro_R(1)*100, macro_R(2)*100);
        fprintf(fp, 'macro_F: %10.2f (%6.2f)\n', macro_F(1)*100, macro_F(2)*100);
%         fprintf(fp, 'micro_P: %10.4f\t%6.4f\n', micro_P(1), micro_P(2));
%         fprintf(fp, 'micro_R: %10.4f\t%6.4f\n', micro_R(1), micro_R(2));
%         fprintf(fp, 'micro_F: %10.4f\t%6.4f\n', micro_F(1), micro_F(2));
        fprintf(fp, 'Time: %10.4f\t%10.4f\n\n', Time(1), Time(2));
        fclose(fp);
    end
end
