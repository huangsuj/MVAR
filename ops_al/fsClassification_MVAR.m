function [results] = fsClassification_MVAR(X, Y, ratio, mu)
    num_views = length(X);
    num_samples = length(Y);
    num_class = length(unique(Y));
    Y_all = zeros(length(Y), 1);
    r = 1;
    maxIter = 50;
    lambda = 1e-3*ones(num_views,1);  %%% diffent view can have different lambda value
    scattered_idx = randperm(num_samples);
    num_labeled = floor(num_samples * ratio);
    idx_labeled = scattered_idx(1:num_labeled);
    idx_unlabeled = scattered_idx(num_labeled+1:end);
     
    
    length(unique(Y))
    % One-shot Y
    Y = double(Y);
    Y_label = Y(scattered_idx,:);
    temp_Y = full(sparse(1:num_samples, Y, 1, num_samples, length(unique(Y))));
%     Y_all(idx_labeled,:) =  Y(idx_labeled,:);
%     Y_all(idx_unlabeled,:) = Y(idx_unlabeled,:);
    save('C:\Users\hsj\OneDrive\my_paper\HGCNNet_for_hsj\results\scatter\MNIST10k\MVAR_label.mat', 'Y_label');
    % data can be pre-processed by centerization or normalization if necessary
    for v = 1:num_views
        X{v} = double(X{v});
        Xl{v} = X{v}(idx_labeled,:);
        Xu{v} = X{v}(idx_unlabeled,:);
    end
    Yl = temp_Y(idx_labeled,:);
    s = ones(num_samples,1);
    s(1:length(idx_labeled)) = mu;
    
    Y_req_all = zeros(length(Y), num_class);
    
    [~,~,~,y_pred, Fu] = MVAR(Xl, Xu, Yl, lambda, s, r);
%     Y_req_all(idx_labeled,:) =  Yl;
%     Y_req_all(idx_unlabeled,:) = Fu;
    Y_req_all = [Yl;Fu];
   
    %% accuracy on unlabeled data
    
    save('C:\Users\hsj\OneDrive\my_paper\HGCNNet_for_hsj\results\scatter\MNIST10k\MVAR_representation.mat', 'Y_req_all');
    results = classification_metrics(Y(idx_unlabeled), y_pred);
end
