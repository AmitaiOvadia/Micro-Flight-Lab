function error =  compare_histograms(Data, sub_sample_indexses, num_of_bins)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
[m,d] = size(Data);
error = 0;
samples = Data(sub_sample_indexses, :);

% for each feature: check distance between histograms of sample and data and sum
for i = 1:d  
    samples_feature = samples(:, i);
    Data_feature = Data(:, i);
    hist_subsample_d =  histogram(samples_feature, num_of_bins,'Normalization','pdf').Values;
    hist_Data_d = histogram(Data_feature, num_of_bins,'Normalization','pdf').Values;
    diff = abs(hist_Data_d - hist_subsample_d);
    distance_between_hist = sum(diff);  % least squers
    error = error + distance_between_hist;
end
end

