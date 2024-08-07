%% Clear environment
clear;
clc;
seed = 100;
rng(seed);

% Specify the directory containing the datasets
data_dir = '/Users/shaozishan/Desktop/Research/24grad_summer/matlab_nystrom/numerical_experiment/data/real/medium/';


% Load the training dataset
train_fname = 'a7a.train.txt';
[train_labels, train_features] = libsvmread(fullfile(data_dir, train_fname));

% Load the test dataset
test_fname = 'a7a.test.txt';
[test_labels, test_features] = libsvmread(fullfile(data_dir, test_fname));

% Check if the number of features is different and pad the training data if needed
num_features_train = size(train_features, 2);
num_features_test = size(test_features, 2);

if num_features_train < num_features_test
    % Add a column of zeros to the training features to match the test set
    train_features(:, num_features_test) = 0;
elseif num_features_train > num_features_test
    % Add a column of zeros to the test features to match the training set
    test_features(:, num_features_train) = 0;
end

% Total number of observations in the test set
num_test_obs = size(test_labels, 1);

% Determine how many more observations are needed to reach 20,000
num_additional_obs = 12000 - num_test_obs;

if num_additional_obs > 0
    % Randomly select the required number of observations from the training set
    rand_indices = randperm(size(train_labels, 1), num_additional_obs);
    selected_train_labels = train_labels(rand_indices);
    selected_train_features = train_features(rand_indices, :);
    
    % Merge the selected training observations with the test set
    final_labels = [test_labels; selected_train_labels];
    final_features = [test_features; selected_train_features];
else
    % If the test set is already 20,000 or more, just use the test set
    final_labels = test_labels;
    final_features = test_features;
end

% Specify the output file name
output_file = fullfile(data_dir, 'adult.txt');

% Save the final dataset to a LIBSVM format file
libsvmwrite(output_file, final_labels, final_features);

% Display a message confirming the save
fprintf('Final dataset (20000 observations) has been saved as %s\n', output_file);
