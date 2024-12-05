%% Test Script for Full Kernel Block Coordinate Descent
clear;
clc;

% Load the dataset using libsvmread
fname = 'mushrooms'; % Dataset filename
[b, A] = libsvmread(strcat('../data/', fname, '.txt'));
[n, m] = size(A);  % n = number of points, m = number of features
fprintf('Number of data points (n): %d\n', n);

%A = normalize(A); % this works badly
%A = zscore(A);

% Parameters
num_epoch = 10;            % Number of epochs
blksize = 500;            % Block size
lambda = 1e-6;     % Regularization parameter
gamma = 2;     % Kernel parameter
seed = 393;

% Run the full kernel BDCD algorithm
alpha = full_kernel_bcd(A, b, num_epoch, blksize, lambda, gamma, seed);


% performance comparison
K = KernelBlock(A, A, gamma);  % Compute the full kernel matrix directly
preds = K * alpha;                 % preds on training data
train_err = norm(b - preds) / n;
fprintf('Training Error: %.16f\n', train_err);


% print out the results
fprintf('\nComputing Direct Solution...\n');
alpha_opt = (K + lambda * eye(n)) \ b;  % Direct solution
Y_pred_opt = K * alpha_opt;             % preds
MSE_opt = mean((Y_pred_opt - b).^2);

% Compute preds for Original BCD
Y_pred_full = K * alpha;  % preds from Original BCD
MSE_full = mean((Y_pred_full - b).^2);

% Display Results for Original BCD
fprintf('\nTraining MSE - Original BCD: %.16f\n', MSE_full);
fprintf('Training MSE - Direct Solution: %.16f\n', MSE_opt);

% Compare alpha_full with alpha_opt
alpha_diff = norm(alpha - alpha_opt) / norm(alpha_opt);
fprintf('Relative difference between alpha_full and alpha_opt: %.16f\n', alpha_diff);

% Analyze preds
disp('First 10 preds (Original BCD):');
disp(table(b(1:10), Y_pred_full(1:10), Y_pred_opt(1:10), ...
    'VariableNames', {'True', 'BCD_Pred', 'Direct_Pred'}));


% display the alpha vectors
%disp("Training Alpha")
%disp(alpha(1:6))

%disp("Optimal Alpha")
%disp(alpha_opt(1:6))




%% Nystrom methods
disp("====================================================")
fprintf("\nNystrom-based Kernel Block Dual Coordinate Descent\n");
p = 2000;               % Number of Nystrom features (the rows sampled)
b_nystrom = 200;        % Block size for Nystrom BCD
lambda_prime = 1e-5;    % Regularizer lambda' (set to zero if not used)

% Run the Nystrom BCD algorithm
[alpha_nystrom, J] = nystrom_kernel_bcd(A, b, num_epoch, p, b_nystrom, lambda, lambda_prime, gamma, seed);

% Compute predictions for Nystrom BCD
K_nystrom = KernelBlock(A, A(J, :), gamma);  % Compute the Nystrom kernel matrix
Y_pred_nystrom = K_nystrom * alpha_nystrom;      % Predictions from Nystrom BCD
MSE_nystrom = mean((Y_pred_nystrom - b).^2);

% Display Results for Nystrom BCD
fprintf('\nTraining MSE - Nystrom BCD: %.16f\n', MSE_nystrom);

% Compute relative difference in predictions
pred_diff = norm(Y_pred_nystrom - Y_pred_opt) / norm(Y_pred_opt);
fprintf('Relative difference between Y_pred_nystrom and Y_pred_opt: %.16f\n', pred_diff);

% Analyze predictions
disp('First 10 predictions (Nystrom BCD):');
disp(table(b(1:10), Y_pred_nystrom(1:10), Y_pred_full(1:10), Y_pred_opt(1:10), ...
    'VariableNames', {'True', 'Nystrom_Pred', 'BCD_Pred', 'Direct_Pred'}));



%% Kernel comparison
% in this case we compare the matrix directly computed and the nystrom
% matrix
K_true = KernelBlock(A, A, gamma); % ground truth K

% Compute Nystrom kernel approximation
K_base = KernelBlock(A(J, :), A(J, :), gamma);
K_nys = KernelBlock(A, A(J, :), gamma) * (K_base \ KernelBlock(A(J, :), A, gamma));

% F-norm difference
F_diff = norm(K_true - K_nys, 'fro');
fprintf('F-Norm Difference of True Kernel v.s Nystrom Kernel: %.16f\n', F_diff);
[m,n] = size(K_nys);
fprintf('Dimension of Kernel: %d, %d\n', m, n);

% Visualize the true and Nystrom kernel matrices 
% see if overall pattern matched up
figure;
subplot(1, 2, 1);
imagesc(K_true);
title('True Kernel Matrix');
colorbar;

subplot(1, 2, 2);
imagesc(K_nystrom);
title('Nystrom Approximation');
colorbar;


