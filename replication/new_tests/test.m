%% Test Script for Full Kernel Block Coordinate Descent
clear;
clc;

% Load the dataset using libsvmread
fname = 'abalone'; % Dataset filename
[b, A] = libsvmread(strcat('../data/', fname, '.txt'));
[n, m] = size(A);  % n = number of points, m = number of features
fprintf('Number of data points (n): %d\n', n);

%A = normalize(A); % this works badly
%A = zscore(A);

% Parameters
num_epoch = 20;            % Number of epochs
blksize = 50;            % Block size
lambda = 1e-6;     % Regularization parameter
gamma = 0.001;     % Kernel parameter
seed = 393;

% Run the full kernel BDCD algorithm
tic;
alpha = full_kernel_bcd(A, b, num_epoch, blksize, lambda, gamma, seed);
full_bcd_time = toc;


% performance comparison
K = KernelBlock(A, A, gamma);  % Compute the full kernel matrix directly
preds = K * alpha;                 % preds on training data
train_err = norm(b - preds) / n;
fprintf('Training Error: %.16f\n', train_err);


% print out the results
fprintf('\nComputing Direct Solution...\n');
alpha_opt = (K + lambda * speye(n)) \ b;  % Direct solution
Y_pred_opt = K * alpha_opt;             % reds
MSE_opt = mean((Y_pred_opt - b).^2);

% Compute preds for Original BCD
Y_pred_full = K * alpha;  % preds from Original BCD
MSE_full = mean((Y_pred_full - b).^2);

% Display Results for Original BCD
fprintf('\nTraining MSE (against b) - Original BCD: %.16f\n', MSE_full);
fprintf('Training MSE (against b) - Direct Solution: %.16f\n', MSE_opt);


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
b_nystrom = 50;        % Block size for Nystrom BCD
lambda_prime = 1e-8;    % Regularizer lambda' (set to zero if not used)

% Run the Nystrom BCD algorithm
tic;
[alpha_nystrom, J] = nystrom_kernel_bcd(A, b, num_epoch, p, b_nystrom, lambda, lambda_prime, gamma, seed);
nys_bcd_time = toc;
%[alpha_nystrom, J] = nystrom_bcd_new(A, b, num_epoch, p, b_nystrom, lambda, lambda_prime, gamma, seed);

%disp("J's value")
%disp(J)

% Compute predictions for Nystrom BCD
K_nystrom = KernelBlock(A, A(J, :), gamma);  % Compute the Nystrom kernel matrix
Y_pred_nystrom = K_nystrom * alpha_nystrom;  % Predictions on training data
MSE_nystrom = mean((Y_pred_nystrom - b).^2);

% Display Results for Nystrom BCD
fprintf('\nTraining MSE (against b) - Nystrom BCD: %.16f\n', MSE_nystrom);

% Compare predictions with full kernel solution
pred_diff = norm(Y_pred_nystrom - Y_pred_opt) / norm(Y_pred_opt);
fprintf('Relative difference between Y_pred_nystrom and Y_pred_opt: %.16f\n', pred_diff);

% Compare alpha_full with alpha_opt
%alpha_diff = norm(alpha_nystrom - alpha_opt) / norm(alpha_opt);
%fprintf('Relative difference between alpha_nystrom and alpha_opt: %.16f\n', alpha_diff);

% Analyze predictions
disp('First 10 predictions (Nystrom BCD):');
disp(table(b(1:10), Y_pred_nystrom(1:10), Y_pred_full(1:10), Y_pred_opt(1:10), ...
    'VariableNames', {'True', 'Nystrom_Pred', 'BCD_Pred', 'Direct_Pred'}));


disp("Full time:");
disp(full_bcd_time);
disp("Nystrom time:");
disp(nys_bcd_time)



%% Kernel comparison
% in this case we compare the matrix directly computed and the nystrom
% matrix
%K_true = KernelBlock(A, A, gamma); % ground truth K

% Compute Nystrom kernel approximation
%K_base = KernelBlock(A(J, :), A(J, :), gamma);
%K_nm = KernelBlock(A, A(J, :), gamma);  % Kernel between full data and landmarks
%K_mm = KernelBlock(A(J, :), A(J, :), gamma) + 1e-10 * eye(length(J));  % Regularized kernel
%K_nys = K_nm * (K_mm \ K_nm');


% F-norm difference
%F_diff = norm(K_true - K_nys, 'fro');
%fprintf('F-Norm Difference of True Kernel v.s Nystrom Kernel: %.16f\n', F_diff);
%[m,n] = size(K_nys);
%fprintf('Dimension of Kernel: %d, %d\n', m, n);

% Visualize the true and Nystrom kernel matrices 
% see if overall pattern matched up
%figure;
%subplot(1, 2, 1);
%imagesc(K_true);
%title('True Kernel Matrix');
%colorbar;

%subplot(1, 2, 2);
%imagesc(K_nys);
%title('Nystrom Approximation');
%colorbar;


