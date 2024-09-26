%% Nystrom KRR experiments
clear;
clc;

% Load dataset (Abalone)
fname = 'pyrim_scale';
[b,A] = libsvmread(strcat('./data/', fname, '.txt'));
[n, m] = size(A);

rng(100)

% Parameters 
p = 4;      % this defines the size of K_II 
gamma = 0.5;  % Gaussian kernel parameter
c = 32;   % Oversampling factor for subset selection


% Nystrom approximation of Gaussian kernel
K_approx = nystrom_gauss_kernel(A, blksize, gamma,c);

% Full Gaussian kernel matrix
K_full = full_gauss_kernel(A, gamma);

% Compute Frobenius norm difference
diff = norm(K_full - K_approx, 'fro');

% Display the approximation
%disp('Approximated Kernel Matrix (Nyström):');
%disp(K_approx);

disp('Frobenius Norm Difference between Full and Nystrom Kernel:');
disp(diff);


