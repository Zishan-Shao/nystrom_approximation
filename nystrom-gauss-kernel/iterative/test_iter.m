%% Test Script for Iterative Nystrom Kernel Approximation
clear;
clc;

% setups
fname = 'pyrim_scale';
[b,A] = libsvmread(strcat('../data/', fname, '.txt'));
[n, m] = size(A);  % n = number of points, m = number of features

% Parameters
blksize = 20;   % Blocksize for each iteration
p = 10;         % Subset size for nystrom (always smaller than blksize as it subsample from block)
c = 1.5;          % Oversampling factor
gamma = 0.5;    % Gaussian kernel parameter

% Run nystrom approximation by iteration
disp('Running Iterative Nystrom Kernel Approximation...');
iterative_gauss_nys(A, blksize, p, c, gamma);

% Now run the non-approximated gaussian kernel
disp('Running Iterative Full Gaussian Kernel Computation...');
iterative_gauss_full(A, blksize, gamma);

%%% I will update the Frobenius norm comparison later
