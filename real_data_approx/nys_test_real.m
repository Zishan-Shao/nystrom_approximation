%%
clear;
clc;

rng(114514);

fname = 'breast-cancer_scale';
[b,A] = libsvmread(strcat('/Users/shaozishan/Desktop/Research/24grad_summer/matlab_nystrom/', fname, '.txt'));

[m , n] = size(A);

% make A sparse if necessary
A = sparse(A);

% Make G sparse if it's not already
G = A' * A;
G = sparse(G);

% Check if G is SPSD
tf = issymmetric(G);
d = eig(G);
isposdef = all(d >= 0); 

if (tf && isposdef)
    disp('SPSD YES');
else
    disp('SPSD NO');
end

% Display G as a sparse matrix
disp('----------------------------')

disp("G")
disp(full(G))
disp("rank G")
disp(rank(full(G)))


% Assume A is your data matrix, k is the rank of approximation, and c is the number of columns to sample
%k = 3; 
c = 6; % Example number of columns to sample
k = 3;

% Apply the Nyström method to dataset A
%approx_G = nystrom_approx(G, c, k, n);

approx_G = approx_real(A, c, k, n); % with replacement

disp("Approx")
disp(approx_G)

disp("-----------------------------")


% display norms and error comparison
% Calculate and display the Frobenius and spectral norms of G
FN_G = norm(G, 'fro'); % frobenus norm
SN_G = svds(G, 1); % Spectral Norm
                   % NOTE: could not be computed directly with matlab, so SVD

fprintf('F-norm of G: %.4f\n', FN_G);
fprintf('S-norm of G: %.4f\n', SN_G);
disp("-----------------------------")

% Display the Frobenius and spectral norms of the approximated G
FN_G_approx = norm(approx_G, 'fro');
SN_G_approx = svds(approx_G, 1); 

fprintf('F-norm of Approx G: %.4f\n', FN_G_approx);
fprintf('S-norm of Approx G: %.4f\n', SN_G_approx);
disp("-----------------------------")


% Add the following lines to compute the norms of the difference
fro_diff = norm(G - approx_G, 'fro'); % Line 87
spec_diff = svds(G - approx_G, 1); % Line 88

% Display the norms of the difference
fprintf('Frobenius norm of the difference: %.4f\n', fro_diff);
fprintf('Spectral norm of the difference: %.4f\n', spec_diff);

% Display the differences
%fprintf('Difference in Frobenius norms: %.4f\n', FN_G - FN_G_approx);
%fprintf('Difference in Spectral norms: %.4f\n', SN_G - SN_G_approx);





% compare the accuracy of the nystrom approximation of G (using c columns of A) 
% and the accuracy of the truncated SVD with rank k = c (so compute only the 
% top c singular values of A). This will tell us how well the nystrom approximation 
% works when compared to SVD

disp(' ');
disp("=====================================================");
disp(' ');
disp("Truncated SVD Approximation with rank k = c");

% Compute G_c, the best rank-c approximation to G
[U, Sigma, V] = svds(G);
Sigma_c = Sigma(1:c, 1:c); %select top c

% Create a zero matrix with the same size as Sigma
Sigma_cs = zeros(size(Sigma));

% Place Sigma_cs at the top left of the zero matrix
Sigma_cs(1:c, 1:c) = Sigma_c;

% Reconstruct the best rank-k approximation to W
approx_G_SVD = U * Sigma_cs * V';

% Display the truncated SVD approximation
%disp("Truncated SVD Approximation Matrix");
%disp(full(approx_G_SVD));


% Calculate and display the Frobenius and spectral norms of G
FN_GSVD = norm(approx_G_SVD, 'fro'); % Frobenus norm
SN_GSVD = svds(approx_G_SVD, 1); % Spectral Norm: Compute only the largest singular value

disp("-----------------------------");
fprintf('F-norm of G: %.4f\n', FN_GSVD);
fprintf('S-norm of G: %.4f\n', SN_GSVD);


% Compare the Nystrom approximation with the SVD approximation
disp("-----------------------------");
disp("Comparison of Approximations");

fro_diff = norm(approx_G - approx_G_SVD, 'fro'); % Line 87
spec_diff = svds(approx_G - approx_G_SVD, 1); % Line 88

% Display the norms of the difference
fprintf('Frobenius norm of the difference between approximations: %.4f\n', fro_diff);
fprintf('Spectral norm of the difference between approximations: %.4f\n', spec_diff);


