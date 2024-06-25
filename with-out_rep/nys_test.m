%%
clear;
clc;

rng(114514);

% Generate a 7x7 SPSD matrix
% sprandsym(n,density,rc), which creates random sparse symmetric matrices with eigenvalues rc
% so a nonnegative vector rc leads to a PSD matrix.

% Set the size of the matrix
n = 400;

% Set the density (sparsity) of the matrix: range 0 ~ 1
density = 0.5; 

% set up the desired rank of the matrix
rank = 5; % NOTE: ensure rank < n

% Set the eigenvalues for the matrix
% A nonnegative vector rc will result in a PSD matrix
rc = rand(1, n); 
rc(rank+1:n) = 0;
rc = rc * 0.1 + 0.001; % Scale and shift to ensure positiveness  
                       % NOTE: this value is arbitrary selected
                       %       the reason I do this is to mimic what
                       %       happens in the SVM and ridge script
                       % if not shifting, then not SPSD
disp(rc)

% Create the random sparse SPSD matrix
G = sprandsym(n, density, rc);

% check if SPSD
tf = issymmetric(G);
d = eig(G);
isposdef = all(d >= 0);

if (tf && isposdef)
    disp('SPSD YES');
else
    disp('SPSD NO');
end
disp('----------------------------')

disp("G")
disp(full(G))


% c is the columns selected, k is the rank considered
c = 200; % Example number of columns to sample
k = 4;

% Apply the Nystrom method to dataset A
%approx_G = nystrom_approx(G, c, k, n);

approx_G = my_approx_v2(G, c, k, n);  % with replacement
%approx_G = my_approx_v2_wor(G, c, k, n); % without replacement

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
[U, Sigma, V] = svds(G,c);

% Reconstruct the best rank-k approximation to W
approx_G_SVD = U * Sigma * V';

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


