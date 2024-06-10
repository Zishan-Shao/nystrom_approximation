%%
clear;
clc;

rng(114514);

% Generate a 7x7 SPSD matrix
% sprandsym(n,density,rc), which creates random sparse symmetric matrices with eigenvalues rc
% so a nonnegative vector rc leads to a PSD matrix.

% Set the size of the matrix
n = 20;

% Set the density (sparsity) of the matrix: range 0 ~ 1
density = 0.7; 

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


% Assume A is your data matrix, k is the rank of approximation, and c is the number of columns to sample
%k = 3; % Example rank for approximation, in our case, for simplicity
%consideration, we set k equals to c
c = 4; % Example number of columns to sample
k = 3;

% Apply the Nyström method to dataset A
%approx_G = nystrom_approx(G, c, k, n);

approx_G = my_approx(G, c, k, n);

disp("Approx")
disp(approx_G)

disp("-----------------------------")


% display norms and error comparison
% Calculate and display the Frobenius and spectral norms of G
FN_G = norm(G, 'fro'); % frobenus norm
[~, ~, sigma_G] = svds(G, 1); % Compute only the largest singular value
SN_G = sigma_G(1); % Spectral Norm
                   % NOTE: could not be computed directly with matlab, so SVD

fprintf('F-norm of G: %.4f\n', FN_G);
fprintf('S-norm of G: %.4f\n', SN_G);
disp("-----------------------------")

% Display the Frobenius and spectral norms of the approximated G
FN_G_approx = norm(approx_G, 'fro');
[~, ~, sigma_G_approx] = svds(approx_G, 1); 
SN_G_approx = sigma_G_approx(1);

fprintf('F-norm of Approx G: %.4f\n', FN_G_approx);
fprintf('S-norm of Approx G: %.4f\n', SN_G_approx);
disp("-----------------------------")

% Display the differences
fprintf('Difference in Frobenius norms: %.4f\n', FN_G - FN_G_approx);
fprintf('Difference in Spectral norms: %.4f\n', SN_G - SN_G_approx);

