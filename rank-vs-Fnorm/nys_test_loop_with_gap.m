%%
clear;
clc;

rng(114514);

%fname = 'breast-cancer_scale';
%[b,A] = libsvmread(strcat('/Users/shaozishan/Desktop/Research/24grad_summer/matlab_nystrom/rank-vs-Fnorm/data/', fname, '.txt'));

%fname = 'breast-cancer_scale';
%fname = 'abalone_scale';
fname = 'colon-cancer';
%fname = 'madelon'; % 2000 x 500 features
%[b,A] = libsvmread(strcat('/Users/shaozishan/Desktop/Research/24grad_summer/matlab_nystrom/rank-vs-Fnorm/data/', fname, '.txt'));
[b,A] = libsvmread(strcat('/Users/shaozishan/Desktop/Research/24grad_summer/matlab_nystrom/rank-vs-Fnorm/data/', fname));


[n , m] = size(A);

% make A sparse if necessary
A = sparse(A);

%A = A';

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

%disp("G")
%disp(full(G))
disp("rank G")
disp(rank(full(G)))



%%%%%%%% Main Part %%%%%%%%
%% In this part, we will have the for loop that iterate through different 
% values of the guessed rank (indicated by the k value), the c is set to be
% dependent to the k value; The k is set to be increasing from 1 to
% min(m,n), which is the dimension of the input dataset A.
% Params: 
%    c: num columns selected, c < min(n,m)
%    k: guessed rank for approximation, k < c

k = 1; % initialize the k value
v = 3; % gap of jump
size = ceil(min(m, n) / v) + 1;

k_vector = [];
k_vector(1:size) = 0; % record the k values
Fnorm_diff = [];
Fnorm_diff(1:size) = 0;  % use this list to record the changes in F-norms
Fnorm_diff_svdk = [];
Fnorm_diff_svdk(1:size) = 0; % record F-norms changes in svd_k approximation of G

%k = 3;
%c = 2 * k; % Example number of columns to sample

full_rank = 0; % check if already run the full rank
i = 1;

disp("---------------------- Approx Accuracy Measures -----------------------")
disp('|    k    |    real G    |    SVD G    |    nys_c      |    svd_c      |');
disp('|---------+--------------+-------------+---------------+---------------|')

% Assume A is your data matrix, k is the rank of approximation, and c is the number of columns to sample
while k <= (min(m, n) + 1) % so that we can run (k = full rank) without being stucked

    c = 2 * k; % Example number of columns to sample

    % Apply the Nyström method to dataset A  
    approx_G = approx_real(A, c, k, max(m,n)); % with replacement
    
    % Part 1: Compare nys_G_k with real G
    p1_diff = norm(G - approx_G, 'fro');


    % Part 2: Compare nys_G_k with SVD G
    % Compute G_c, the best rank-c approximation to G
    [U, Sigma, V] = svds(G,c);
    %Sigma_c = Sigma(1:c, 1:c); %select top c
    %disp(Sigma)
    %disp(size(Sigma))

    % Create a zero matrix with the same size as Sigma
    %Sigma_cs = zeros(size(Sigma));
    
    % Place Sigma_cs at the top left of the zero matrix
    %Sigma_cs(1:c, 1:c) = Sigma_c;
    
    % Reconstruct the best rank-k approximation to W
    approx_G_SVD = U * Sigma * V';
    p2_diff = norm(approx_G - approx_G_SVD, 'fro'); % Line 87


    % Part 3: Compare nys_G_k with nys_G_c
    nys_G_C = approx_real(A, c, c, max(m,n)); % with replacement
    p3_diff = norm(approx_G - nys_G_C, 'fro');
    

    % Part 4: Compare nys_G_k with SVD k
    [U, Sigma, V] = svds(G,k);
    %Sigma_k = Sigma(1:k, 1:k); %select top c
    
    % Create a zero matrix with the same size as Sigma
    %Sigma_ks = zeros(size(Sigma));
    
    % Place Sigma_cs at the top left of the zero matrix
    %Sigma_ks(1:k, 1:k) = Sigma_k;
    
    % Reconstruct the best rank-k approximation to W
    approx_G_SVD_k = U * Sigma * V';
    p4_diff = norm(approx_G - approx_G_SVD_k, 'fro'); % Line 87


    % display the result in formatted list
    fprintf('|    %d    |   %.3f   |   %.3f   |    %.3f     |    %.3f    |\n', k, p1_diff, p2_diff,p3_diff, p4_diff);
    disp('|----------------------------------------------------------------------|')


    % record the F-norm changes with increasing rank
    Fnorm_diff(i) = p1_diff;
    Fnorm_diff_svdk(i) = p4_diff;
    k_vector(i) = i;


    % Update the k value
    k = min(k + v, min(m, n));
    if (k == min(m, n)) % in case if we have run once of 
        if (full_rank == 1)
            break;
        else
            full_rank = 1; % set the condition to 1
        end
    end
    i = i + 1;
end 


%disp("Approx")
%disp(approx_G)


%% Plotting Error
% Plotting the error
%figure; 
%plot(k, Fnorm_diff, 'r-', 'DisplayName', 'Nystrom Method'); % Plot the Nystrom method differences
%hold on; 
%plot(k, Fnorm_diff_svdk, 'b--', 'DisplayName', 'SVD Method'); % Plot the SVD method differences
%hold off;

% Plot the Nystrom method differences with markers at each point
plot(k_vector, Fnorm_diff, 'r-o', 'LineWidth', 1, 'MarkerSize', 6, 'DisplayName', 'Nystrom Method');
hold on; % Hold the current plot

% Plot the SVD method differences with markers at each point
plot(k_vector, Fnorm_diff_svdk, 'b--s', 'LineWidth', 1, 'MarkerSize', 6, 'DisplayName', 'SVD Method');
hold off;

% Adding plot annotations
xlabel('Rank (k)'); % Label the x-axis
ylabel('Frobenius Norm Difference'); % Label the y-axis
title('Frobenius Norm Difference vs. Rank (k)'); % Add a title to the plot
% legend show; % Display the legend

legend('Nystrom', 'SVD-k','Location','NE')

% Enhancing plot appearance (optional)
grid on; % Add grid lines for better readability
set(gca, 'XTick', k); % Set the x-axis ticks to correspond to the k values





