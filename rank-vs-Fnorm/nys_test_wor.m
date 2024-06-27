%%
clear;
clc;

rng(114514);

%fname = 'breast-cancer_scale';
%fname = 'abalone_scale';
fname = 'madelon'; % 2000 x 500 features
[b,A] = libsvmread(strcat('/Users/shaozishan/Desktop/Research/24grad_summer/matlab_nystrom/rank-vs-Fnorm/data/', fname, '.txt'));
%[b,A] = libsvmread(strcat('/Users/shaozishan/Desktop/Research/24grad_summer/matlab_nystrom/rank-vs-Fnorm/data/', fname));

[n , m] = size(A);

% make A sparse if necessary
A = sparse(A);

A = A';

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

Fnorm_diff = [];
Fnorm_diff(1:min(m,n)) = 0;  % use this list to record the changes in F-norms
Fnorm_diff_wor = [];
Fnorm_diff_wor(1:min(m,n)) = 0; % record F-norms changes in svd_k approximation of G

%k = 3;
%c = 2 * k; % Example number of columns to sample


% Assume A is your data matrix, k is the rank of approximation, and c is the number of columns to sample
for k = 1:min(m,n)

    c = 2 * k; % Example number of columns to sample

    % Apply the Nyström method to dataset A  
    approx_G = approx_real(A, c, k, n); % with replacement
    
    % Part 1: Compare nys_G_k with real G
    p1_diff = norm(G - approx_G, 'fro');


    % Part 2: Compare nys_G_k without replacement with G
    approx_G_wor = approx_real_wor(A, c, k, n);
    p2_diff = norm(approx_G_wor - G, 'fro'); % Line 87


    % record the F-norm changes with increasing rank
    Fnorm_diff(k) = p1_diff;
    Fnorm_diff_wor(k) = p2_diff;

end 


%disp("Approx")
%disp(approx_G)


%% Plotting Error
k = (1:1:min(n,m));
% Plotting the error
%figure; 
%plot(k, Fnorm_diff, 'r-', 'DisplayName', 'Nystrom Method'); % Plot the Nystrom method differences
%hold on; 
%plot(k, Fnorm_diff_svdk, 'b--', 'DisplayName', 'SVD Method'); % Plot the SVD method differences
%hold off;

% Plot the Nystrom method differences with markers at each point
plot(k, Fnorm_diff, 'r-o', 'LineWidth', 1, 'MarkerSize', 6, 'DisplayName', 'Nys-wr');
hold on; % Hold the current plot

% Plot the SVD method differences with markers at each point
plot(k, Fnorm_diff_wor, 'b--s', 'LineWidth', 1, 'MarkerSize', 6, 'DisplayName', 'Nys-wor');
hold off;

% Adding plot annotations
xlabel('Rank (k)'); % Label the x-axis
ylabel('Frobenius Norm Difference'); % Label the y-axis
title('Nystrom Approximation with/out Replacement of Column Selection'); % Add a title to the plot
% legend show; % Display the legend

legend('Nystrom-wr', 'Nystrom-wor','Location','NE')

% Enhancing plot appearance (optional)
grid on; % Add grid lines for better readability
set(gca, 'XTick', k); % Set the x-axis ticks to correspond to the k values





