%% Nystrom krr experiments
clear;
clc;
%A = 50*rand(100,10);
%b = rand(100,1);

fname = 'a8a';
[b,A] = libsvmread(strcat('./data/real/medium/', fname, '.txt'));
%[b,A] = libsvmread(strcat('./data/real/large/', fname));
[n, m] = size(A);
OPTS = optimset('TolFun', 1e-4);
smeig = eigs(@(x) (A*(A'*x)), n, 1, 'SM', OPTS);
lmeig = eigs(@(x) (A*(A'*x)), n, 1, 'LM', OPTS);

%linear kernel solution
lambda = 1e1*smeig;
K_linear = A*A';
a_opt = ((1/(lambda*n^2))*K_linear + 1/n*eye(n))\(b/n);

%bdcd solution
tic;
s = 2048;
maxit = inf;
tol = 1e-12;
seed = 100;
blksize = 16;
freq = n;
opt.kernel = 'linear';
opt.ref_sol = a_opt;

% nystrom parameters
k = ceil(blksize * 0.8);
sk = floor(k * s * 0.05);
osfct = 1.5;


disp('KRR begins')
%res = nystrom_krr(A', b, k, osfct, lambda, blksize, maxit, tol, seed, freq, opt);

disp('======================')
disp('CA-KRR begins')
%opt.ref_del_a = res.del_a;
%opt.ref_idx = res.idx;
%opt.ref_r = res.r;
%opt.ref_alpha = res.ref_alpha;
ca_res = nystrom_s_step_krr(A', b, sk, osfct, lambda, blksize, s, maxit, tol, seed, freq, opt);


fprintf('[Linear] blksize = %d, s = 1, Relative solution error: %0.16g\n', blksize, norm(res.alpha - a_opt)/norm(a_opt))
fprintf('[Linear] blksize = %d, s = %d, Relative solution error: %0.16g\n', blksize, s, norm(ca_res.alpha - a_opt)/norm(a_opt))

%% Convergence Plot (Only nystrom)
% Ensure that the lengths of sol_err arrays are sufficient
len_res = length(res.sol_err);
len_ca_res = length(ca_res.sol_err);

% Determine the index where each method first reaches the tolerance
index_res_tol = find(res.sol_err <= tol, 1, 'first');
index_ca_res_tol = find(ca_res.sol_err <= tol, 1, 'first');

% If either method does not reach the tolerance, plot all available points
if isempty(index_res_tol)
    index_res_tol = len_res;
end
if isempty(index_ca_res_tol)
    index_ca_res_tol = len_ca_res;
end

% Plotting
figure;

% Plot for the current Nyström-based KRR method
xvals_res = 0:index_res_tol-1;
semilogy(xvals_res*n+1, res.sol_err(1:index_res_tol), '-k', 'LineWidth', 2)
hold on;

% Plot for the current Nyström-based CA-KRR method
xvals_ca_res = 0:index_ca_res_tol-1;
semilogy(xvals_ca_res*n+1, ca_res.sol_err(1:index_ca_res_tol), 'ob', 'LineWidth', 2, 'MarkerSize', 8)

% Add horizontal dashed line at tolerance level
yline(tol, '--r', 'LineWidth', 2, 'Label', 'Tolerance', 'LabelHorizontalAlignment', 'right');

% Labels and Settings
xlabel('Iterations (H)')
ylabel('relative solution error')
set(gca, 'FontSize', 24);

% Update the legend to include only Nyström-based plots
legend('\mu = 1, s = 1', strcat('\mu = ', string(blksize),', s = ', string(s)),...
    'Location','NE')

% Save the plot as a PDF with 300 DPI
save_dir = './convergence/';
if ~exist(save_dir, 'dir')
    mkdir(save_dir); % Create the directory if it does not exist
end
file_name = sprintf('nys_conv_%s_b%d_s%d.pdf',fname, blksize, s);
save_path = fullfile(save_dir, file_name);
print(save_path, '-dpdf', '-r300'); % Save as PDF with 300 DPI

toc;
