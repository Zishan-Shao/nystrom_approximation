%%
clear;
clc;
%A = 50*rand(100,10);
%b = rand(100,1);

fname = 'abalone_scale';
[b,A] = libsvmread(strcat('/Users/shaozishan/Desktop/Research/24grad_summer/matlab_nystrom/rank-vs-Fnorm/data/', fname, '.txt'));
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
s = 10;
maxit = inf;
tol = 1e-4;
seed = 100;
blksize = 10;
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



figure;
npts = 15;
xvals = ceil(linspace(0, length(res.sol_err)-1, npts));
semilogy(xvals*n+1, res.sol_err(xvals+1), '-k', 'LineWidth', 2)
hold on;
semilogy(xvals*n+1, ca_res.sol_err(xvals+1), 'ob', 'LineWidth', 2, 'MarkerSize', 8)
xlabel('Iterations (H)')
ylabel('relative solution error')
set(gca, 'FontSize', 24);


legend('\mu = 1, s = 1', strcat('\mu = 1, s = ', string(s)),...
    strcat('\mu = ', string(blksize))+', s = 1',...
    strcat('\mu = ', string(blksize)) + strcat(' s = ', string(s)),...
    strcat('tol = ', string(tol)'),'Location','SE')
toc;


