%%
clear;
clc;
%A = 50*rand(100,10);
%b = rand(100,1);

fname = 'w7a';
[b,A] = libsvmread(strcat('../data/real/large/', fname, '.txt'));
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
s = 128;
maxit = inf;
tol = 1e-10;
seed = 100;
blksize = 64;
freq = n;
opt.kernel = 'linear';
opt.ref_sol = a_opt;

disp('KRR begins')

res = krr_bdcd(A', b, lambda, blksize, maxit, tol, seed, freq, opt);


disp('======================')
disp('CA-KRR begins')
%opt.ref_del_a = res.del_a;
%opt.ref_idx = res.idx;
%opt.ref_r = res.r;
%opt.ref_alpha = res.ref_alpha;
ca_res = ca_krr_bdcd(A', b, lambda, blksize, s, maxit, tol, seed, freq, opt);


fprintf('[Linear] blksize = %d, s = 1, Relative solution error: %0.16g\n', blksize, norm(res.alpha - a_opt)/norm(a_opt))
fprintf('[Linear] blksize = %d, s = %d, Relative solution error: %0.16g\n', blksize, s, norm(ca_res.alpha - a_opt)/norm(a_opt))

