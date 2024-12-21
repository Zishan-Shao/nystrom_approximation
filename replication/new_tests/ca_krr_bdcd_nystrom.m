function [ results ] = ca_krr_bdcd_nystrom(A, b, lambda, blksize, s, maxit, tol, seed, freq, gamma, p)
% Nystrom version of s-step kernel block descent using RBF kernel only.
%
% A: data matrix d x m
% b: labels/targets vector m x 1
% lambda: regularization parameter
% blksize: block size
% s: number of steps before updating alpha
% maxit: max number of iterations
% tol: tolerance on solution error
% seed: random seed
% freq: frequency of solution error check
% gamma: RBF kernel parameter
% p: number of landmark points for Nystrom approximation

rng(seed);
[d, m] = size(A);

% Select p landmark points
J = randperm(m, p);
A_landmarks = A(:, J);

% Compute W = K(X_J, X_J)
W = gaussian(A_landmarks, A_landmarks, gamma);

% Compute C = K(X, X_J)
C = gaussian(A, A_landmarks, gamma); % Size: m x p

% Regularize W to improve conditioning if needed
% Here we add a small ridge for stability, can be tuned
eps_reg = 1e-12;
W_reg = W + eps_reg*eye(p);

% Precompute W_inv:
W_inv = inv(W_reg);

% Initialize alpha
alpha = zeros(m,1);
del_a = zeros(s*blksize, 1);
I = speye(m,m);

iter = 1;
% Suppose we have a reference solution opt.ref_sol for computing error:
% If not, skip error computations or adapt accordingly.
opt.ref_sol = zeros(m,1); % For demonstration only
results.alpha = alpha;
results.sol_err(1) = norm(alpha - opt.ref_sol)/norm(opt.ref_sol);

while (iter <= maxit)
    idx = zeros(s*blksize, 1);
    for i=1:s
        ps = (i-1)*blksize + 1;
        pe = i*blksize;
        idx(ps:pe) = randsample(m, blksize);
    end
    idx_overlap = I(:,idx)'*I(:,idx);

    % Instead of directly computing M = gaussian(A(:,idx), A(:,idx), gamma),
    % use Nystrom: M ≈ C(idx,:) * W_inv * C(idx,:)' 
    %
    % Similarly, we need something analogous to v = gaussian(A(:,idx), A, gamma).
    % With Nystrom, K ≈ C W_inv C', so:
    % v(ptr_start:ptr_end,:) = K(idx, :) = C(idx,:) * W_inv * C'

    C_idx = C(idx,:); % m->blksize*s, p
    % To compute v for a specific block of idx and all points:
    % v = K(idx,:) = C(idx,:) * W_inv * C'
    % However, v would be a large matrix (blksize x m).
    % We only need products like v * alpha and v(:, idx_s):
    % We'll compute these on-demand.

    % Precompute for efficiency: W_inv_C' = W_inv * C'
    W_inv_Ct = W_inv * C';

    % When we need v*alpha = (C(idx,:)*W_inv*(C')*alpha)
    % = C(idx,:)*W_inv*(C'^alpha) = C(idx,:)* (W_inv * (C'^alpha))
    % Precompute u = C'^alpha (p x 1)
    u = C' * alpha;
    W_inv_u = W_inv * u;
    
    for i=1:s
        ps = (i-1)*blksize + 1;
        pe = i*blksize;
        idx_s = idx(ps:pe);
        
        C_idx_s = C_idx(ps:pe, :); % blksize x p

        % Compute local block M ≈ C(idx_s,:) W_inv C(idx_s,:)' 
        M = C_idx_s * (W_inv * C_idx_s');

        % T = (1/(lambda*m^2))*M + (eye(blksize)/m)
        T = (1/(lambda*m^2))*M + (eye(blksize)/m);

        % Compute the residual r:
        % r = b(idx_s)/m - alpha(idx_s)/m - (1/(lambda*m^2))*v(idx_s,:)*alpha
        % But v(idx_s,:)*alpha = C_idx_s * W_inv_u (since v*alpha = C(idx,:)*W_inv_u)
        v_alpha_block = C_idx_s * W_inv_u;
        r = b(idx_s)/m - alpha(idx_s)/m - (1/(lambda*m^2))*v_alpha_block;

        % Adjust residual for previously solved sub-blocks in this s-step:
        if(i > 1)
            % Need v(idx_s, idx(1:(i-1)*blksize)) * del_a(1:(i-1)*blksize)
            % This is K(idx_s, idx_old) approx:
            % K(idx_s, idx_old) = C_idx_s * W_inv * C(idx_old,:)'
            idx_old = idx(1:(i-1)*blksize);
            C_idx_old = C(idx_old,:);
            K_idx_s_idx_old = C_idx_s * (W_inv * C_idx_old'); 
            
            r = r - (1/(lambda*m^2))*(K_idx_s_idx_old * del_a(1:(i-1)*blksize));
            r = r - idx_overlap(ps:pe, 1:(i-1)*blksize)*del_a(1:(i-1)*blksize)/m;
        end

        % Solve T * del_a_block = r
        del_a(ps:pe) = T\r;
        iter = iter + 1;

        if (mod(iter, freq) == 0 || iter > maxit)
            tmp_alpha = alpha;
            for j=1:i
                pss = (j-1)*blksize + 1;
                pee = j*blksize;
                idx_sub = idx(pss:pee);
                tmp_alpha(idx_sub) = tmp_alpha(idx_sub) + del_a(pss:pee);
            end
            results.sol_err(end + 1) = norm(tmp_alpha - opt.ref_sol)/norm(opt.ref_sol);
            fprintf('Iteration %d, solution error: %0.16g\n', iter, results.sol_err(end))
            if(results.sol_err(end) <= tol || iter > maxit)
                alpha = tmp_alpha;
                results.alpha = alpha;
                return;
            end
        end
    end

    % After s steps, update alpha
    for i=1:s
        ps = (i-1)*blksize + 1;
        pe = i*blksize;
        idx_s = idx(ps:pe);
        alpha(idx_s) = alpha(idx_s) + del_a(ps:pe);
    end
    del_a = zeros(s*blksize, 1);

end

results.alpha = alpha;
end

function k = gaussian(u,v, gamma)
    blksize = size(u,2);
    k = zeros(blksize, size(v,2));
    for i=1:blksize
        k(i, :) = exp(-gamma*(sum((u(:,i)-v).^2, 1)));
    end
end
