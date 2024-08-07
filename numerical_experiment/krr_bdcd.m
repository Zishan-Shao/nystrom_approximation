function [ results ] = krr_bdcd(A, b, lambda, blksize, maxit, tol, seed, freq, opt)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
rng(seed);
[m n] = size(A');
alpha = zeros(m,1);
v = zeros(n,1);
del_a = zeros(blksize,1);

iter = 1;
idx_prev = -1;
idx = -1;
results.alpha = alpha;
%results.del_a = [];
%results.idx = [];
%results.r = [];
%results.ref_alpha = [];
results.sol_err = norm(alpha-opt.ref_sol)/norm(opt.ref_sol);
    while(iter <= maxit)
        idx = randsample(m,blksize);
        %results.idx(end+1) = idx;
        if(strcmpi(opt.kernel, 'poly'))
            v = poly(A(:,idx), A, opt.degree);

            %M = poly(A(:, idx), A(:, idx), opt.degree);
            M = v(:,idx);
        elseif(strcmpi(opt.kernel, 'linear'))
            v = linear(A(:,idx), A);
            %M = linear(A(:, idx), A(:, idx));
            M = v(:,idx);

        elseif(strcmpi(opt.kernel, 'gauss'))
            v = gaussian(A(:,idx), A, opt.gamma);
            %M = gaussian(A(:,idx), A(:,idx));
            M = v(:,idx);
            
        end
    
        T = (1/(lambda*m^2))*M + (eye(blksize)/m);
        r = b(idx)/m - alpha(idx)/m - (1/(lambda*m^2))*v*alpha;
        
        del_a = T\r;
        %results.del_a(end + 1) = del_a;
        %results.r(end+1) = r;
        alpha(idx) = alpha(idx) + del_a;
        %results.ref_alpha(end + 1,:) = alpha;
        iter = iter + 1;
        
        if(mod(iter, freq) == 0)
            results.sol_err(end+1) = norm(alpha - opt.ref_sol)/norm(opt.ref_sol);
            %results.sol_err(end)
            fprintf('KRR soultion error (iter %d): %0.16g\n', iter, results.sol_err(end))
            
            if(results.sol_err(end) <= tol)
                results.alpha = alpha;
                return;
            end
                
        end
    end
    results.sol_err(end)
    results.alpha = alpha;
    
end


function k = linear(u, v)
    k = u'*v;
end

function k = poly(u, v, d)
    k = (u'*v).^d;
end

function k = gaussian(u,v, gamma)
    blksize = size(u,2);  
    k = zeros(blksize, size(v,2));
    for i=1:blksize
        k(i, :) = exp(-gamma*(sum((u(:,i)-v).^2, 1)));
    end
end
