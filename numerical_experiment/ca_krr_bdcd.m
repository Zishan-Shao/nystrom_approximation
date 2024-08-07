function [ results ] = ca_krr_bdcd(A, b, lambda, blksize, s, maxit, tol, seed, freq, opt)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
rng(seed);
[m n] = size(A');
alpha = zeros(m,1);
del_a = zeros(s*blksize, 1);
I = speye(m,m);

% predefine the x vector, the weight vector that is [n,1]
x = zeros(n,1);

iter = 1;
results.alpha = alpha;
results.sol_err = norm(alpha-opt.ref_sol)/norm(opt.ref_sol);
    while(iter <= maxit)
        for i=1:s
            ptr_start = (i-1)*blksize + 1;
            ptr_end = i*blksize;
            idx(ptr_start:ptr_end) = randsample(m,blksize);
            %if((opt.ref_idx(iter-1 + i) - idx(i)) ~= 0)
            %    opt.ref_idx(iter-1 + i) - idx(i)
            %end
            
        end
        idx_overlap = I(:,idx)'*I(:,idx);
        if(strcmpi(opt.kernel, 'poly'))
            v = poly(A(:,idx), A, opt.degree);

            %M = poly(A(:, idx), A(:, idx), opt.degree);
            %M = v(:,idx);
        elseif(strcmpi(opt.kernel, 'linear'))
            v = linear(A(:,idx), A);
            %M = linear(A(:, idx), A(:, idx));
            %M = v(:,idx);

        elseif(strcmpi(opt.kernel, 'gauss'))
            v = gaussian(A(:,idx), A, opt.gamma);
            %M = gaussian(A(:,idx), A(:,idx));
            %M = v(:,idx);
            
        end

        dd =  1/m*(A(:,idx))'*x;
        
        for i=1:s
            ptr_start = (i-1)*blksize + 1;
            ptr_end = i*blksize;
            
            idx_s = idx(ptr_start:ptr_end);
            
            M = v(ptr_start:ptr_end,idx_s);
            T = (1/(lambda*m^2))*M + (eye(blksize)/m);
            
            r = b(idx_s)/m - alpha(idx_s)/m - (1/(lambda*m^2))*v(ptr_start:ptr_end, :)*alpha;
            %r = b(idx_s)/m - alpha(idx_s)/m - dd(ptr_start:ptr_end);
            %disp('correct');
            %disp((1/(lambda*m^2))*v(ptr_start:ptr_end, :)*alpha);
            %disp('incorrect');
            %disp(dd(ptr_start:ptr_end));


            %Update/correct residual from previous solution.
            if(i > 1)
                %v(ptr_start:ptr_end, idx(1:(i-1)*blksize)) - linear(A(:,idx_s), A(:,idx(1:(i-1)*blksize)))
                r = r - (1/(lambda*m^2))*v(ptr_start:ptr_end, idx(1:(i-1)*blksize))*del_a(1:(i-1)*blksize);
                %disp(full(v(ptr_start:ptr_end, idx(1:(i-1)*blksize))));
                %disp(r)
                r = r - idx_overlap(ptr_start:ptr_end, 1:(i-1)*blksize)*del_a(1:(i-1)*blksize)/m;
            end
            %fprintf('residual diff: %0.16g\n', r - opt.ref_r(iter))

            

            del_a(ptr_start:ptr_end) = T\r;
            %disp(del_a)
            %fprintf('del_a diff: %0.16g\n', del_a(i) - opt.ref_del_a(iter))
%            if((del_a(ptr_start:ptr_end) - opt.ref_del_a(iter)) ~= 0)
                %del_a(ptr_start:ptr_end) - opt.ref_del_a(iter)
                %r - opt.ref_r(iter)
%               end
            iter = iter + 1;
            
            if(mod(iter, freq) == 0)
                tmp_alpha = alpha;                
                for j=1:i
                    ptr_start = (j-1)*blksize + 1;
                    ptr_end = j*blksize;
                    idx_s = idx(ptr_start:ptr_end);

                    tmp_alpha(idx_s) = tmp_alpha(idx_s) + del_a(ptr_start:ptr_end);
                end
                results.sol_err(end + 1) = norm(tmp_alpha - opt.ref_sol)/norm(opt.ref_sol);
                %fprintf('soultion error: %0.16g\n', results.sol_err(end))
                fprintf('CA-KRR soultion error (iter %d): %0.16g\n', iter, results.sol_err(end))
                
                
                if(results.sol_err(end) <= tol)
                    results.alpha = tmp_alpha;
                    return;
                end
            end
            
            if(iter > maxit)
                tmp_alpha = alpha;
                
                for j=1:i
                    ptr_start = (j-1)*blksize + 1;
                    ptr_end = j*blksize;
                    idx_s = idx(ptr_start:ptr_end);

                    tmp_alpha(idx_s) = tmp_alpha(idx_s) + del_a(ptr_start:ptr_end);
                end
                results.alpha = tmp_alpha;
                return;
            end
        end
        
        %T = M/lambda/m^2 + (eye(blksize)/m);
        %r = b(idx)/m - alpha(idx)/m - 1/lambda/m^2*v*alpha;
        
        %del_a = T\r;
        %fprintf('del_a diff = %.16g\n', norm(del_a - opt.ref_del_a(iter-s:iter-1)'))
        for i=1:s
            ptr_start = (i-1)*blksize + 1;
            ptr_end = i*blksize;
            idx_s = idx(ptr_start:ptr_end);
            
            alpha(idx_s) = alpha(idx_s) + del_a(ptr_start:ptr_end);
            x = x + 1/(lambda*m) * A(:,idx_s) * del_a(ptr_start:ptr_end);
            % first round x is the same
        end
        
        %fprintf('outer alpha update = %.16g\n', norm(alpha - opt.ref_alpha(iter-1, :)'))
        del_a = zeros(s*blksize, 1);

        %disp('alpha');
        %disp(sum(alpha.^2));
        %disp('x');
        %disp(x);
        
        %if(mod(iter, floor(m/4)) == 0)
        %    norm(alpha - opt.ref_sol)/norm(opt.ref_sol)
        %end
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
