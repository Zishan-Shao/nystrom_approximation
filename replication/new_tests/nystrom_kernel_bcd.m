function [alpha, J] = nystrom_kernel_bcd(X, Y, num_epoch, p, blksize, lambda, lambda_p, gamma, seed)
    
    rng(seed);

    n = size(X, 1);    % Num obs
    k = 1;             % Num classes (assume binary in this case)

    % Initialization
    alpha = zeros(p, k);
    R = zeros(n, k); % this is the residual

    % Randomly draw p samples without replacement, p in 1 ... n
    J = randperm(n, p)';

    % Partition J into p/b blocks I_1, ..., I_{p/b}
    num_blocks = p / blksize;
    if mod(p, blksize) ~= 0
        error('p must be divisible by b.');
    end
    I = reshape(J, blksize, num_blocks)';  % I is num_blocks x b

    for epoch = 1:num_epoch
        fprintf('Epoch %d/%d\n', epoch, num_epoch);
        % Randomly permute block indices (pi)
        pi_perm = randperm(num_blocks);

        for i = 1:num_blocks
            % Block indices B
            B = ((pi_perm(i)-1)*blksize + 1) : (pi_perm(i)*blksize);
            alpha_b = alpha(B, :); % Extract block coefficients alpha_b
            I_pi_i = I(pi_perm(i), :)'; 
            S_b = sparse(I_pi_i, 1:blksize, 1, n, blksize); % Selector matrix S_b : [n, b]

            % Compute kernel block K_b : [n, b]
            K_b = KernelBlock(X, X(I_pi_i, :), gamma);
           
            % Update residual 
            % R_n+1 = R_n - (K_b + n * lambda * S_b) alpha_b
            R = R - (K_b + n*lambda_p*S_b) * alpha_b;

            % Subblock K_{bb} = K_b(I_{pi_i}, :)
            K_bb = K_b(I_pi_i, :);
            
            % solve the norm equaltion
            % (K_b^T K_b + n * lambda * K_{bb} + n * lambda' * I_b)
            %T = K_b' * K_b + n*lambda * K_bb + n*lambda_p * eye(blksize);
            %v = K_b' * (Y - R);% K_b^T (Y - R)
            %alpha_b_new = T \ v;
            
            % Form the left-hand side matrix with added regularization
            T = K_b' * K_b + n * lambda * K_bb + n * lambda_p * eye(blksize);
            
            % Cholesky decomposition for stability
            L = chol(T, 'lower');
            z = L \ (K_b' * (Y - R)); 
            %z = L * inv(K_b' * (Y - R)); 
            alpha_b_new = L' \ z;    

            % Update residual
            R = R + (K_b + n*lambda_p*S_b) * alpha_b_new;
            alpha(B, :) = alpha_b_new;
        end
    end
end
