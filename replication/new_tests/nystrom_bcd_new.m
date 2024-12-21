function [alpha_full, J] = nystrom_bcd_new(X, Y, num_epoch, p, blksize, lambda, lambda_p, gamma, seed)
    rng(seed);

    n = size(X, 1);  % Number of observations
    k = size(Y, 2);  % Number of outputs/classes

    % Initialization
    alpha = zeros(p, k);  % Reduced alpha (landmarks only)
    R = zeros(n, k);  % Residual
    J = randperm(n, p)';  % Sample p landmarks without replacement

    % Partition J into blocks
    num_blocks = p / blksize;
    if mod(p, blksize) ~= 0
        error('p must be divisible by block size.');
    end
    I = reshape(J, blksize, num_blocks)';  % Blocks of landmark indices

    for epoch = 1:num_epoch
        fprintf('Epoch %d/%d\n', epoch, num_epoch);
        pi_perm = randperm(num_blocks);  % Randomly permute block indices

        for i = 1:num_blocks
            B = ((pi_perm(i)-1)*blksize + 1):(pi_perm(i)*blksize);
            alpha_b = alpha(B, :);
            I_pi_i = I(pi_perm(i), :)';
            S_b = sparse(I_pi_i, 1:blksize, 1, n, blksize);

            % Compute kernel blocks
            K_b = KernelBlock(X, X(I_pi_i, :), gamma);  % [n x b]
            K_bb = K_b(I_pi_i, :);  % Subblock [b x b]

            % Update residual
            R = R - (K_b + n * lambda_p * S_b) * alpha_b;

            % Solve for alpha_b
            left_matrix = K_b' * K_b + n * lambda * K_bb + n * lambda_p * eye(blksize);
            right_vec = K_b' * (Y - R);
            alpha_b_new = left_matrix \ right_vec;

            % Update residual
            R = R + (K_b + n * lambda_p * S_b) * alpha_b_new;
            alpha(B, :) = alpha_b_new;
        end
    end

    % Expand alpha to full dimension
    K_nm = KernelBlock(X, X(J, :), gamma);  % Kernel between all data and landmarks
    alpha_full = K_nm * alpha;  % Expand reduced alpha to full dimension

    % Return full-dimension alpha
end
