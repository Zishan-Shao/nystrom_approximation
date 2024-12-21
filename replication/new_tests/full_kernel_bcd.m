function alpha = full_kernel_bcd(X, Y, num_epoch, blksize, lambda, gamma, seed)

    rng(seed);

    [n, ~] = size(X);
    k = size(Y, 2);

    % Initialize alpha
    alpha = zeros(n, k);
    %alpha = randn(n, 1) * 0.0001; % Small random init % not good
    perm_idx = randperm(n);

    % Partition data into blocks
    blocks = {};
    for i = 1:blksize:n
        block_end = min(i+blksize-1, n);
        blocks{end+1} = perm_idx(i:block_end);
    end
    num_blks = length(blocks);

    for ell = 1:num_epoch
        fprintf('Epoch %d/%d\n', ell, num_epoch);

        % Randomly permute the block indices
        pi = randperm(num_blks);

        for i = 1:num_blks
            curr_idx = pi(i);
            curr_blk = blocks{curr_idx};
            X_b = X(curr_blk, :);
            Y_b = Y(curr_blk, :);
            b_size = length(curr_blk);

            % Initialize residual
            R = zeros(b_size, k);

            %Make this more efficient, create an index set (multiply entire other)
            %for j = 1:num_blks
            %    if pi(j) ~= curr_idx
            %        other_block_idx = pi(j);
            %        other_block = blocks{other_block_idx};

                    % Compute K between curr_blk and other_block
            %        K_bj = KernelBlock(X_b, X(other_block, :), gamma);

                    % Accumulate R
            %        R = R + K_bj * alpha(other_block, :);
            %    end
            %end
            % Get all indices and subtract the current block's indices
            all_idx = 1:size(X, 1); 
            rest_idx = setdiff(all_idx, blocks{curr_idx});
            
            % Compute the kernel matrix between the current block and all other blocks
            K_bj = KernelBlock(X_b, X(rest_idx, :), gamma);
            
            % Accumulate R in one step
            R = R + K_bj * alpha(rest_idx, :);

            
            % compute K[I,I] matrix
            K_bb = KernelBlock(X_b, X_b, gamma);

            % Solve for alpha_b
            %alpha_b = (K_bb +lambda * speye(b_size)) \ (Y_b - R); % 
            %alpha(curr_blk, :) = alpha_b; 

            % Form the matrix A
            A = K_bb + n*lambda * speye(b_size); % this is an SPD matrix
            
            % Cholesky decomposition
            L = chol(A, 'lower'); % as A is SPD, so it could be decomposed with stability
            
            % Solve using forward and backward substitution
            z = L \ (Y_b - R);          % Forward substitution
            alpha_b = L' \ z;           % Backward substitution
            
            % Store result
            alpha(curr_blk, :) = alpha_b;
        end
    end
end
