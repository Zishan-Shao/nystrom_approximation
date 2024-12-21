function alpha = full_kernel_bcd_sstep(X, Y, num_epoch, blksize, s, lambda, gamma, seed)
    rng(seed);

    [n, ~] = size(X);
    k = size(Y, 2);

    % Initialize alpha
    alpha = zeros(n, k);

    % Randomly permute data indices
    perm_idx = randperm(n);

    % Partition data into blocks
    blocks = {};
    for i = 1:blksize:n
        block_end = min(i+blksize-1, n);
        blocks{end+1} = perm_idx(i:block_end);
    end
    num_blks = length(blocks);

    % Buffer to store block updates for s steps
    del_alpha = cell(s, 1);

    for ell = 1:num_epoch
        fprintf('Epoch %d/%d\n', ell, num_epoch);

        % Random permutation of the block indices
        pi = randperm(num_blks);

        % Process blocks in s steps
        for i = 1:s:num_blks
            % Reset the buffer for current s steps
            for step = 1:s
                del_alpha{step} = zeros(size(alpha));
            end

            % Perform s updates
            for step = 1:s
                if i + step - 1 > num_blks
                    break; % Stop if we go beyond the total blocks
                end

                % Current block indices
                curr_blk_idx = pi(i + step - 1);
                curr_blk = blocks{curr_blk_idx};
                X_b = X(curr_blk, :);
                Y_b = Y(curr_blk, :);
                b_size = length(curr_blk);

                % Initialize residual for this block
                R = zeros(b_size, k);

                % Compute residual from other blocks
                for j = 1:num_blks
                    if pi(j) ~= curr_blk_idx
                        other_blk_idx = pi(j);
                        other_block = blocks{other_blk_idx};

                        % Compute kernel between current block and other block
                        K_bj = KernelBlock(X_b, X(other_block, :), gamma);

                        % Accumulate residual contribution
                        R = R + K_bj * alpha(other_block, :);
                    end
                end

                % Compute local kernel block K_bb
                K_bb = KernelBlock(X_b, X_b, gamma);

                % Solve for delta alpha in the current block
                del_alpha_b = (K_bb + lambda * eye(b_size)) \ (Y_b - R);

                % Store the update for the current block
                del_alpha{step}(curr_blk, :) = del_alpha_b;
            end

            % After s steps, update alpha
            for step = 1:s
                alpha = alpha + del_alpha{step};
            end
        end
    end
end

