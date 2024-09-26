function iterative_gauss_full(X, b, gamma)
    % X: Full data matrix (n x m), n = number of data points, m = features
    % b: blksize, number of data points to process in each iteration
    % gamma: Gaussian kernel parameter

    [n, m] = size(X); % Get the size of the full data matrix
    
    % Iterate over blocks of the data
    for i = 1:b:b
        % Select block X_block of size b from the dataset X
        end_idx = min(i + b - 1, n);  % Ensure we don't go out of bounds
        X_block = X(i:end_idx, :);
        blksize = size(X_block, 1);  % Actual blksize may be smaller at the last block
        
        % Compute the exact Gaussian kernel matrix for X_block
        K_block = zeros(blksize, blksize);
        for i_blk = 1:blksize
            for j_blk = 1:blksize
                % Compute the Gaussian kernel value for the exact matrix
                K_block(i_blk, j_blk) = exp(-gamma * norm(X_block(i_blk, :) - X_block(j_blk, :))^2);
            end
        end
        
        % Display or use the exact kernel block
        disp('Exact Gaussian Kernel Block:');
        disp(K_block);
        disp(size(K_block))
    end
end
