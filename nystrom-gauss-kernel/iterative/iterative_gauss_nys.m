function iterative_gauss_nys(X, b, p, c, gamma)
    % X: Full data matrix (n x m), n = number of data points, m = features
    % b: blksize, number of data points to process in each iteration
    % p: Subset size for Nyström approximation
    % c: Oversampling factor
    % gamma: Gaussian kernel parameter

    [n, m] = size(X); % we use n, m in this case because the paper use n,p, while we already used p lol
    
    % Iterate over blocks of the data
    for i = 1:b:b
        % Select block sample_X of size b from the dataset X
        end_idx = min(i + b - 1, n);  %
        sample_X = X(i:end_idx, :);
        blksize = size(sample_X, 1);  
        
        % Select a random subset sample_X_nys of size c * p from the block
        scale_factor = min(c * p, blksize);  % Ensure we don't oversample more than the blksize
        subset_idx = randperm(blksize, scale_factor);  % Randomly select subset indices
        sample_X_nys = sample_X(subset_idx, :); % this is the sub block selected for nystrom from sample_X
        
        % Compute K_I: kernel between all points in sample_X and sample_X_nys
        K_I = zeros(blksize, scale_factor);
        for i_blk = 1:blksize
            for j_blk = 1:scale_factor
                % Compute the Gaussian kernel value
                K_I(i_blk, j_blk) = exp(-gamma * norm(sample_X(i_blk, :) - sample_X_nys(j_blk, :))^2);
            end
        end
        
        % Compute K_II: kernel between points in sample_X_nys
        K_II = zeros(scale_factor, scale_factor);
        for i_subset = 1:scale_factor
            for j_blk = 1:scale_factor
                % Compute the Gaussian kernel value between subset points
                K_II(i_subset, j_blk) = exp(-gamma * norm(sample_X_nys(i_subset, :) - sample_X_nys(j_blk, :))^2);
            end
        end
        
        % Compute the pseudo-inverse of K_II
        K_II_inv = pinv(K_II);
        
        % Approximate the kernel matrix for the current block
        K_block = K_I * K_II_inv * K_I';
        
        % Display or use the approximated kernel block
        disp('Approximated Kernel Block:');
        disp(K_block);
        disp(size(K_block))
        %return K_block;
    end
end
