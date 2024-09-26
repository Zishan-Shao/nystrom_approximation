function K_approx = nystrom_gauss_kernel(A, p, gamma, c)
    [n, ~] = size(A);
    
    % Adjust the number of sampled points with oversampling factor c
    scale_factor = ceil(p * c); % may not exceed n
    
    % Select scale_factor random subset points
    idx = randperm(n, scale_factor);
    sample_A = A(idx, :);
    
    % Compute K_I (n x scale_factor) - kernel between all points and subset
    K_I = zeros(n, scale_factor);
    % Basically I_jAA^T: exp(-gamma * ||A(i,:) - sample_A(j,:)||^2)
    for i = 1:n
        for j = 1:scale_factor
            K_I(i, j) = exp(-gamma * norm(A(i, :) - sample_A(j, :))^2);
        end
    end
    
    % Compute K_II (scale_factor x scale_factor) in blocks
    K_II = zeros(scale_factor, scale_factor);
    for i = 1:p:scale_factor
        for j = 1:p:scale_factor
            end_i = min(i + p - 1, scale_factor);
            end_j = min(j + p - 1, scale_factor);
            K_II(i:end_i, j:end_j) = block_kernel_comp(sample_A(i:end_i, :), sample_A(j:end_j, :), gamma);
        end
    end
    
    % Apply low-rank SVD for inversion  % The problem is it not always work
    %[U, S, V] = svds(K_II, "econ");
    %S_inv = diag(1 ./ diag(S));  % Invert only the diagonal
    %K_II_inv = V * S_inv * U';
    K_II_inv = pinv(K_II);
    
    % Step 5: Compute Nyström approximation K_approx = K_I * K_II^-1 *
    % K_I', which is blksize * blksize
    K_approx = K_I * K_II_inv * K_I';
end


% Here is the gaussian kernel in K-RR
% In K-RR, we will have this communicated and K_II updated locally each
% iteration
function K_block = block_kernel_comp(sample_A, A, gamma)
    [n1, ~] = size(sample_A);
    [n2, ~] = size(A);
    K_block = zeros(n1, n2);
    for i = 1:n1
        for j = 1:n2
            K_block(i, j) = exp(-gamma * norm(sample_A(i, :) - A(j, :))^2);
        end
    end
end
