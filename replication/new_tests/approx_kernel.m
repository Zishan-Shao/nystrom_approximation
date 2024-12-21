function [K_nys_improved] = approx_kernel(A, gamma, c, k)
    % A: Data matrix (n x m)
    % gamma: Kernel parameter
    % c: Number of sampled columns
    % k: Rank of approximation
    
    [n, ~] = size(A);

    % Ensure A is sparse if it's not already
    A = sparse(A);

    % Step 1: Initialize sampling and rescaling matrices
    S = zeros(n, c); % Sampling matrix (n x c)
    D = eye(c);      % Rescaling matrix (c x c)

    % Step 2: Compute probabilities for weighted sampling
    sq_sum = sum(A.^2, 2); % Squared row norms of A
    probabilities = sq_sum ./ sum(sq_sum); % Normalize to probabilities

    % Step 3: Sample columns of A based on probabilities
    for t = 1:c
        i = randsample(1:n, 1, true, probabilities); % Weighted sampling
        S(i, t) = 1; % Define sampling matrix S
        D(t, t) = 1 / sqrt(c * probabilities(i)); % Update rescaling matrix D
    end

    % Step 4: Compute C = A * S * D (selected columns of A's kernel product)
    K_A = KernelBlock(A, A, gamma); % Full kernel matrix for A
    C = K_A * S * D; % C has dimensions (n x c)

    % Step 5: Compute W = D * S' * C (kernel approximation of sampled block)
    W = D * S' * C; % W has dimensions (c x c)

    % Step 6: Compute rank-k approximation of W using svds
    [U, Sigma, V] = svds(W, k);
    Sigma_k_inv = diag(1 ./ diag(Sigma)); % Invert top-k singular values

    % Step 7: Compute the final kernel approximation
    K_nys_improved = C * U * Sigma_k_inv * V' * C';

    return;
end
