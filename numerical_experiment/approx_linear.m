function [approx_G] = approx_linear(A, c, k, n)

    %disp(size(A))

    % Ensure A is sparse if it's not already
    A = sparse(A);
    
    % define the (n x c) matrix S = 0 n x c
    S = zeros(n, c); % (n x c) sampling matrix

    % define the (c x c) matrix D = 0 c x c
    D = eye(c); % (c x c) rescaling matrix, initialized to identity

    % Compute probabilities for weighted sampling based on the diagonal of A^TA
    sq_sum = diag(A' * A);
    probabilities = sq_sum ./ sum(sq_sum); % normalize to get probabilities

    % Sample columns of A according to the probabilities
    for t = 1:c
        % perform weighted sampling with replacement
        i = randsample(1:n, 1, true, probabilities);
        % define S
        S(i,t) = 1;
        % Update D using the selected index
        pi = probabilities(i);
        D(t,t) = 1 / sqrt(c * pi);
    end 

    % in this case, we want to compute the A A^t, in which the A has many
    % features, so very large n. In terms of KSVM, first A is [1, n] as idx
    % is 1
    % The purpose of S matrix is to sample the number of feature columns,
    % bounded by n for approximation --> in C code, we can indexing to
    % select columns

    % let C = GSD and W = DS^T GSD
    % Compute C, C's dimension is [n, c]
    C = A' * (A * S * D); % so that we computes selected columns and A's dot product

    % Compute W using only the selected columns of A^TA
    % Do not form W_k explicitly. You can just perform C * U * sigma_k_inv * V^T * C^T
    %W = D * ((S' * A') * (A * S)) * D;
    W = D * S' * C; % W's dimension is [c, c]

    % Compute the best rank-k approximation to W using svds
    [U, Sigma, V] = svds(W, k);
    Sigma_k_inv = diag(1 ./ diag(Sigma)); % Invert the top-k singular values

    % Perform the multiplication to get the approximation of G
    approx_G = C * U * Sigma_k_inv * V' * C';

    return;
end