function [approx_G] = approx_real_wor(A, c, k, n)
    % Ensure A is sparse if it's not already
    A = sparse(A);

    % define the (n x c) matrix S = 0 n x c
    S = zeros(n, c); % (n x c) sampling matrix

    % define the (c x c) matrix D = 0 c x c
    D = eye(c); % (c x c) rescaling matrix, initialized to identity

    % Compute probabilities for weighted sampling based on the diagonal of A^TA
    sq_sum = diag(A' * A);
    probabilities = sq_sum ./ sum(sq_sum); % normalize to get probabilities

     % NOTE: the randsample does not provide randsample service, so I need
    % to do it myself
    % we use a vector to store history of the selected indices
    history = zeros(1, c);

    for t = 1:c
        % weighted sampling without replacement
        while true
            i = randsample(1:n, 1, true, probabilities);
            if ~ismember(i, history)  % Check if the index was already selected
                break; 
            end
        end
        % Update the selected indices
        history(t) = i;

        % Compute the pi
        %pi = probabilities(i);
        D(t,t) = 1 / sqrt(c * probabilities(i));

        % Define S
        S(i,t) = 1;
    end

    % let C = GSD and W = DS^T GSD
    % Compute C
    C = A' * (A * S * D); % so that we computes selected columns and A's dot product

    % Compute W using only the selected columns of A^TA
    % Do not form W_k explicitly. You can just perform C * U * sigma_k_inv * V^T * C^T
    %W = D * ((S' * A') * (A * S)) * D;
    W = D * S' * C;

    % Compute the best rank-k approximation to W using svds
    [U, Sigma, V] = svds(W, k);
    Sigma_k_inv = diag(1 ./ diag(Sigma)); % Invert the top-k singular values

    % Perform the multiplication to get the approximation of G
    approx_G = C * U * Sigma_k_inv * V' * C';

    return;
end