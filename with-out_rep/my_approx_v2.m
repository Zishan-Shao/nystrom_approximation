function [approx_G] = my_approx_v2(G, c, k, n)


    % define the (n x c) matrix S = 0 n x c
    S = zeros(n, c); % (n x c) sampling matrix

    % define the (c x c) matrix D = 0 c x c
    D = eye(c); % (c x c) rescaling matrix, initialized to identity

    % for loop to precompute the probability of each feature
    probabilities = diag(G).^2 / sum(diag(G).^2); % compute probabilities once
    for t = 1:1:c
        % perform weighted sampling with or without replacement
        i = randsample(1:n, 1, true, probabilities);
        % compute the pi
        pi = probabilities(i);
        D(t,t) = 1 / sqrt(c * pi);
        % define S
        S(i,t) = 1;
    end 

    % let C = GSD and W = DS^T GSD
    C = G * S * D;
    
    % replace svd with svds. You can pass 'k' as input parameter to only compute the top-k singular values

    % Do not form W_k explicitly. You can just perform C * U * sigma_k_inv * V^T * C^T

    W = D * (S' * G * S) * D;

    % Compute W_k, the best rank-k approximation to W
    [U, Sigma, V] = svds(W,k);
    %disp(Sigma);
    Sigma_k_inv = diag(1 ./ diag(Sigma)); %select top k
    

    % Avoid forming W_k explicitly and perform the multiplication directly
    approx_G = C * U * Sigma_k_inv * V' * C';

    return;
end