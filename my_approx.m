function [approx_G] = nystrom_approx(G, c, k, n)

    % define the (nxc) matrix S = 0 nxc
    S = zeros(n, c); % (n x c) sampling matrix

    % define the (c x c) matrix D = 0 nxc
    D = eye(c); % (c x c) rescaling matrix, initialized to identity

    % for loop to assign the probability of each feature
    for t = 1:1:c
        % randomly select an indices of 1:n
        i = randi(n); % this is the i_t
        % disp(i)
        % compute the pi (should have 3 options)
        pi = G(i,i)^2 / (sum(diag(G).^2));
        D(t,t) = 1 / sqrt(c * pi);
        %disp(D)
        
        % define S
        S(i,t) = 1;
    end 

    % let C = GSD and W = DS^T GSD
    C = G * S * D;
    W = D * (S' * G * S) * D;

    % Compute W_k, the best rank-k approximation to W
    [U, Sigma, V] = svd(W);
    Sigma_ks = Sigma(1:k, 1:k); %select top k

    % Create a zero matrix with the same size as Sigma
    Sigma_k = zeros(size(Sigma));

    % Place Sigma_ks at the top left of the zero matrix
    Sigma_k(1:k, 1:k) = Sigma_ks;
    
    % Reconstruct the best rank-k approximation to W
    W_k = U * Sigma_k * V';

    approx_G = C * W_k * C';

    return;
    
end