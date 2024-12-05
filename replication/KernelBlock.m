function K = KernelBlock(X1, X2, gamma)
    % Computes the RBF kernel matrix with gamma as a scaling parameter
    % Inputs:
    %   X1    - n1 x m matrix
    %   X2    - n2 x m matrix
    %   gamma - Kernel scaling parameter
    % Output:
    %   K     - n1 x n2 kernel matrix

    % Compute pairwise squared Euclidean distances
    D = pdist2(X1, X2, 'euclidean').^2;

    % Compute RBF kernel
    K = exp(-gamma * D);
end
