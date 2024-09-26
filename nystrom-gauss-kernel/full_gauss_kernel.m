% This is just the full gaussian kernel in K-RR
function K = full_gauss_kernel(A, gamma)
    
    % Get the number of points (n)
    [n, ~] = size(A);
    
    K = zeros(n, n);
    % full Gaussian kernel matrix
    for i = 1:n
        for j = 1:n
            K(i, j) = exp(-gamma * norm(A(i, :) - A(j, :))^2);
        end
    end
end
