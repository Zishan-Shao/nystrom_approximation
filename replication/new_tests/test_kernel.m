%% Kernel comparison
% in this case we compare the matrix directly computed and the nystrom
% matrix
K_true = KernelBlock(A, A, gamma); % ground truth K

% Compute Nystrom kernel approximation
%K_base = KernelBlock(A(J, :), A(J, :), gamma);
%K_base_pinv = pinv(K_base);
%K_nys = KernelBlock(A, A(J, :), gamma) * K_base_pinv * KernelBlock(A(J, :), A, gamma);

K_nm = KernelBlock(A, A(J, :), gamma);  % Kernel between full data and landmarks
K_mm = KernelBlock(A(J, :), A(J, :), gamma) + lambda * eye(length(J));  % Regularized kernel
K_nys = K_nm * (K_mm \ K_nm');


disp("condition of K_b");
disp(cond(K_nm));
%K_nys = KernelBlock(A, A(J, :), gamma) * (K_base \ KernelBlock(A(J, :), A, gamma));
disp("condition of K_nys");
disp(cond(K_nys));

% F-norm difference
F_diff = norm(K_true - K_nys, 'fro');
fprintf('F-Norm Difference of True Kernel v.s Nystrom Kernel: %.16f\n', F_diff);
[m,n] = size(K_nys);
fprintf('Dimension of Kernel: %d, %d\n', m, n);

% Visualize the true and Nystrom kernel matrices 
% see if overall pattern matched up
figure;
subplot(1, 2, 1);
imagesc(K_true);
title('True Kernel Matrix');
colorbar;

subplot(1, 2, 2);
imagesc(K_nys);
title('Nystrom Approximation');
colorbar;