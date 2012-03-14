function [mu,sigma,pi] = m_step_gaussian_mixture(data,gamma)
% Performs the M-step of the EM algorithm for gaussain mixture model.
%
% @param data   : n x d matrix with rows as d dimensional data points
% @param gamma  : n x k matrix of resposibilities
%
% @return mu    : d x k matrix of maximized cluster centers
% @return sigma : cell array of maximized 
%

n = size(data,1);

d = size(data,2);

k = size(gamma,2);

mu = zeros(d,k);

pi = ones(1,k);

sigma = zeros(d,d,k);

Number_k = zeros(1,k);


for i = 1:k
    for j = 1:n
        Number_k(i) = Number_k(i) + gamma(j,i);
        mu(:,i) = mu(:,i) + gamma(j,i) * data(j,:)';
    end
    mu(:,i) = mu(:,i) / Number_k(i);
end


for i = 1:k
    for j = 1:n
        pi(i) = pi(i) + gamma(j,i);
    end
end
pi = pi / n;


for i = 1:k
    for j = 1:n
        temp2 = data(j,:) - mu(:,i)';
        sigma(:,:,i) = sigma(:,:,i) + gamma(j,i).*(temp2' * temp2);
    end
    sigma(:,:,i) = sigma(:,:,i)/Number_k(i);
end


for i = 1:k
    [~ ,p] = chol(sigma(:,:,i));
    while p ~= 0
        sigma(:,:,i) = sigma(:,:,i) + eye(d) * (0.001);
        [~ ,p] = chol(sigma(:,:,i));
    end
end

    
    
    
    
    
    