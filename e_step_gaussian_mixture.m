function gamma = e_step_gaussian_mixture(data,pi,mu,sigma)
% Returns a matrix of responsibilities.
%
% @param    data : data matrix n x d with rows as elements of data
% @param    pi   : column vector of probabilities for each class
% @param    mu   : d x k matrix of class centers listed as columns
% @param    sigma: cell array of class covariance matrices (d x d)
%
% @return   gamma: n x k matrix of responsibilities

n = size(data,1);

d = size(data,2);

k = size(mu,2);

gamma = zeros(n,k);

sigma_det = zeros(1,k);

sigma_inverse = zeros(d,d,k);

mul_parameter = (2 * 3.1415926535897932354626)^(0.5 * d);


for i = 1:k
    sigma_det(i) = sqrt(det(sigma(:,:,i)));
    sigma_inverse(:,:,i) = inv(sigma(:,:,i));
end


for i = 1:n
    for j = 1:k
        temp = data(i,:) - mu(:,j)';
        mul_parameter2 = exp(-0.5 * temp * sigma_inverse(:,:,j) * temp')/(mul_parameter * sigma_det(j));
        gamma(i,j) = pi(j) * mul_parameter2;
    end
    gamma(i,:) = gamma(i,:)/sum(gamma(i,:));
end