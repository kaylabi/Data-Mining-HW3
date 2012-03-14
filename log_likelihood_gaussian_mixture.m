function ll = log_likelihood_gaussian_mixture(data,mu,sigma,pi)
% Calculates the log likelihood of the data given the parameters of the
% model
%
% @param data   : each row is a d dimensional data point
% @param mu     : a d x k dimensional matrix with columns as the means of
% each cluster
% @param sigma  : a cell array of the cluster covariance matrices
% @param pi     : a column matrix of probabilities for each cluster
%
% @return ll    : the log likelihood of the data (scalar)

n = size(data,1);

d = size(data,2);

k = size(mu,2);

sigma_det = zeros(1,k);

sigma_inverse = zeros(d,d,k);

mul_parameter = (2 * 3.1415926535897932354626)^(0.5 * d);


for i = 1:k
    sigma_det(i) = sqrt(det(sigma(:,:,i)));
    sigma_inverse(:,:,i) = inv(sigma(:,:,i));
end


logLH = zeros(n,k);
for i = 1:n
    for j = 1:k
        temp = data(i,:) - mu(:,j)';
        mul_parameter2 = (exp(-0.5 * temp * sigma_inverse(:,:,j) * temp'))/(mul_parameter * sigma_det(j));
        logLH(i,j) = pi(j) * mul_parameter2;
    end
    logLH1(i,:) = sum(logLH(i,:));
end


inner = zeros(n,1);
for i = 1:n
    inner(i,1) = sum(log(logLH1(i,:)));
end

ll = sum(inner);











