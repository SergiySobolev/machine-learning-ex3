function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); 
J = 0;
grad = zeros(size(theta));
h_theta = sigmoid(X*theta); % h_theta is a size(X,1) by 1 vector

J = (-y' * log(h_theta) - (1-y)' * log(1-h_theta))/m + lambda/2/m*theta(2:end)'*theta(2:end);

grad = X'*(h_theta - y)/m + lambda*theta/m;

grad(1) = grad(1) - lambda*theta(1)/m;

grad = grad(:);

end
