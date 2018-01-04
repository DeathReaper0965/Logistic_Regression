function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

x1 = X(:, 1);
x2 = X(:, 2:end);

h = sigmoid(X * theta);
shift = theta(2:size(theta));
theta_reg = [0;shift];

J = ((1/m) * ((-y' * log(h)) - ((1-y)' * log(1-h)))) + ((lambda / (2*m))*(theta_reg' * theta_reg));

grad(1) = (1/m) * x1' * (h-y);
grad(2:end) = ((1/m) * x2' * (h-y)) + (lambda/m) * theta(2:end);

end
