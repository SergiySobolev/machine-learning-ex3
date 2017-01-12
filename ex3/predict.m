function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

inputExamplesNum = size(X,1);
hiddenLayerDim = size(Theta1, 1);
hiddenLayer = zeros(inputExamplesNum, hiddenLayerDim+1);  
for j=1:inputExamplesNum
    jInputExample = [1,X(j, :)];
    jHiddenLayer = sigmoid([1,jInputExample * Theta1']);
    hiddenLayer(j,:) = jHiddenLayer;   
    jOutputLayer = sigmoid(jHiddenLayer*Theta2');
    p(j) = find(jOutputLayer == (max(jOutputLayer)));
end;
% 
% p1 = zeros(size(X, 1), 1);
% a1 = [ones(m, 1) X];
% z2 = a1 * Theta1';
% a2 = [ones(size(z2), 1) sigmoid(z2)];
% z3 = a2 * Theta2';
% a3 = sigmoid(z3);
% 
% [val, p1] = max(a3, [], 2);








end
