function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 

% Create the options for the fmincg function.  GradObj means we will
% provide a gradient to the algorithm and MaxIter sets the maximum
% iterations to 50.
options = optimset('GradObj', 'on', 'MaxIter', 50);

for class = 1:num_labels
    % We don't know theta yet, so set it to the zero vector of R^(n+1)
    initial_theta = zeros(n + 1, 1);
    
    % Calculate theta using the advanced optimization algorithm, fmincg.
    % This is essentially a very efficient alternative to gradient descent.
    [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == class), lambda)), ...
        initial_theta, options);        
    
    % Update the value of theta in the class row.
    all_theta(class,:) = theta;      
    
end

% =========================================================================


end
