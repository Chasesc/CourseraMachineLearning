function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

possibleValues = [0.01 0.03 0.1 0.3 1 3 10 30];

% Default values
bestC = 1; 
bestSigma = 0.3;
bestError = 1;

for i = 1:length(possibleValues)
    for j = 1:length(possibleValues)
        C =  possibleValues(i);
        sigma = possibleValues(j);
        
        % Train the model with the new parameters and calculate the error.
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        
        % Update the parameters if we get a lower error.
        if error < bestError
            bestC = C; bestSigma = sigma; bestError = error;
        end
    end
end

C = bestC;
sigma = bestSigma;

% =========================================================================

end
