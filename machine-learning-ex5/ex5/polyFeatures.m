function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

len = length(X);
powers = 1:p; % Create a row vector of elements from 1 to p.
powers = powers(ones(len, 1), :); % Create a matrix with len number of these row vectos

X = X(:, ones(p, 1)); % Make X a matrix with its rows being p copies of the first element

X_poly = X .^ powers;
% =========================================================================

end
