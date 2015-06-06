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

C_test = [.01; .03; .1; .3; 1; 3; 10; 30];
sigma_test = [.01; .03; .1; .3; 1; 3; 10; 30];

pred_error = zeros(size(C_test, 1), size(sigma_test, 1));

% computer predition error for every combination of C and sigma on the test
% set Xval and yval
for i = 1 : size(C_test, 1)
    for j = 1 : size(sigma_test, 1)
        model = svmTrain(X, y, C_test(i), @(x1, x2)...
            gaussianKernel(x1, x2, sigma_test(j)));
        predictions = svmPredict(model, Xval);
        pred_error(i,j) = mean(double(predictions ~= yval));
    end
end

[min_pred, k_idx] = min(pred_error(:));
[C_idx, sigma_idx] = ind2sub(size(pred_error), k_idx);

C = C_test(C_idx);
sigma = sigma_test(sigma_idx);




% =========================================================================

end
