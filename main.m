% [alpha bias w2] = matCudaMPSVM('train', single([Xnorm1]), single([Y1]), int32([size(Xnorm1,1)]), int32([0]), single([1]), single([0]), int32(1000), single(1e-5), single(1e-5), single(CUDA_MAX));
clear all;
CUDA_MIN = 1.17549e-38;
CUDA_MAX = 1.70141e+38;
CUDA_EPSILON = 1.19209e-07;        

fcv = 2;
load wine3.mat;
p1 = 1;
p2 = 0;
ker = 'rbf';

errList =[];

for idxvalue = 1:fcv
    idx = num2str(idxvalue);
    
    switch lower(ker)
      case 'linear'
        k = 0;
      case 'poly'
        k = 1;
      case 'rbf'
        k = 2;
      case 'erbf'
        k = 3;
      case 'sigmoid'
        k = 4;      
    end

    feature1 = full(features(find(full(labels(:))==1),:));
    feature2 = full(features(find(full(labels(:))==-1),:));

    training_X1 = [];
    training_X2 = [];

    for i = 1:fcv
        if i ~= str2num(idx)
            training_X1 = [training_X1; feature1(i:fcv:size(feature1,1),:)];
            training_X2 = [training_X2; feature2(i:fcv:size(feature2,1),:)];
        end
    end

    training_X = [training_X1; training_X2];
    training_label = [ones(size(training_X1,1),1); -ones(size(training_X2,1),1)];
    training_coeff = [-ones(size(training_X1,1),1); -ones(size(training_X2,1),1)];

    testing_X1 = [];
    testing_X2 = [];
    testing_X1 = [testing_X1; feature1(i:fcv:size(feature1,1),:)];
    testing_X2 = [testing_X2; feature2(i:fcv:size(feature2,1),:)];

    testing_X = [testing_X1; testing_X2];
    testing_label = [ones(size(testing_X1,1),1); -ones(size(testing_X2,1),1)];

    % X1 = X1-ones(size(X1,1),1)*mean(X1);
    % X2 = X2-ones(size(X2,1),1)*mean(X2);

    matCudaM3SVM('initial', int32(0));
    [alpha beta] = matCudaM3SVM('train', single(training_X), single(training_label), single(training_coeff), int32(size(training_X,1)), int32(k), single(p1), single(p2), int32(1000), single(1e-3), single(1e-3), single(CUDA_MAX));
    result = double(matCudaM3SVM('predict', single(testing_X), int32(size(testing_X,1)), single(training_X), single(training_label), single(alpha), single(beta), int32(size(training_X,1)), int32(k), single(p1), single(p2), single(CUDA_MIN)));
    matCudaM3SVM('release');

    err = size(find(testing_label ~= sign(result)),1)/size(result(:), 1);
    errList = [errList; err];
end

fprintf('Accuracy: %f(percent)\n', 100*(1-mean(errList)));