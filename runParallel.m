
load DataSetB.mat;
Xnorm1 = svdatanorm(X,ker);
Y1 = Y;

load DataSetA.mat;
Xnorm2 = svdatanorm(X,ker);
Y2 = Y;

load DataSetB.mat;
Xnorm3 = svdatanorm(X,ker);
Y3 = Y;

load DataSetA.mat;
Xnorm4 = svdatanorm(X,ker);
Y4 = Y;

load DataSetB.mat;
Xnorm5 = svdatanorm(X,ker);
Y5 = Y;

load DataSetA.mat;
Xnorm6 = svdatanorm(X,ker);
Y6 = Y;

load DataSetB.mat;
Xnorm7 = svdatanorm(X,ker);
Y7 = Y;

load DataSetA.mat;
Xnorm8 = svdatanorm(X,ker);
Y8 = Y;

load DataSetA.mat;
Xnorm9 = svdatanorm(X,ker);
Y9 = Y;

load DataSetA.mat;
Xnorm10 = svdatanorm(X,ker);
Y10 = Y;

% CUDA_MIN = 1.17549e-38;
% CUDA_MAX = 1.70141e+38;
% CUDA_EPSILON = 1.19209e-07;        

matCudaM3SVM('initial', int32(0));
[alpha bias] = matCudaM3SVM('train', single([Xnorm1]), single([Y1]), int32([size(Xnorm1,1)]), int32([0]), single([1]), single([0]), int32(1000), single(0.001), single(0.001), single(-1), int32(5), int32(0));

tic;
[alpha bias] = matCudaM3SVM('train', single([Xnorm1]), single([Y1]), int32([size(Xnorm1,1)]), int32([0]), single([1]), single([0]), int32(1000), single(0.001), single(0.001), single(-1), int32(5), int32(0));
toc;
drawnow;

Xnorm1x = [];
Y1x = [];
sizeXnorm1x = [];
aa = [];
bb = [];
cc = [];
for i = 1:100,
    Xnorm1x = [Xnorm1x; Xnorm1];
    Y1x = [Y1x; Y1];    
    sizeXnorm1x = [sizeXnorm1x; size(Xnorm1,1)];
    aa = [aa; 0];
    bb = [bb; 1];
    cc = [cc; 0];
end;
tic;
[alpha bias] = matCudaM3SVM('train', ...
    single(Xnorm1x), ...
    single(Y1x), ...
    int32(sizeXnorm1x), ...
    int32(aa), ...
    single(bb), ...
    single(cc), ...
    int32(1000), single(0.001), single(0.001), single(-1), int32(5), int32(0));
toc;
drawnow;
tic;
for i = 1:100
    [alpha bias] = matCudaM3SVM('train', single([Xnorm1]), single([Y1]), int32([size(Xnorm1,1)]), int32([0]), single([1]), single([0]), int32(1000), single(0.001), single(0.001), single(-1), int32(5), int32(0));
end
toc;
drawnow;

%alpha = double(alpha);
%bias = double(bias);

%figure; svcplot(Xnorm1,Y1,'linear',alpha(1:size(Y1,1),:),bias(1));
%figure; svcplot(Xnorm2,Y2,'rbf',alpha(size(Y1,1)+1:size(Y1,1)+size(Y2,1),:),bias(2));

matCudaM3SVM('release');
clear mex;


