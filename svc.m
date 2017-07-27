function [nsv, alpha, b0] = svc(X,Y,ker,C)
%SVC Support Vector Classification
%
%  Usage: [nsv alpha bias] = svc(X,Y,ker,C)
%
%  Parameters: X      - Training inputs
%              Y      - Training targets
%              ker    - kernel function
%              C      - upper bound (non-separable case)
%              nsv    - number of support vectors
%              alpha  - Lagrange Multipliers
%              b0     - bias term
%
%  Author: Steve Gunn (srg@ecs.soton.ac.uk)

  if (nargin <2 | nargin>4) % check correct number of arguments
    help svc
  else

    fprintf('Support Vector Classification\n')
    fprintf('_____________________________\n')
    n = size(X,1);
    if (nargin<4) C=Inf;, end
    if (nargin<3) ker='linear';, end


    global GPU p1 p2 iter iterTol EPS wrap1 wrap2;
    
    % tolerance for Support Vector Detection
    epsilon = svtol(C);
        
    if GPU == 0,

        % Construct the Kernel matrix
        fprintf('Constructing ...\n');
        H = zeros(n,n);  
        for i=1:n
           for j=1:n
              H(i,j) = Y(i)*Y(j)*svkernel(ker,X(i,:),X(j,:));
           end
        end
        c = -ones(n,1);  

        % Add small amount of zero order regularisation to 
        % avoid problems when Hessian is badly conditioned. 
        H = H+1e-10*eye(size(H));

        % Set up the parameters for the Optimisation problem

        vlb = zeros(n,1);      % Set the bounds: alphas >= 0
        vub = C*ones(n,1);     %                 alphas <= C
        x0 = zeros(n,1);       % The starting point is [0 0 0   0]
        neqcstr = nobias(ker); % Set the number of equality constraints (1 or 0)  
        if neqcstr
           A = Y';, b = 0;     % Set the constraint Ax = b
        else
           A = [];, b = [];
        end

        % Solve the Optimisation Problem

        fprintf('Optimising ...\n');
        st = cputime;
        
        [alpha lambda how] = qp(H, c, A, b, vlb, vub, x0, neqcstr);
    else,
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
          otherwise
            warndlg('This kernel function isn''t implemented!','!! Warning !!');
            return;
        end

        if C == Inf
            gpuC =  -1;
        else
            gpuC = C;
        end
    
        fprintf('Optimising ...\n');
        st = cputime;
        matCudaM3SVM('initial', int32(0));
        [gpuAlpha gpuBias] = matCudaM3SVM('train', single(X), single(Y), int32(size(X,1)), int32(k), single(p1), single(p2), int32(iter), single(iterTol), single(EPS), single(gpuC), int32(wrap1), int32(wrap2));
        matCudaM3SVM('release');
        clear mex;
        
        alpha = double(gpuAlpha);
        how = 'N/A.';
    end;
    
    fprintf('Execution time: %4.1f seconds\n',cputime - st);
    fprintf('Status : %s\n',how);
    % w2 = alpha'*H*alpha;
    % fprintf('|w0|^2    : %f\n',w2);
    % fprintf('Margin    : %f\n',2/sqrt(w2));
    fprintf('Sum alpha : %f\n',sum(alpha));
    
        
    % Compute the number of Support Vectors
    svi = find( alpha > epsilon);
    nsv = length(svi);
    fprintf('Support Vectors : %d (%3.1f%%)\n',nsv,100*nsv/n);

    % Implicit bias, b0
    b0 = 0;

    % Explicit bias, b0 
    % if nobias(ker) ~= 0
      % find b0 from average of support vectors on margin
      % SVs on margin have alphas: 0 < alpha < C
      svii = find( alpha > epsilon & alpha < (C - epsilon));
      if length(svii) > 0
          if GPU == 0,
            b0 =  (1/length(svii))*sum(Y(svii) - H(svii,svi)*alpha(svi).*Y(svii));
          else,
            b0 = double(gpuBias);
          end;
      else 
        fprintf('No support vectors on margin - cannot compute bias.\n');
      end
    % end
    
  end
 
    
