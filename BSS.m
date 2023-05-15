function  [mu, Q] = BSS_fn(returns, factRet, K)
    
    % Use this function to perform a basic OLS regression with all factors. 
    % You can modify this function (inputs, outputs and code) as much as
    % you need to.
 
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    
    % Number of observations and factors
    [T, p] = size(factRet);
    s = size(returns,2);
    
    % Data matrix
    X = [ones(T,1) factRet];


    % Q setup
    H = (X' * X);
    H = [H zeros(p+1); 
        zeros(p+1,2*(p+1))];
    % Gurobi Setup

    ub = 30;
    lb = -30;

    varTypes = [repmat('C', p+1, 1); repmat('B', p+1, 1)];

    % Beta is within upper and lower bounds and sum binary = k
    A = [eye(p+1) -ub*eye(p+1);
        -eye(p+1) lb*eye(p+1);
        zeros(1,p+1) ones(1,p+1)];

    b=[zeros(1,p+1) zeros(1,p+1) K]';
    lbm = [ones(1,p+1)*lb zeros(1,p+1)]';
    ubm = [ones(1,p+1)*ub ones(1,p+1)]';

    beta_matrix = zeros((p+1), s);
    for i = 1:s
        % Build model for asset i
        clear model;
        model_rets = returns(:,i); 
        
        % Quadratic Objective
        model.Q = sparse(H);

        % Linear Objective
        c = -2*(X' * model_rets);
        c_obj = [c zeros(p+1, 1)];
        model.obj = c_obj;

        % Contraints
        model.A = sparse(A);
        model.rhs = b;

        % Bounds
        model.lb = lbm;
        model.ub = ubm;

        % Inequalities and variable types
        model.sense = repmat('<',2*(p+1)+1,1);
        model.vtype = varTypes;
        
        clear params;
        params.TimeLimit = 100;
        params.OutputFlag = 0;
        results = gurobi(model,params);

        betas = results.x(1:(p+1));
        beta_matrix(:,i) = betas;

    end
    
    B = beta_matrix;

    % Separate B into alpha and betas
    a = B(1,:)';     
    V = B(2:end,:); 
    
    % Residual variance
    ep       = returns - X * B;
    sigma_ep = 1/(T - p - 1) .* sum(ep .^2, 1);
    D        = diag(sigma_ep);
    
    % R^2 calculation
    ybar = mean(returns);
    SSR = sum(((X * B) - ybar).^2);
    SST = sum((returns - ybar).^2);
    r2 = 1 - (SSR./SST);
    adj_r2 = mean(1 - ((1-r2)*(T-1)/(T-p-1)));

    % Factor expected returns and covariance matrix
    f_bar = mean(factRet,1)';
    F     = cov(factRet);
    
    % Calculate the asset expected returns and covariance matrix
    mu = a + V' * f_bar;
    Q  = V' * F * V + D;
    
    % Sometimes quadprog shows a warning if the covariance matrix is not
    % perfectly symmetric.
    Q = (Q + Q')/2;
    
    %----------------------------------------------------------------------
    
end