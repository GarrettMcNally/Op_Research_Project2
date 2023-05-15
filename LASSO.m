function  [mu, Q] = Lasso_fn(returns, factRet, lambda)
    
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
    
    % Set bounds to deal with abs in Lasso %
    lb = zeros((p+1)*2,1);
    A = ones(1,(p+1)*2);
    b = ones(1,1) * 1/lambda;

    % Set H for optimization in Quadprog %
    H_1 = 2* (X' * X);
    H = [H_1, -H_1;
        -H_1, H_1];

    options = optimset('Display', 'off');
    beta_matrix = zeros((p + 1)*2, s);
    for i = 1:s
        % get rets for asset i and optimize for beta %
        model_rets = returns(:,i);        
        f_1 = -2 * (X' * model_rets);
        f = [f_1, -f_1];
        
        betas = quadprog(H, f, A, b, [], [], lb, [], [], options);
        beta_matrix(:,i) = betas;
    end
    
    B = beta_matrix(1:p+1,:)-beta_matrix(p+2:end,:);
    
    p = sum(abs(round(B,4))>0);


    % Separate B into alpha and betas
    a = B(1,:)';     
    V = B(2:end,:); 
    
    % Residual variance
    ep       = returns - X * B;
    sigma_ep = 1./(T - p - 1) .* sum(ep .^2, 1);
    D        = diag(sigma_ep);

    % R^2 calculation
    ybar = mean(returns);
    SSR = sum((returns - (X * B)).^2);
    SST = sum((returns - ybar).^2);
    r2 = 1 - (SSR./SST);
    adj_r2 = mean(1 - ((1-r2)*(T-1)./(T-p-1)));

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