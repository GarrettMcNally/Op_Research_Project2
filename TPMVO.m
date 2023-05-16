function  x = TPMVO_fn(mu, Q, TC)
    
    % Use this function to construct an example of a MVO portfolio.
    %
    % An example of an MVO implementation is given below. You can use this
    % version of MVO if you like, but feel free to modify this code as much
    % as you need to. You can also change the inputs and outputs to suit
    % your needs. 
    
    % You may use quadprog, Gurobi, or any other optimizer you are familiar
    % with. Just be sure to include comments in your code.

    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    
    % Find the total number of assets
    n = size(Q,1); 
    
    % Set the target as the average expected return of all assets
    targetRet = mean(mu);
    
    % Q Refactor
    Q = [Q zeros(n); 
        zeros(n,2*(n))];

        % Gurobi Setup

    ub = 0.25;
    lb = 0;

    varTypes = [repmat('C', n, 1); repmat('B', n, 1)];

    % Beta is within upper and lower bounds target return, weights = 1
    A = [eye(n) -ub*eye(n);
        -eye(n) lb*eye(n)
        ones(1,n) zeros(1,n)
        mu' zeros(1,n)];

    b=[zeros(1,n) zeros(1,n) 1 targetRet]';
    lbm = [ones(1,n)*lb zeros(1,n)]';
    ubm = [ones(1,n)*ub ones(1,n)]';

    clear model;    
    % Quadratic Objective
    model.Q = sparse(Q);

    % Linear Objective
    c_obj = [zeros(n, 1) (1/TC.^2)*ones(n, 1)];
    model.obj = c_obj;

    % Contraints
    model.A = sparse(A);
    model.rhs = b;

    % Bounds
    model.lb = lbm;
    model.ub = ubm;

    % Inequalities and variable types
    model.sense = [repmat('<',2*(n),1)' repmat('=',1,1) repmat('>',1,1)];
    model.vtype = varTypes;
    
    clear params;
    params.TimeLimit = 100;
    params.OutputFlag = 0;
    results = gurobi(model,params);
    display(results.x)
    x = results.x(1:n);


    
    %----------------------------------------------------------------------
    
end