function  x = TPMVO(mu, Q, TC)
    
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

    % Optimize with CVX
    cvx_begin
        % Variable of length n weights
        variable x(n)

        % Min variance + some trading penalty (L2 norm of weights)
        minimize(x.'*Q*x + TC*norm(x,2))

        subject to
            % Weight must = 1
            ones(1,n)*x == 1;

            % No position over 10%
            x <= 0.10;

            % No shorting
            x >= 0;

            % Portfolio Return must exceed target
            mu.' * x >= targetRet;
            
    cvx_end

  
    %----------------------------------------------------------------------
    
end