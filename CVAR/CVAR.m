function [x, optLoss, EWLoss] = CVAR(alpha, sim_returns, returns)

    % Set our target return 
    num_simulations = size(sim_returns, 3);
    target_return = (geomean(returns(:) + 1) - 1);

    % Estimate the geometric mean (for our target return)
    mu = (geomean(sim_returns + 10) - 10)';
    R = ((abs(target_return))+1) * mean(mu); % Target return
    
    % Determine the number of assets and scenarios
    [S, n] = size(sim_returns);
    
    % Define the lower and upper bounds for our portfolio
    lb = [zeros(n, 1); zeros(S, 1); -inf];
    ub = [inf(n, 1); inf(S, 1); inf];


    
    % Define the inequality constraint matrices A and b
    A = [-sim_returns -eye(S) -ones(S,1); -mu' zeros(1,S) 0 ];
    b = [zeros(S, 1); -R];
    
    % Define the equality constraint matrices Aeq and beq (Long-only)
    Aeq = [ones(1, n), zeros(1, S), 0];
    beq = 1;
    
    % Define our objective linear cost function c
    k = (1 / ((1 - alpha) * S));
    c = [zeros(n,1); k * ones(S,1); 1];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 3. Find the optimal portfolio
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    % Set the linprog options to increase the solver tolerance
    options = optimoptions('linprog','TolFun',1e-9);

    % Use 'linprog' to find the optimal portfolio
    y = linprog(c, A, b, Aeq, beq, lb, ub, options);
    
    % Retrieve the optimal portfolio weights
    x = y(1:n);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Calculate the simulated loss distribution for the optimal portfolio
    optLoss = -sim_returns * x;
    
    % Calculate the simulated loss distribution for the equally-weighted 
    % portfolio (for comparison)
    EWLoss = -sim_returns * ones(n,1) / n;
    
    fig1 = figure(1);
    ax1  = subplot(2,1,1);
    histogram(optLoss, 50);
    xlabel('Portfolio losses ($\%$)','interpreter', 'latex','FontSize',14);
    ylabel('Frequency','interpreter','latex','FontSize',14); 
    title('Optimal portfolio','interpreter', 'latex','FontSize',16);
    
    ax2  = subplot(2,1,2);
    histogram(EWLoss, 50);
    xlabel('Portfolio losses ($\%$)','interpreter', 'latex','FontSize',14);
    ylabel('Frequency','interpreter','latex','FontSize',14); 
    title('Equally-weighted portfolio','interpreter', 'latex','FontSize',16);
    
    set(fig1,'Units','Inches', 'Position', [0 0 10, 8]);
    pos2 = get(fig1,'Position');
    set(fig1,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos2(3), pos2(4)])

    
end


