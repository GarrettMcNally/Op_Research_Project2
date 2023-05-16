function x = Project2_Function(periodReturns, periodFactRet, x0)

    % Use this function to implement your algorithmic asset management
    % strategy. You can modify this function, but you must keep the inputs
    % and outputs consistent.
    %
    % INPUTS: periodReturns, periodFactRet, x0 (current portfolio weights)
    % OUTPUTS: x (optimal portfolio)
    %
    % An example of an MVO implementation with OLS regression is given
    % below. Please be sure to include comments in your code.
    %
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------

    % Example: subset the data to consistently use the most recent 3 years
    % for parameter estimation
    returns = periodReturns(end-35:end,:);
    factRet = periodFactRet(end-35:end,:);
    
    % Example: Use an OLS regression to estimate mu and Q
    [mu, Q] = LASSO(returns, factRet, 4);
    
    num_simulations = 10000;
    num_periods = 12;
    sim_returns = mc_sim(mu, Q, num_simulations, num_periods);

    
    alpha = 0.90; % set the confidence level for CVaR
    target_return = (geomean(returns(:,1) + 1) - 1);
    mu = ( geomean(returns + 1) - 1 )';
    R = ((target_return) * mean(mu)) + mean(mu); % Target return
    [x, optLoss, EWLoss] = CVAR(alpha, sim_returns, returns);

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

    
    print(fig1,'CVar_vs_EW_final_iteration','-dpng','-r0');

    % display(z_opt)
    % display(gamma_opt)

    
    %----------------------------------------------------------------------
end
