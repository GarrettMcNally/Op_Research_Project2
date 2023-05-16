function total_returns = mc_sim(mu, Q, num_simulations, num_periods)
    % This function does Monte Carlo simulation of stock returns
    % for the next 'num_periods' periods, based on the expected returns 'mu'
    % and the covariance matrix 'Q'.
 
    % Number of assets
    n = size(mu, 1);
    
    % Simulate stock returns for the next 'num_periods' periods
    sim_returns = zeros(n, num_periods, num_simulations);
    for i = 1:num_simulations
        % Generate random normal returns for the next 'num_periods' periods
        r = mvnrnd(mu, Q, num_periods);

        % Calculate the total returns for each asset
        total_returns(i,:) = prod(1+r)-1;
    end
    
end