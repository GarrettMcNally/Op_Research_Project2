function x = Equally_Weighted(mu, Q)

    num_assets = size(Q,1);
    x = ones(num_assets,1) * (1/num_assets);
    
end
