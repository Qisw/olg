function params = kts_huggett1996_setparams(params)

    params.agrid = [params.UNDERA: (params.UPPERA-params.UNDERA)/(params.NA-1): params.UPPERA]';

    params.rgrid = [params.UNDERr: (params.UPPERr-params.UNDERr)/(params.Nr-1): params.UPPERr]';


    params.mu = zeros(params.I,1);
    params.mu(1) = 1;
    for i=1:params.I-1;
        params.mu(i+1) = params.s(i+1)*params.mu(i)*(1+params.N)^(-1);
    end;

    F_e = griddedInterpolant(params.age_e, params.earnings_e);                  % Interpolate earnings by age
    params.earnings = F_e([20:1:98]');
    params.earnings = params.earnings .* (params.earnings > 0);                 % If earning < 0, then set 0.

    F_p = griddedInterpolant(params.age_p, params.earnings_p);                  % Interpolate labor participation rate by age
    params.participation = F_p([20:1:98]');
    params.participation = params.participation .* (params.participation > 0);  % If participation < 0, then set 0.

    params.earningsprofile = params.earnings .* params.participation;           % average earnings by age
    params.ybar = log(params.earningsprofile);                                  % log (average earnings by age)

    
    %% Discretization of z (individual productivity shock)
    
    % (note) Following Tauchen (1986) method. AR(1) process of earnings
    %        shock(z) is discretized into 18-state Markov Chain.
    %        z ~ N(0, SIGMAy^2): initial distribution for z
    %        z' = GAMMA*z + e, e ~ N(0, SIGMAe^2)
    % (note) pi: discretized Markov Matrix
    
    [params.pi, params.p, params.e] = kts_hugget1996_Tauchen(params);
    
    
end