function [mumat, smat, popsize, Nmat] = kts_huggett1996_pop_transition(params0, params1, NP)
                  
    mu0 = params0.mu;
    s0  = params0.s;
    N0  = params0.N;
    I0   = params0.I;

    mu1 = params1.mu;
    N1  = params1.N;
    
    Nmat = ones(I0,NP)*N0;
    for t=1:NP;
        for i=1:I0;
            if i < t;
               Nmat(i,t) = N1;
            end;
        end;
    end;
    
    smat = [s0, repmat(s0, 1, NP-1)];
    mumat = zeros(size(mu0,1), NP);

    popsize = zeros(I0, NP);

    for t=1:NP;
        
        if t == 1; 
           popsize(1, t) = 1;
        else
           popsize(1, t) = popsize(1, t-1)*(1 + Nmat(1, t));
        end;
        
        for i=2:I0;
            popsize(i, t) = popsize(i-1, t)/(1 + Nmat(i, t))*smat(i, t);
        end;
        
    end;    

    for t=1:NP;
        mumat(:, t) = popsize(:, t)/popsize(1, t);
    end;
         
end
