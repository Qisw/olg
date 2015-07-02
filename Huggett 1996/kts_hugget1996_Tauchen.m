function [pi, p, e] = kts_hugget1996_Tauchen(params)

        I      = params.I;
        NZ     = params.NZ;
        GAMMA  = params.GAMMA;
        SIGMAe = params.SIGMAe;
        SIGMAy = params.SIGMAy;       
        
        ybar   = params.ybar;

        zgrid = zeros(NZ, 1);
        range = zeros(NZ+1, 1);
        
        zgrid(1) = -4*SIGMAy;
        
        for j=2:(NZ-1);
            zgrid(j) = zgrid(j-1) + 0.5*SIGMAy;                                     % -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0
        end;                                                                        % 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0
        
        zgrid(NZ) = 6*SIGMAy;
        
        range(1) = -inf;
        
        for j=2:NZ;
            range(j) = (zgrid(j-1) + zgrid(j))*0.5;
        end;
        
        range(NZ+1) = inf;
        
        pi = zeros(NZ, NZ);
        
        for i=1:NZ;
            for j=1:NZ;
                pi(i,j) = normcdf((range(j+1) - GAMMA*zgrid(i))*SIGMAe^(-1), 0, 1) - normcdf((range(j) - GAMMA*zgrid(i))*SIGMAe^(-1), 0, 1);
            end;
        end;
        
        % p(j,i): the probability of being in zgrid(j) when age i
        p = zeros(NZ, I);
        
        for j=1:NZ;
            p(j,1) = normcdf(range(j+1), 0, SIGMAy) - normcdf(range(j), 0, SIGMAy);
        end;
        
        for j=2:I;
            p(:,j) = (p(:,j-1)'*pi)';
        end;
        
        % individual productivity e(j,i)
        e = zeros(NZ, I);
        for j=1:NZ;
            for i=1:I;
                e(j,i) = exp(ybar(i) + zgrid(j));
            end;
        end;
        

end