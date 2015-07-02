function [savingv, savingg, savingg_ordered] = kts_hugget1996_vf_transition(params, r, w, b, BQ0, savingv_next)

        SIGMA  = params.SIGMA;
        BETA   = params.BETA;
        I      = params.I;
        NA     = params.NA;
        NZ     = params.NZ;
        TAU    = params.TAU;
        THETA  = params.THETA;
        
        agrid = params.agrid;
        mu = params.mu;
        e = params.e;
        s = params.s;
        pi = params.pi;

       %% Finding the value function and the policy function for each age group
        % (note) savingv(:,:,i) represents the value function of age i.
        %        The j_th column of savingv(:,:,i) represents the value function of age i when the productivity shock is in j-th state.
        % (note) savingg(:,:,i) represents the policy function of age i.
        %        The j-th column of savingg(:,:,i) represents the policy function of age i when the productivity shock is in j-th state.
        
        savingv = zeros(NA, NZ, I+1);
        savingg = zeros(NA, NZ, I);
        
        % (I+1)-period value function, I-period policy function
        % I+1기의 경우 사망하므로 value function은 zero가 됨.
        
        savingv(:,:,I+1) = zeros(NA, NZ);
        savingg(:,:,I)   = zeros(NA, NZ);
        
        % I-period value function
        
        c = repmat((1+(1-TAU)*r)*agrid, [1,NZ]) + repmat((1-THETA-TAU)*w*e(:,I)', [NA,1]) + b(I)*ones(NA,18) + BQ0/sum(mu)*ones(NA,18);
        adjc = @(a) a.*(a > 0) + exp(-10^10).*(a <= 0);
        c = adjc(c);
        
        uc = c.^(1-SIGMA)*(1-SIGMA)^(-1);
        
        savingv(:,:,I) = uc;                                                % v(I,t) = uc + BETA*s(I+1)*v(I+1,t+1). By the way, v(I+1) = 0. 
        
        % i-period value function and policy function for i < I
        
        for i=(I-1):-1:1;
            for j=1:NZ;
                c = repmat(((1+(1-TAU)*r)*agrid + ((1-THETA-TAU)*w*e(j,i)+b(i)+BQ0/sum(mu))*ones(NA,1)), [1,NA]) - repmat(agrid', [NA,1]);
                c = adjc(c);
                uc = c.^(1-SIGMA)*(1-SIGMA)^(-1);
                star = uc + BETA*s(i+1)*repmat((pi(j,:)*savingv_next(:,:,i+1)'), [NA,1]);
                [aa, bb] = max(star');
                savingv(:, j, i) = aa';
                savingg(:, j, i) = agrid(bb');
            end;
        end;
        
        % ordered policy function
        
        savingg_ordered = zeros(size(savingg));
        
        for i = 1:I;
            for j = 1:NA;
                for k = 1:NZ;
                    savingg_ordered(j,k,i) = NA-sum(agrid > savingg(j,k,i));
                end;
            end;
        end;


end