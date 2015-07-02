function [savingpop] = kts_hugget1996_simulation_transition(params, savingg_ordered, savingpop_pre)

        I      = params.I;
        NA     = params.NA;
        NZ     = params.NZ;        
        p = params.p;
        pi = params.pi;
       
        savingpop = zeros(NA, NZ, I);
        savingpop(1,:,1) = p(:,1)';
        
        for i = 2:I;                                                        % 나이
            for j = 1:NA;                                                   % 자산
                for k = 1:NZ;                                               % 생산성
                    pop = zeros(NA, NZ);
                    pop(savingg_ordered(j,k,i-1),:) = savingpop_pre(j,k,i-1)*pi(k,:);
                    savingpop(:,:,i) = savingpop(:,:,i) + pop;
                end;
            end;
        end;


end