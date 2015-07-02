function [savingpop] = kts_hugget1996_simulation_transition(params, savingg_ordered, savingpop_pre)

        I      = params.I;
        NA     = params.NA;
        NZ     = params.NZ;        
        p = params.p;
        pi = params.pi;
       
        savingpop = zeros(NA, NZ, I);
        savingpop(1,:,1) = p(:,1)';
        
        for i = 2:I;                                                        % ����
            for j = 1:NA;                                                   % �ڻ�
                for k = 1:NZ;                                               % ���꼺
                    pop = zeros(NA, NZ);
                    pop(savingg_ordered(j,k,i-1),:) = savingpop_pre(j,k,i-1)*pi(k,:);
                    savingpop(:,:,i) = savingpop(:,:,i) + pop;
                end;
            end;
        end;


end