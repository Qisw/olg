function result = kts_huggett1996_newmain(params)

    A = params.A;
    ALPHA = params.ALPHA;
    DELTA = params.DELTA;
    THETA = params.THETA;
    I = params.I;
    R = params.R;
    NA = params.NA;
    NZ = params.NZ;
    ZETA = params.ZETA;

    agrid = params.agrid;
    s = params.s;
    p = params.p;
    e = params.e;
    mu = params.mu;
    rgrid = params.rgrid;
    
    Nr = params.Nr;

    diff_r = zeros(Nr,1);                                                   % (note) diff_r saves the differences between KS and KD for each r on rgrid
    
    KSmat = zeros(Nr,1);
    KDmat = zeros(Nr,1);
    BQmat = zeros(Nr,1);
    rmat  = zeros(Nr,1);
    BQTAXmat = zeros(Nr, 1);

    parfor i_r = 1:Nr;                                                      % paralell computing by individual r
    
        r = rgrid(i_r);
    
        % 총노동공급
        L = 0;
        for i=1:I;
            L = L + (p(:,i)'*e(:,i)*mu(i));                                 % probability of productivity by age x productivity by age x population share by age
        end;
    
        % 총자본수요
        KD = ((r+DELTA)/(ALPHA*A))^(1/(ALPHA-1))*L;
    
        % 임금
        w = ((r+DELTA)/(ALPHA*A))^(ALPHA/(ALPHA-1))*(1-ALPHA)*A;
    
        % 퇴직자에게 주는 연금
        b = zeros(I,1);
        b(1:R-1) = zeros(R-1,1);
        mass_after = [zeros(1,R-1), ones(1, I-R+1)]*mu;                     % 은퇴자(46세 이상, i >= R) 인구 수
        b(R:I) = ones(I-R+1,1)*(THETA*w*L*mass_after^(-1));                 % 은퇴자 1인당 받는 연금액
    
        % 유산 초기값 설정
        BQ0 = 0;
        diff_BQ = 100;
    
        while abs(diff_BQ) > 0.01;                                          % 유산의 초기값과 실제 실현된 유산의 규모가 같아질 때까지 반복
        
            % Value Function Iteration
            
            [savingv, savingg, savingg_ordered] = kts_hugget1996_vf(params, r, w, b, BQ0);    % value function, policy function, ordered policy function
        
            % Monte Carlo Simulation for a cohort
        
            [savingpop] = kts_hugget1996_simulation(params, savingg_ordered);
        
            % Aggregate capital supply (KS) and bequest (BQ)
            KS = 0;
            BQ1 = 0;
            for i=1:I;
                KS = KS + s(i)*agrid'*sum(savingpop(:,:,i)')'*mu(i);
                BQ1 = BQ1 + (1-s(i))*agrid'*sum(savingpop(:,:,i)')'*mu(i);
            end;            
            BQTAX = BQ1*ZETA;
            BQ1 = (1-ZETA)*BQ1;
        
            diff_BQ = BQ1 - BQ0;
        
            fprintf('r = %1.6f, KS = %3.4f, KD = %3.4f, BQ0 = %2.4f, BQ1 = %2.4f, diff_BQ = %2.4f\n', r, KS, KD, BQ0, BQ1, diff_BQ);
        
            BQ0 = BQ1;
        
        end;
    
        % Difference between aggregate capital supply (KS) and demand (KD)    
        diff_r(i_r) = KS - KD;
        
        % 금리별 총자본 공급 및 수요, 유산, 금리를 저장
        KSmat(i_r) = KS;
        KDmat(i_r) = KD;
        BQmat(i_r) = BQ0;
        BQTAXmat(i_r) = BQTAX;
        rmat(i_r)  = r;
    
    end;
    
    
    %% 균형 r 계산: Binary Search Algorithm
       
    for i=1:Nr-1;
        if diff_r(i)*diff_r(i+1) < 0;                                       % 초과공급이 마이너스에서 플러스로 전환되는 시점 포착
           r0 = rgrid(i); 
           r1 = rgrid(i+1);
           diffnew = diff_r(i+1);
           break;
        end;
    end;
    
    it = 0;
    
    while abs(diffnew) > 0.1 && abs(r0-r1) > 0.0001;                        % !!!!!!!!!!!!!!!!! and not or !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
        it = it + 1;
        r = (r0 + r1)*0.5;
                  
        % 총노동공급
        L = 0;
        for i=1:I;
            L = L + (p(:,i)'*e(:,i)*mu(i));                                 % probability of productivity by age x productivity by age x population share by age
        end;
    
        % 총자본수요
        KD = ((r+DELTA)/(ALPHA*A))^(1/(ALPHA-1))*L; % 총자본 수요
    
        % 임금
        w = ((r+DELTA)/(ALPHA*A))^(ALPHA/(ALPHA-1))*(1-ALPHA)*A;
    
        % 퇴직자에게 주는 연금
        b = zeros(I,1);
        b(1:R-1) = zeros(R-1,1);
        mass_after = [zeros(1,R-1), ones(1, I-R+1)]*mu;                     % 은퇴자(46세 이상, i >= R) 인구 수
        b(R:I) = ones(I-R+1,1)*(THETA*w*L*mass_after^(-1));                 % 은퇴자 1인당 받는 연금액
    
        % 유산 초기값 설정
        BQ0 = 0;
        diff_BQ = 100;
    
        while abs(diff_BQ) > 0.01;                                          % 유산의 초기값과 실제 실현된 유산의 규모가 같아질 때까지 반복
        
            % Value Function Iteration
           
            [savingv, savingg, savingg_ordered] = kts_hugget1996_vf(params, r, w, b, BQ0);    % value function, policy function, ordered policy function
        
            % Monte Carlo Simulation for a cohort
        
            [savingpop] = kts_hugget1996_simulation(params, savingg_ordered);
        
            % Aggregate capital supply (KS) and bequest (BQ)
            KS = 0;
            BQ1 = 0;
            for i=1:I;
                KS = KS + s(i)*agrid'*sum(savingpop(:,:,i)')'*mu(i);
                BQ1 = BQ1 + (1-s(i))*agrid'*sum(savingpop(:,:,i)')'*mu(i);
            end;
            BQTAX = BQ1*ZETA;
            BQ1 = (1-ZETA)*BQ1;
        
            diff_BQ = BQ1 - BQ0;
        
            fprintf('r = %1.6f, KS = %3.4f, KD = %3.4f, BQ0 = %2.4f, BQ1 = %2.4f, diff_BQ = %2.4f\n', r, KS, KD, BQ0, BQ1, diff_BQ);
        
            BQ0 = BQ1;
        
        end;          
          
        diffnew = KS - KD;
        
        if diffnew >= 0;
           r1 = r;
        else
           r0 = r;
        end;
        
        fprintf('diffnew = %3.6f, r0 = %1.6f, r1 = %1.6f\n', diffnew, r0, r1);

        KSmat = [KSmat; KS];
        KDmat = [KDmat; KD];
        BQmat = [BQmat; BQ0];
        BQTAXmat = [BQTAXmat; BQTAX];
        rmat  = [rmat; r]; 
               
    end;
    
    % rmat을 기준으로 sorting
    Allmat = [KSmat, KDmat, BQmat, BQTAXmat, rmat];
    Allmat = sortrows(Allmat, 5);
    
    KSmat = Allmat(:,1);
    KDmat = Allmat(:,2);
    BQmat = Allmat(:,3);
    BQTAXmat = Allmat(:,4);
    rmat  = Allmat(:,5);
    
    % 결과물을 구조체에 저장
    field01 = 'savingv';         value01 = savingv;                         % value function
    field02 = 'savingg';         value02 = savingg;                         % policy function
    field03 = 'savingg_ordered'; value03 = savingg_ordered;                 % ordered policy function
    field04 = 'savingpop';       value04 = savingpop;                       % simulation result
    field05 = 'r';               value05 = r;                               % real interest rate in equilibrim
    field06 = 'w';               value06 = w;                               % wage in equilibrium
    field07 = 'b';               value07 = b;                               % pension for retirees
    field08 = 'L';               value08 = L;                               % Aggregate labor supply
    field09 = 'KSmat';           value09 = KSmat;                           % vector of aggregate capital supply
    field10 = 'KDmat';           value10 = KDmat;                           % vector of aggregate capital demand
    field11 = 'BQmat';           value11 = BQmat;                           % vector of aggregate bequest
    field12 = 'BQTAXmat';        value12 = BQTAXmat;                        % vector of bequest tax
    field13 = 'rmat';            value13 = rmat;                            % vector of real interest rate
    field14 = 'KS';              value14 = KS;                              % capital supply in equilibrium
    field15 = 'KD';              value15 = KD;                              % capital demand in equilibrium
    field16 = 'BQ';              value16 = BQ0;                             % bequest in equilibrium
    field17 = 'BQTAX';           value17 = BQTAX;                           % bequest tax
    field18 = 'mu';              value18 = mu;                              % relative population size by age
    
    result = struct(field01, value01, field02, value02, field03, value03, field04, value04, field05, value05, ...
                    field06, value06, field07, value07, field08, value08, field09, value09, field10, value10, ...
                    field11, value11, field12, value12, field13, value13, field14, value14, field15, value15, ...
                    field16, value16, field17, value17, field18, value18);
                    

end