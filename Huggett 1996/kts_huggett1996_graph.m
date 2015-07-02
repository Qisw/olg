function kts_huggett1996_graph(params, result)

    A = params.A;
    ALPHA = params.ALPHA;
    R = params.R;

    agrid = params.agrid;
    s = params.s;
    I = params.I;
    
    mu = params.mu;
    
    figure;
    
    % 생존률    
    subplot(3,3,1);
    plot(1:I+1,s);
    xlabel('age');
    title('Survival rate');    

    % 자본시장 균형    
    subplot(3,3,2);
    plot(result.rmat, result.KSmat); hold on;
    plot(result.rmat, result.KDmat);
    plot(result.rmat, result.BQmat); hold off;
    legend('KS', 'KD', 'BQ');
    xlabel('r');
    title('Equilbrium of Capital Market');
    str1 = ['KS = ', num2str(result.KS)];
    str2 = ['KD = ', num2str(result.KD)];
    str3 = ['BQ = ', num2str(result.BQ)];
    str4 = ['r  = ', num2str(result.r)];
    text(0.02, 100, str1);
    text(0.02,  80, str2);
    text(0.02,  60, str3);
    text(0.02,  40, str4);            
        
    % 나이별 1인당 자본량 및 유산
    KSi = zeros(I,1);
    BQi = zeros(I,1);   
    for i=1:I;
        KSi(i) = s(i)*agrid'*sum(result.savingpop(:,:,i)')';
        BQi(i) = (1-s(i))*agrid'*sum(result.savingpop(:,:,i)')';
    end;

    subplot(3,3,3);
    plot(1:I, KSi); hold on;
    plot(1:I, BQi); hold off;
    legend('Capital per capita by age', 'Bequest per capita by age');
    xlabel('age');
    title('Capital and Bequest per capita by age');
    
    % 각종 통계
   
    KS = result.KS;                                                         % 총자본 공급
    KD = result.KD;                                                         % 총자본 수요
    k = KS/sum(mu);                                                         % 1인당 자본
    
    L = result.L;                                                           % 총노동
    r = result.r;                                                           % 균형이자율
    w = result.w;                                                           % 균형임금
    b = result.b;                                                           % 1인당 연금
    
    Y = A*KS^ALPHA*L^(1-ALPHA);                                             % 총산출
    y = Y/sum(mu);                                                          % 1인당 산출    
    
    share_old = sum(mu(R:I))/sum(mu);                                       % 노인인구 비중
    share_worker = 1 - share_old;                                           % 노동자 비중

    fprintf('========================================\n');
    fprintf('총자본 공급 = %3.4f\n', KS);
    fprintf('총자본 수요 = %3.4f\n', KD);
    fprintf('균형이자율 = %1.6f\n', r);
    fprintf('1인당 자본 = %3.4f\n', k);
    fprintf('========================================\n');   
    fprintf('총노동 = %3.4f\n', L);
    fprintf('균형임금 = %2.4f\n', w);
    fprintf('========================================\n');      
    fprintf('총산출 = %3.4f\n', Y);
    fprintf('1인당 산출 = %2.4f\n', y);
    fprintf('========================================\n');          
    fprintf('1인당 연금 = %2.4f\n', b(I));
    fprintf('노인인구 비율 = %1.2f\n', share_old);
    fprintf('생산가능인구 비율 = %1.2f\n', share_worker);
    fprintf('========================================\n');
    
    
end