function kts_huggett1996_graph(params, result)

    A = params.A;
    ALPHA = params.ALPHA;
    R = params.R;

    agrid = params.agrid;
    s = params.s;
    I = params.I;
    
    mu = params.mu;
    
    figure;
    
    % ������    
    subplot(3,3,1);
    plot(1:I+1,s);
    xlabel('age');
    title('Survival rate');    

    % �ں����� ����    
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
        
    % ���̺� 1�δ� �ں��� �� ����
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
    
    % ���� ���
   
    KS = result.KS;                                                         % ���ں� ����
    KD = result.KD;                                                         % ���ں� ����
    k = KS/sum(mu);                                                         % 1�δ� �ں�
    
    L = result.L;                                                           % �ѳ뵿
    r = result.r;                                                           % ����������
    w = result.w;                                                           % �����ӱ�
    b = result.b;                                                           % 1�δ� ����
    
    Y = A*KS^ALPHA*L^(1-ALPHA);                                             % �ѻ���
    y = Y/sum(mu);                                                          % 1�δ� ����    
    
    share_old = sum(mu(R:I))/sum(mu);                                       % �����α� ����
    share_worker = 1 - share_old;                                           % �뵿�� ����

    fprintf('========================================\n');
    fprintf('���ں� ���� = %3.4f\n', KS);
    fprintf('���ں� ���� = %3.4f\n', KD);
    fprintf('���������� = %1.6f\n', r);
    fprintf('1�δ� �ں� = %3.4f\n', k);
    fprintf('========================================\n');   
    fprintf('�ѳ뵿 = %3.4f\n', L);
    fprintf('�����ӱ� = %2.4f\n', w);
    fprintf('========================================\n');      
    fprintf('�ѻ��� = %3.4f\n', Y);
    fprintf('1�δ� ���� = %2.4f\n', y);
    fprintf('========================================\n');          
    fprintf('1�δ� ���� = %2.4f\n', b(I));
    fprintf('�����α� ���� = %1.2f\n', share_old);
    fprintf('���갡���α� ���� = %1.2f\n', share_worker);
    fprintf('========================================\n');
    
    
end