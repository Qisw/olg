clc;
clear;
close all;

%--------------------------------------------------------------------------
% Huggett (1996)  "Wealth distribution in life-cycle economies," Journal of
% Monetary Economics, 38(3), 469-494
%--------------------------------------------------------------------------

tic;

%% parameters

field01 = 'SIGMA';      value01  = 1.5;                                     % risk aversion
field02 = 'BETA';       value02  = 0.994;                                   % time preference
field03 = 'ALPHA';      value03  = 0.36;                                    % income share of capital
field04 = 'DELTA';      value04  = 0.06;                                    % depreciation rate
field05 = 'N';          value05  = 0.012;                                   % population growth
field06 = 'I';          value06  = 79;                                      % life span
field07 = 'R';          value07  = 46;                                      % retirement age
field08 = 'UNDERA';     value08  = 0;                                       % lower bound of asset
field09 = 'UPPERA';     value09  = 50;                                      % upper bound of asset
field10 = 'NA';         value10 = 50;                                      % number of grids for asset
field11 = 'UNDERr';     value11 = 0;                                        % lower bound of r: -DELTA
field12 = 'UPPERr';     value12 = 0.2;                                      % upper bound of r: "1/(s*BETA)-1" but s is not scalar but vector.
field13 = 'Nr';         value13 = 6;                                        % number of grids for real interest rate
field14 = 'NZ';     	value14 = 18;                                       % number of grids for idiosyncratic productivity shocks
field15 = 'A';          value15 = 1;                                        % TFP
field16 = 'GAMMA';      value16 = 0.96;                                     % Tauchen parameters
field17 = 'SIGMAe';     value17 = 0.045^0.5;                                % Tauchen parameters
field18 = 'SIGMAy';     value18 = 0.38^0.5;                                 % standard deviation of productivity shock z
field19 = 'TAU';        value19 = 0.2378;                                   % 소득세율 (자본소득 및 노동소득)
field20 = 'THETA';      value20 = 0.1;                                      % 연금세율 (노동소득)
field21 = 'ZETA';       value21 = 1.0;                                      % 상속세율
field22 = 'age_e';      value22 = [20, 35, 55, 70]';                        % 나이
field23 = 'earnings_e'; value23 = [0.11931, 0.41017, 0.47964, 0.23370]';    % 연령별 소득(백만달러)
field24 = 'age_p';      value24 = [22.5, 30, 40, 50, 60, 70]';              % 나이
field25 = 'earnings_p'; value25 = [0.796, 0.919, 0.919, 0.875, 0.687, 0.19]';  % 연령별 노동시장참가율
field26 = 's';          value26 = [1.00000; 0.99962; 0.99960; 0.99958; 0.99956; 0.99954; 0.99952; 0.99950; 0.99947; 0.99945;    % survival rate
                                   0.99942; 0.99940; 0.99938; 0.99934; 0.99930; 0.99925; 0.99919; 0.99910; 0.99899; 0.99887;
                                   0.99875; 0.99862; 0.99848; 0.99833; 0.99816; 0.99797; 0.99775; 0.99753; 0.99731; 0.99708;
                                   0.99685; 0.99659; 0.99630; 0.99599; 0.99566; 0.99529; 0.99492; 0.99454; 0.99418; 0.99381;
                                   0.99340; 0.99291; 0.99229; 0.99150; 0.99057; 0.98952; 0.98841; 0.98719; 0.98582; 0.98422;
                                   0.98241; 0.98051; 0.97852; 0.97639; 0.97392; 0.97086; 0.96714; 0.96279; 0.95795; 0.95241;
                                   0.94646; 0.94005; 0.93274; 0.92434; 0.91518; 0.90571; 0.89558; 0.88484; 0.87352; 0.86166;
                                   0.84930; 0.83652; 0.82338; 0.80997; 0.79638; 0.78271; 0.76907; 0.75559; 0.74239; 0.00000];
                               
field27 = 'agrid';              value27 = [];
field28 = 'mu';                 value28 = [];

field29 = 'earnings';           value29 = [];
field30 = 'participation';      value30 = [];
field31 = 'earningsprofile';    value31 = [];
field32 = 'ybar';               value32 = [];

field33 = 'pi';                 value33 = [];
field34 = 'p';                  value34 = [];
field35 = 'e';                  value35 = [];

field36 = 'rgrid';              value36 = [];

params = struct(field01, value01, field02, value02, field03, value03, field04, value04, field05, value05, ...
                field06, value06, field07, value07, field08, value08, field09, value09, field10, value10, ...
                field11, value11, field12, value12, field13, value13, field14, value14, field15, value15, ...
                field16, value16, field17, value17, field18, value18, field19, value19, field20, value20, ...
                field21, value21, field22, value22, field23, value23, field24, value24, field25, value25, ...
                field26, value26, field27, value27, field28, value28, field29, value29, field30, value30, ...
                field31, value31, field32, value32, field33, value33, field34, value34, field35, value35, ...
                field36, value36);

params = kts_huggett1996_setparams(params);                                 % 이미 설정된 parameters를 이용하여 비어있는 나머지 parameters를 설정하는 함수


%% Finding Steady State - params0: N = +0.012, params1: N = -0.012

% 인구증가율: N = +0.012
tic;
params0 = params;

result0 = kts_huggett1996_newmain(params0);                                 % 인구증가율: +1.2%
save result0;
save param0;
kts_huggett1996_graph(params0, result0);
toc;


% 인구증가율: N = -0.012
tic;
params1 = params;

params1.N = -0.012;
params1 = kts_huggett1996_setparams(params1);                               % 인구증가율 N을 바꾸었으므로 mu 등 N의 변화에 의존하는 변수들을 다시 계산해야 함

result1 = kts_huggett1996_newmain(params1);                                 % 인구증가율: -1.2%
save result1;
save params1;
kts_huggett1996_graph(params1, result1);
toc;



%% Population Transition

load('result0.mat');
load('result1.mat');
load('params0.mat');
load('params1.mat');

NP = 160;                                                                               % transition 기간 - 160기 후에는 새로운 steady state에 들어갈 것으로 예상

[mumat, smat, popsize, Nmat] = kts_huggett1996_pop_transition(params0, params1, NP);    % 인구구조 변화 - N: +0.012 => -0.012
                                                                                        % mumat: 160기 동안 mu의 변화, popsize: 160개 동안 population size의 변화
                                                                                                                                                      
%% Transition

ALPHA   = params.ALPHA;
DELTA   = params.DELTA;
BETA    = params.BETA;
SIGMA   = params.SIGMA;
TAU     = params.TAU;
THETA   = params.THETA;
ZETA    = params.ZETA;
A       = params.A;
NA      = params.NA;
NZ      = params.NZ;
R       = params.R;
I       = params.I;
p       = params.p;
e       = params.e;
agrid   = params.agrid;

rgrid0 = (result0.r: (result1.r - result0.r)/(NP - 1): result1.r)';         % r을 기준으로 Transition을 진행 (Aiyagari는 K를 기준으로 진행하였음)
rgrid1 = zeros(size(rgrid0));

Lmat  = zeros(size(rgrid0));
KDmat = zeros(size(rgrid0));
KSmat = zeros(size(rgrid0));
BQmat = zeros(size(rgrid0));
BQTAXmat = zeros(size(rgrid0));
wmat  = zeros(size(rgrid0));
bmat  = zeros(I, size(rgrid0, 1));

savingvmat = zeros(NA, NZ, I+1, NP);
savinggmat = zeros(NA, NZ, I, NP);
savingg_orderedmat = zeros(NA, NZ, I, NP);
savingpopmat = zeros(NA, NZ, I, NP);

rgridmat = zeros(size(rgrid0, 1), 100);                                     % Tgrid 값들을 저장, 100은 임의의 큰 수 - iteration이 100을 넘지는 않을 것을 봄
rgridmat(:,1) = rgrid0;

maxmat = zeros(1, 100);                                                     % max(abs(Tgrid0-Tgrid1)) 값들을 저장, 100은 임의의 큰 수

it = 0;

while (max(abs(rgrid0 - rgrid1)) > 0.0001);

    it = it + 1;
    
    maxmat(it) = max(abs(rgrid0 - rgrid1));
    
    if (it > 1);
       rgrid0 = (rgrid0 + rgrid1)/2;
    end;
    
    fprintf('it = %d, max = %f\n', it, maxmat(it));
      
    for t = 1:NP;
        % 총노동공급
        Lmat(t) = 0;
        for i=1:I;
            Lmat(t) = Lmat(t) + (p(:,i)'*e(:,i)*mumat(i,t));                  % probability of productivity by age x productivity by age x population share by age
        end;
        
        % 총자본 및 임금
        KDmat(t) = ((rgrid0(t)+DELTA)/(ALPHA*A*Lmat(t)^(1-ALPHA)))^(1/(ALPHA-1));
        wmat(t) = (1-ALPHA)*A^(1/(1-ALPHA))*((rgrid0(t)+DELTA)/ALPHA)^(ALPHA/(ALPHA-1));
        
        % 퇴직자에게 주는 연금
        b = zeros(I,1);
        b(1:R-1) = zeros(R-1,1);
        mass_after = [zeros(1,R-1), ones(1, I-R+1)]*mumat(:, t);            % 은퇴자(46세 이상, i >= R) 인구 수
        b(R:I) = ones(I-R+1,1)*(THETA*wmat(t)*Lmat(t)*mass_after^(-1));     % 은퇴자 1인당 받는 연금액
        bmat(:, t) = b;
    end;
    
    % (I+1)-period value function, I-period policy function
    % I+1기의 경우 사망하므로 value function은 zero가 됨.
    
    KSmat(NP) = result1.KS;
    BQmat(NP) = result1.BQ;
    BQTAXmat(NP) = result1.BQTAX;
    
    savingvmat(:, :, :, NP) = result1.savingv;
    savinggmat(:, :, :, NP) = result1.savingg;
    savingg_orderedmat(:,:,:,NP) = result1.savingg_ordered;
        
    for t=1:NP-1;                                                           % 뒤에서 부터 역으로 value function과 policy function을 계산한다.
         
        % 유산 초기값 설정
        BQ0 = 0;
        diff_BQ = 100;
        
        % parameter 값 설정
        
        sim_params = params;
        
        sim_params.N = Nmat(NP-t);
        sim_params.s = smat(:,NP-t);
        sim_params.mu = mumat(:,NP-t);
        
        while abs(diff_BQ) > 0.01;                                          % 유산의 초기값과 실제 실현된 유산의 규모가 같아질 때까지 반복
        
            % Value Function Iteration
            
            [savingvmat(:,:,:,NP-t), savinggmat(:,:,:,NP-t), savingg_orderedmat(:,:,:,NP-t)] = kts_hugget1996_vf_transition(sim_params, rgrid0(NP-t), wmat(NP-t), bmat(:,NP-t), BQ0, savingvmat(:,:,:,NP-t+1));    % value function, policy function, ordered policy function
            
            % Monte Carlo Simulation for a cohort
        
            [savingpopmat(:,:,:,NP-t)] = kts_hugget1996_simulation(sim_params, savingg_orderedmat(:,:,:,NP-t));
        
            % Aggregate capital supply (KS) and bequest (BQ)
            KSmat(NP-t) = 0;
            BQ1 = 0;
            for i=1:I;
                KSmat(NP-t) = KSmat(NP-t) + smat(i, NP-t)*agrid'*sum(savingpopmat(:,:,i,NP-t)')'*mumat(i,NP-t);
                BQ1 = BQ1 + (1-smat(i, NP-t))*agrid'*sum(savingpopmat(:,:,i,NP-t)')'*mumat(i,NP-t);
            end;
            BQTAX = BQ1*ZETA;
            BQ1 = (1-ZETA)*BQ1;
        
            diff_BQ = BQ1 - BQ0;
        
            fprintf('it = %d, t = %d, r = %1.6f, KS = %3.4f, KD = %3.4f, BQ0 = %2.4f, BQ1 = %2.4f, diff_BQ = %2.4f\n', it, t, rgrid0(NP-t), KSmat(NP-t), KDmat(NP-t), BQ0, BQ1, diff_BQ);
        
            BQ0 = BQ1;
            
            BQmat(NP-t) = BQ0;
            BQTAXmat(NP-t) = BQTAX;
        
        end;

     end;    
          
     rgrid1(1) = result0.r;                                                 % rgrid(1) 초기값은 steady state 값을 사용
     
     savingpopmat(:,:,:,1) = result0.savingpop;                             % savingpop 초기값은 최초 steady state 값을 사용    
     
     for t=2:NP;                                                            % 순방향으로 rgrid1을 구함
         
         sim_params = params;
        
         sim_params.N = Nmat(t);
         sim_params.s = smat(:,t);
         sim_params.mu = mumat(:,t);
                  
         [savingpopmat(:,:,:,t)] = kts_hugget1996_simulation_transition(sim_params, savingg_orderedmat(:,:,:,t), savingpopmat(:,:,:,t-1));
         
         % Aggregate capital supply (KS) and bequest (BQ)
         KS = 0;
         BQ1 = 0;
         for i=1:I;
             KS = KS + smat(i,t)*agrid'*sum(savingpopmat(:,:,i,t)')'*mumat(i,t);
             BQ1 = BQ1 + (1-smat(i,t))*agrid'*sum(savingpopmat(:,:,i,t)')'*mumat(i,t);
         end;
         BQTAX = BQ1*ZETA;
         BQ1 = (1-ZETA)*BQ1;
         
         rgrid1(t) = ALPHA*A*KS^(ALPHA-1)*Lmat(t)^(1-ALPHA) - DELTA;
         
         fprintf('t=%d, rgrid1=%2.4f\n', t, rgrid1(t));
         
     end;
     
     rgridmat(:,it) = rgrid1;
     
     figure;
     plot(rgrid0, 'b--', 'LineWidth', 3); hold on;
     plot(rgrid1, 'r:', 'LineWidth', 3);
     legend('rgrid0', 'rgrid1');
     set(gca, 'Fontsize', 20);
     axis tight;
     drawnow;
     
end;








toc;