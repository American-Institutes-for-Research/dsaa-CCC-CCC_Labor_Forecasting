%% Panel Forecasts of Country-Level Covid-19 Infections
%  Laura Liu, Hyungsik Roger Moon, and Frank Schorfheide

%  The code in this replication package illustrates how to produce country-level 
%  density forecasts and generate figures in Appendix A (forecast origin: 2020-04-18).

clear all
close all
clc

addpath('tools')
result_path = 'data_results/';
temp_path = 'data_results/';
is_full = 0;

%% Data samples
t_ma = 3; % one-sided 3-day rolling average
T_raw = datenum('4/18/2020')-datenum('1/22/2020')+1; % number of periods in raw data
T_all = T_raw-t_ma+1; % number of periods after rolling average
T = T_all-2; % number of periods for estimation 
             % -2: log difference for growth rates and initial observation for AR(1)
H = 60; % forecast horizon
dtrim = 3; % trim before/after peak for limit-info analysis

% Dimensions of estimation
n_param = 2;
n_est = 2;
n_homog = 1;
n_heter = 4;

%% Construct sample
% Load data
raw = importdata([temp_path 'data_daily.txt']);
v_country = raw.textdata(1:T_raw:end,1);
v_date = raw.textdata(t_ma+2:T_raw,2);
v_datenum = nan(T,1);
for t = 1:T
    v_datenum(t) = datenum(v_date{t});
end
confirmed = reshape(raw.data(:,1),T_raw,[]);
confirmed = movmean(confirmed,[2 0],'Endpoints','discard');
infected = reshape(raw.data(:,4),T_raw,[]);
infected = movmean(infected,[2 0],'Endpoints','discard');

% Sample selection
N = size(confirmed,2);
Y_level = nan(T_all,N);
ix_delete = zeros(1,N);
ix1 = nan(T,N); % index for all periods used in limit-info likelihood
ix2 = nan(T,N); % index for the second half used in limit-info likelihood
for i = 1:N
    % 1. Eliminate locations that have not reached 100 active infections
    starti = find(confirmed(:,i)>100,1,'first');
    if isempty(starti) 
        ix_delete(i) = 1;
        continue        
    end
    
    % 2. Eliminate locations for which t_max - dtrim < 0
    [~,t_max0] = max(infected(starti+2:end,i)); % t_max, event time
    t_max = t_max0+starti-1; % t_max, calendar time
    if isempty(t_max) || starti>t_max-dtrim
        ix_delete(i) = 1;
        continue
    end
    
    % 3. Eliminate locations where the OLS estimate of the time-trend coefficient
    % is positive because the SIR model implies a decreasing growth rate
    aux = diff(log(max(infected(starti:end,i),1)));
    corr_aux = corrcoef(aux,(1:length(aux))');
    if corr_aux(1,2) > 0 
        ix_delete(i) = 1;
        continue
    end
    ix1(starti:t_max-dtrim,i) = 1; 
    ix2(starti:t_max-dtrim,i) = 0;
    if t_max+dtrim <= T
        ix1(t_max+dtrim:end,i) = 1;
        ix2(t_max+dtrim:end,i) = 1;
    end
    Y_level(starti:end,i) = infected(starti:end,i);
end
Y_level(:,ix_delete==1) = [];
ix1(:,ix_delete==1) = [];
ix2(:,ix_delete==1) = [];
N = size(Y_level,2);
Y = diff(log(max(Y_level,1)));

v_country(ix_delete==1) = [];
for i = 1:N
    v_country{i} = erase(v_country{i},'*');
end

% Construct xt for limit-info analysis
XX_homog = Y(1:T,:);
YY = Y(2:T+1,:);
XX_heter = nan(T,N,n_heter/2);
XX_heter(:,:,1) = 1;
for i = 1:N
    starti = find(ix1(:,i)==1,1,'first');
    XX_heter(starti:end,i,2) = 1:T-starti+1;
end
XX_heter(:,:,3:4) = XX_heter(:,:,1:2).*ix2;

%% Estimation setup
% Hyperpior parameters
param0 = [];
param0.M_coef_homog = 0.5*ones(n_homog,1);
param0.dof0 = (2*n_heter+1)*(n_heter-1)+1;
param0.C0 = zeros(n_heter,1);
param0.d0 = 1;
param0.a_bb0 = 0.01;
param0.b_bb0 = 0.01;
param0.a_aa0 = 1;
param0.b_aa0 = 0.01;
param0.c_aa0 = 0.01;

param0.V_coef_homog = eye(n_homog);

mean_x_heter = reshape(nanmean(nanmean(XX_heter.*ix1)),[],1);
param0.W0 = var(nanmean(YY.*ix1))*(param0.dof0-n_heter-1)*eye(n_heter)...
    /diag(mean_x_heter.^2)/n_heter;

% Initial values: pooled OLS
AX = [reshape(XX_homog,[],n_homog) reshape(XX_heter,[],n_heter)];
AY = YY(:);
AX = AX(ix1(:)==1,:);
AY = AY(ix1(:)==1);
Coef0 = (AX'*AX)\(AX'*AY);
beta0 = max(min(Coef0(1),.99),0);
ZZ = YY-beta0*XX_homog;

init.Lamb = nan(n_heter,N);
ZZZ = nan(T,N);
ix2_sum = sum(~isnan(ix2))>0;

trans_mat = (1-beta0)*eye(4);
trans_mat(1,2) = beta0;
trans_mat(3,4) = beta0;
for i = 1:N
    if ix2_sum(i) == 1
        n_reg = 4;
    else 
        n_reg = 2;
    end
    XXi = reshape(XX_heter(ix1(:,i)==1,i,1:n_reg),[],n_reg)*trans_mat(1:n_reg,1:n_reg);
    if det(XXi'*XXi) > 1e-2
        init.Lamb(1:n_reg,i) = (XXi'*XXi)\(XXi'*ZZ(ix1(:,i)==1,i));
    else
        init.Lamb(1:n_reg,i) = mvnrnd(param0.C0(1:n_reg)',param0.d0*param0.W0(1:n_reg,1:n_reg))';
    end
    if n_reg == 2
        init.Lamb(3:4,i) = mvnrnd(param0.C0(3:4)',param0.d0*param0.W0(3:4,3:4))';
    end
    init.Lamb(2:3,i) = min(init.Lamb(2:3,i),-1e-2);
    init.Lamb(4,i) = max(min(init.Lamb(4,i),-1e-2/2-init.Lamb(2,i)),1e-2/4);
    ZZZ(ix1(:,i)==1,i) = ZZ(ix1(:,i)==1,i)-XXi*init.Lamb(1:n_reg,i);
end
init.sigma2 = max(nanvar(ZZZ),1e-6);

% Numbers of draws
B_draw = 1000;
N_draw = 9000;

%% Panel predictors
param_heter(param0,YY,XX_homog,XX_heter,ix1,[B_draw,N_draw],{temp_path,result_path,...
    ''},init,Y_level,v_country,v_datenum,H,is_full);