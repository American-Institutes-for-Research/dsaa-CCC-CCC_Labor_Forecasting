function param_heter(param,YY,XX_homog,XX_heter,ix1,num_draw,tags,init,Y_level,...
    v_country,v_datenum,H,is_full)

%% Inputs
% Hyperpior parameters
M_coef_homog = param.M_coef_homog;
V_coef_homog = param.V_coef_homog;
dof0 = param.dof0;
W0 = param.W0;
C0 = param.C0;
d0 = param.d0;
a_bb0 = param.a_bb0;
b_bb0 = param.b_bb0;
a_aa0 = param.a_aa0;
b_aa0 = param.b_aa0;
c_aa0 = param.c_aa0;

% Data
[T,N] = size(YY);
n_homog = size(XX_homog,3);
n_heter = size(XX_heter,3);

% MCMC
B_draw = num_draw(1);
N_draw = num_draw(2);

%% Other specification
% for adaptive RWMH
c = 0.55;
ar_tg = 0.3;
zeta = 1;

%% Prepare
% Store the draws
Coef_homog_draw = nan(B_draw+N_draw,n_homog);
Lamb_draw = nan(B_draw+N_draw,N,n_heter);
sigma2_draw = nan(B_draw+N_draw,N);
if is_full == 1
    Mu_draw = nan(B_draw+N_draw,n_heter);
    Omega2_draw = nan(B_draw+N_draw,n_heter,n_heter);
end

% Initial values 
Lamb0 = init.Lamb;
sigma20 = init.sigma2;
Coef_homog0 = M_coef_homog;

dof = dof0+N;
d = 1/(1/d0+N);

aa = 2;
bb = a_bb0/b_bb0;

%% Gibbs sampler
tic
for i_draw = 1:B_draw+N_draw
    % Display iteration
    if mod(i_draw,1000) == 0
        disp(['param (heter) draw ' num2str(i_draw)])
    end
    
    % Draw mu, omega2, aa, bb
    C = d*(C0/d0+sum(Lamb0,2));
    W = W0+C0*C0'/d0+Lamb0*Lamb0'-C*C'/d;
    Omega2 = iwishrnd(W,dof);
    Mu = mvnrnd(C',d*Omega2)';
    
    loga_aa1 = log(a_aa0)+sum(log(sigma20));
    b_aa1 = b_aa0+N;
    c_aa1 = c_aa0+N;
    aac = randn(1)*zeta+aa;
    if aac>0
        ar_ratio = exp((-loga_aa1+log(bb)*c_aa1)*(aac-aa)-b_aa1*(gammaln(aac)-gammaln(aa)));
    else
        ar_ratio = 0;
    end
    ar = min(1,ar_ratio);
    u = rand(1);
    aa = aac*(u<=ar)+aa*(u>ar);
    zeta = min(max(exp(log(zeta)+i_draw^(-c)*(ar-ar_tg)),1e-10),1e10);
    
    a_bb1 = a_bb0+N*aa;
    b_bb1 = b_bb0+sum(1./sigma20);
    bb = gamrnd(a_bb1,1/b_bb1);
    
    % Draw lambda_i, sigma2_i
    ZZ = YY;
    for i_homog = 1:n_homog
        ZZ = ZZ-Coef_homog0(i_homog)*XX_homog(:,:,i_homog);
    end
    XZ = XX_heter*(1-Coef_homog0); 
    XZ(:,:,[2 4]) = XZ(:,:,[2 4])+XX_heter(:,:,[1 3])*Coef_homog0;
    for i = 1:N
        X_aux = reshape(XZ(ix1(:,i)==1,i,:),[],n_heter);
        V_lamb = eye(n_heter)/(eye(n_heter)/Omega2+(X_aux'*X_aux)/sigma20(i));
        M_lamb = V_lamb*(Omega2\Mu+(X_aux'*ZZ(ix1(:,i)==1,i))/sigma20(i));
        
        for j = 1:n_heter
            aux_M = M_lamb;
            aux_M(j) = [];
            aux_lamb = Lamb0(:,i);
            aux_lamb(j) =[];
            
            aux_V = V_lamb;
            aux_V(j,:) = [];
            aux_cov = aux_V(:,j);
            aux_V(:,j) = [];
            
            aux_m = M_lamb(j)+(aux_cov'/aux_V)*(aux_lamb-aux_M);
            aux_s = sqrt(V_lamb(j,j)-(aux_cov'/aux_V)*aux_cov);
            
            if j == 1 
                Lamb0(j,i) = aux_m+aux_s*randn(1);
            elseif j==2 
                Lamb0(j,i) = aux_m+aux_s*trandn(-inf,(0-Lamb0(4,i)-aux_m)/aux_s);
            elseif j==3
                Lamb0(j,i) = aux_m+aux_s*trandn(-inf,(0-aux_m)/aux_s);
            else
                Lamb0(j,i) = aux_m+aux_s*trandn((-0-aux_m)/aux_s,(0-Lamb0(2,i)-aux_m)/aux_s);
            end
        end
    end
    
	ZZZ = ZZ;
    for i_heter = 1:n_heter
        ZZZ = ZZZ-Lamb0(i_heter,:).*XZ(:,:,i_heter);
    end
	
    sigma20 = 1./gamrnd(aa+nansum(ix1)/2,1./(bb+nansum((ZZZ.*ix1).^2)/2));
    
    % Draw common coefficients
    AY = YY;
    AX = XX_homog;
    for i_heter = 1:n_heter
        AY = AY-Lamb0(i_heter,:).*XX_heter(:,:,i_heter);
        AX = AX-Lamb0(i_heter,:).*(XX_heter(:,:,i_heter)-(i_heter==2)...
            -(i_heter==4)*(XX_heter(:,:,i_heter)>0)); 
    end
    s_fac = reshape(repmat(sqrt(sigma20),T,1),[],1);
    AYs = AY(:)./s_fac;
    AXs = AX(:)./repmat(s_fac,1,n_homog);
    V_coef_homog1 = eye(n_homog)/(eye(n_homog)/V_coef_homog+(AXs(ix1(:)==1,:)'*AXs(ix1(:)==1,:)));
    M_coef_homog1 = V_coef_homog1*(V_coef_homog\M_coef_homog+AXs(ix1(:)==1,:)'*AYs(ix1(:)==1,:));
    
    Coef_homog0 = M_coef_homog1+sqrt(V_coef_homog1)*trandn((0-M_coef_homog1)/sqrt(V_coef_homog1),...
        (.99-M_coef_homog1)/sqrt(V_coef_homog1)); 
    
    Coef_homog_draw(i_draw,:) = Coef_homog0;
    Lamb_draw(i_draw,:,:) = Lamb0';
    sigma2_draw(i_draw,:) = sigma20;
    
    if is_full == 1
        Mu_draw(i_draw,:) = Mu;
        Omega2_draw(i_draw,:,:) = Omega2;
    end
    
end
toc

%% Output
% Save MCMC draws of the f(lambda_i)
if is_full == 1
    save([tags{1} 'mcmc' tags{3} '_dist.mat'],'Mu_draw','Omega2_draw','-v7.3')
    clear Mu_draw Omega2_draw;
end

% Forecasts
in_out_smpl_draws;

% Figures
fcst_plot;

% Save
save([tags{1} 'mcmc' tags{3} '.mat'],'-v7.3')