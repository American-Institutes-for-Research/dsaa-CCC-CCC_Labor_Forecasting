import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr
from scipy.signal import savgol_filter

# Data samples
t_ma = 3 # one-sided 3-day rolling average
T_raw = (datetime.strptime("04/18/2020", "%m/%d/%Y") - datetime.strptime("01/22/2020", "%m/%d/%Y")).days + 1 # number of periods in raw data
T_all = T_raw - t_ma + 1 # number of periods after rolling average
T = T_all - 2 # number of periods for estimation
             # -2: log difference for growth rates and initial observation for AR(1)
H = 60 # forecast horizon
dtrim = 3 # trim before/after peak for limit-info analysis

# Dimensions of estimation
n_param = 2
n_est = 2
n_homog = 1
n_heter = 4

# Construct sample
# Load data
raw = pd.read_csv("data_daily.txt", sep=' ')
v_country = raw.iloc[0,:]
v_date = raw.iloc[:,1]
v_datenum = v_date.copy()

confirmed = raw.iloc[:,2].values
confirmed = savgol_filter(confirmed, 3, 2, axis=0)

infected = raw.iloc[:,5].values
infected = savgol_filter(infected, 3, 2, axis=0)


# Sample selection
N = len(confirmed)
Y_level = np.empty((T_all,N))
ix_delete = np.zeros(N)
ix1 = np.empty((T,N)) # index for all periods used in limit-info likelihood
ix2 = np.empty((T,N)) # index for the second half used in limit-info likelihood
for i in range(N):
    # 1. Eliminate locations that have not reached 100 active infections
    starti = np.argwhere(confirmed[i] > 100)
    if not starti:
        ix_delete[i] = 1
        continue

    # 2. Eliminate locations for which t_max - dtrim < 0
    t_max0 = np.argmax(infected[starti+2:, i]) # t_max, event time
    t_max = t_max0 + starti - 1 # t_max, calendar time
    if not t_max or starti > t_max-dtrim:
        ix_delete[i] = 1
        continue

    # 3. Eliminate locations where the OLS estimate of the time-trend coefficient
    # is positive because the SIR model implies a decreasing growth rate
    aux = np.diff(np.log(np.maximum(infected[starti:, i], 1)))
    corr_aux = np.corrcoef(aux,np.arange(len(aux)))
    if corr_aux[0,1] > 0:
        ix_delete[i] = 1
        continue
    ix1[starti:t_max-dtrim,i] = 1
    ix2[starti:t_max-dtrim,i] = 0
    if t_max+dtrim <= T:
        ix1[t_max+dtrim:,i] = 1
        ix2[t_max+dtrim:,i] = 1
    Y_level[starti:,i] = infected[starti:,i]

Y_level = Y_level[:,ix_delete==0]
ix1 = ix1[:,ix_delete==0]
ix2 = ix2[:,ix_delete==0]
N = Y_level.shape[1]
Y = np.diff(np.log(np.maximum(Y_level,1)))

v_country = np.delete(v_country, np.where(ix_delete==1))
for i in range(N):
    v_country[i] = v_country[i].replace('*','')

# Construct xt for limit-info analysis
XX_homog = Y[:T,:]
YY = Y[1:T+1,:]
XX_heter = np.empty((T,N,n_heter/2))
XX_heter[:,:,0] = 1
for i in range(N):
    starti = np.argwhere(ix1[:,i]==1)[0][0]
    XX_heter[starti:,i,1] = np.arange(1,T-starti+1)
XX_heter[:,:,2:4] = XX_heter[:,:,:2]*ix2

# Estimation setup
# Hyperpior parameters
param0 = {}
param0['M_coef_homog'] = 0.5*np.ones((n_homog,1))
param0['dof0'] = (2*n_heter+1)*(n_heter-1)+1
param0['C0'] = np.zeros((n_heter,1))
param0['d0'] = 1
param0['a_bb0'] = 0.01
param0['b_bb0'] = 0.01
param0['a_aa0'] = 1
param0['b_aa0'] = 0.01
param0['c_aa0'] = 0.01

param0['V_coef_homog'] = np.eye(n_homog)

mean_x_heter = np.reshape(np.nanmean(np.nanmean(XX_heter*ix1)),[],1)
param0['W0'] = np.var(np.nanmean(YY*ix1))*(param0['dof0']-n_heter-1)*np.eye(n_heter)\
    /np.diag(mean_x_heter**2)/n_heter

# Initial values: pooled OLS
AX = np.concatenate((np.reshape(XX_homog, [-1, n_homog]), np.reshape(XX_heter, [-1, n_heter])), axis=1)
AY = YY.flatten()
AX = AX[ix1.flatten()==1,:]
AY = AY[ix1.flatten()==1]
Coef0 = np.linalg.solve(np.dot(AX.T, AX),np.dot(AX.T,AY))
beta0 = max(min(Coef0[0],.99),0)
ZZ = YY-beta0*XX_homog

init_Lamb = np.full((n_heter, N), np.nan)
ZZZ = np.full((T, N), np.nan)
ix2_sum = (np.sum(~np.isnan(ix2)) > 0)

trans_mat = (1 - beta0) * np.eye(4)
trans_mat[0][1] = beta0
trans_mat[2][3] = beta0

for i in range(N):
    if ix2_sum[i] == 1:
        n_reg = 4
    else:
        n_reg = 2
    XXi = reshape(XX_heter[ix1[:,i]==1,i,1:n_reg],[],n_reg)*trans_mat[:n_reg,:n_reg]
    if np.linalg.det(XXi.T @ XXi) > 1e-2:
        init_Lamb[:n_reg,i] = np.linalg.inv(XXi.T @ XXi) @ (XXi.T @ ZZ[ix1[:,i]==1,i])
    else:
        init_Lamb[:n_reg,i] = np.random.multivariate_normal(param0.C0[:n_reg], param0.d0 * param0.W0[:n_reg, :n_reg]).T
    if n_reg == 2:
        init_Lamb[2:4,i] = np.random.multivariate_normal(param0.C0[2:4], param0.d0 * param0.W0[2:4, 2:4]).T
    init_Lamb[1:3,i] = np.minimum(init_Lamb[1:3,i],-1e-2)
    init_Lamb[3,i] = np.maximum(np.minimum(init_Lamb[3,i],-1e-2/2-init_Lamb[1,i]),1e-2/4)
    ZZZ[ix1[:,i]==1,i] = ZZ[ix1[:,i]==1,i]-XXi*init_Lamb[:n_reg,i]

init["sigma2"] = max(np.nanvar(ZZZ), 1e-6)

# Numbers of draws
B_draw = 1000
N_draw = 9000

## Panel predictors
param_heter(param0, YY, XX_homog, XX_heter, ix1, [B_draw, N_draw], {temp_path, result_path, ""}, init, Y_level, v_country, v_datenum, H, is_full)
