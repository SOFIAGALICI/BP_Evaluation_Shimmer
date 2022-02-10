# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 10:19:37 2021

@author: engal
"""

import csv
import operator
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io
from scipy.signal import butter, find_peaks
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

#=======================================================================================
# DEFINITION OF ENVELOPE FUNCTION
def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal 
    is too big
    split: bool, optional, if True, split the signal in half along its mean, might help
    to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    s_filt=np.zeros(len(s))
    n=0
    for i in range(len(s)):
        if i==lmax[n]:
            s_filt[i]=s[i]-s[lmax[n]]
            if n<len(lmax)-1:
                n=n+1
        else:
            s_filt[i]=s[i]-s[lmax[n]]
            
    return s_filt

# PEAKS DETECTION FUNCTION
def peaks_detection(s_filt,ts,time,th):
    
    pks=find_peaks(s_filt,height=th)  
    ind_pks=pks[0]
    ts_pks=np.zeros(len(ts))
    vect_pks=np.zeros(len(ts))
    vect_pks[ind_pks]=s_filt[ind_pks]        
    ts_pks[ind_pks]=ts[ind_pks]      

    # Local maximus deletion
    int_t=round(0.5/(time[-1]/len(time)))    
    for i in range(len(vect_pks)):
        if i>len(vect_pks)-int_t: 
            break
        if vect_pks[i]>0: 
            for j in range(1,int_t): 
                if vect_pks[i+j]>0: 
                    vect_pks[i+j]=0
                    ts_pks[i+j]=0
                    
    return vect_pks,ts_pks

# FEATURE REDUCTION FUNCTION
def feat_reduction(feat,t_fitted):
    
    row=np.zeros(len(t_fitted))
    T=0
    for i in range(len(t_fitted)-1):
        if i>=T:
            for j in range(i+1,len(t_fitted)):
                # Time window of 10 seconds
                if t_fitted[j]-t_fitted[i]>=10:
                    ind1=np.arange(i,j)
                    
                    # Feature averaging
                    vect_feat=feat[ind1]
                    val_feat=np.mean(vect_feat)
                    row[ind1]=val_feat
                    T=j
                    break

    # If the last window is smaller than 10 s, the last values are averaged and fitted in a 10 s time window
    for i in range(len(row)):
        if row[i]==0:
            ind1=np.arange(i,len(row))
            
            # Feature averaging
            val_feat=np.mean(feat[ind1])
            row[ind1]=val_feat
            break
        
    return row

# REGRESSION PROCESS FUNCTION
def regression_process(model,matr_train,matr_test,i_train,i_test,sbp,dbp):
    
    # SBP
    modelfit_SBP = model.fit(matr_train,sbp[i_train]) # Model training
    sbp_pred=modelfit_SBP.predict(matr_test)             # Model testing
    mae_sbp=mean_absolute_error(sbp[i_test], sbp_pred)           # SBP MEA (mmHg)
    dev_sbp=np.std(sbp_pred)                                        # SBP dev std (mmHg)
    err_sbp=abs(sbp_pred-sbp[i_test])
    n=np.array(np.where(err_sbp>5))
    num_sbp=len(np.transpose(n))
    
    # DBP
    modelfit_DBP = model.fit(matr_train,dbp[i_train]) # Model training
    dbp_pred=modelfit_DBP.predict(matr_test)             # Model testing
    mae_dbp=mean_absolute_error(dbp[i_test], dbp_pred)           # DBP MEA (mmHg)
    dev_dbp=np.std(dbp_pred)                                        # SBP dev std (mmHg)
    err_dbp=abs(dbp_pred-dbp[i_test])
    n=np.array(np.where(err_dbp>5))
    num_dbp=len(np.transpose(n))
    
    return modelfit_SBP,modelfit_DBP,sbp_pred,dbp_pred,mae_sbp,mae_dbp,dev_sbp,dev_dbp

# CALIBRATION TIME WITHOUT TIME INTERVAL DATASET DIVISION FUNCTION
def calibration_time(t_min,t_table,mae_sbp,mae_dbp,dev_sbp,dev_dbp,sbp,dbp,model):
    
    N=0
    while(mae_sbp>2 or mae_dbp>2 or dev_sbp>8 or dev_dbp>8):
        t_min=t_min+60
        
        for i in range(len(t_table)):
            if t_table[i]-t_table[0]>=t_min:              # Time (s)
                ind_train=np.arange(0,i)
                ind_test=np.arange(i,len(t_table))
                break
            
        trainData_PTT=ptt[ind_train] 
        testData_PTT = ptt[ind_test]
        trainData_HR=hr[ind_train] 
        testData_HR = hr[ind_test]
        X_train=np.transpose(np.array([trainData_PTT,trainData_HR]))
        X_test=np.transpose(np.array([testData_PTT,testData_HR]))
        
        # Regression process
        modelfit_SBP,modelfit_DBP,SBP_pred,DBP_pred,mae_sbp,mae_dbp,dev_sbp,dev_dbp,=regression_process(model,X_train,X_test,ind_train,ind_test,sbp,dbp)
        
        N=N+1
        if N>11:
            t_min=0
            break
            
    return t_min

# CALIBRATION TIME WITHIN TIME INTERVAL DATASET DIVISION FUNCTION
def calibration_time_window(t_min,t_table,mae_sbp,mae_dbp,dev_sbp,dev_dbp,sbp,dbp,model,ptt,hr,index):
    
    N=0
    while(mae_sbp>2 or mae_dbp>2 or dev_sbp>8 or dev_dbp>8):
        t_min=t_min+60
        
        for i in range(len(t_table)):
            if t_table[i]-t_table[0]>=t_min:              # Time (s)
                ind_train=np.arange(0,i)
                ind_test=np.arange(i,len(t_table))
                break

        trainData_PTT=ptt[ind_train] 
        testData_PTT = ptt[ind_test]
        trainData_HR=hr[ind_train] 
        testData_HR = hr[ind_test]
        X_train=np.array([trainData_PTT,trainData_HR])
        X_test=np.array([testData_PTT,testData_HR])

        # Windows concatenation in a single training matrix
        PTT_temp=trainData_PTT
        HR_temp=trainData_HR
        PTT_regr=np.zeros((len(ind_train),np.dtype('int64').type(index)))
        HR_regr=np.zeros((len(ind_train),np.dtype('int64').type(index)))
        ind_temp=np.arange(index)
        
        for i in range(len(trainData_PTT)):
            PTT_regr[i,:]=PTT_temp[ind_temp]
            PTT_temp=np.roll(PTT_temp,1)
            HR_regr[i,:]=HR_temp[ind_temp]
            HR_temp=np.roll(HR_temp,1)

        X_train=np.concatenate((PTT_regr,HR_regr), axis=1)

        # Windows concatenation in a single test matrix
        PTT_temp=testData_PTT
        HR_temp=testData_HR
        PTT_regr=np.zeros((len(ind_test),np.dtype('int64').type(ind)))
        HR_regr=np.zeros((len(ind_test),np.dtype('int64').type(ind)))
        ind_temp=np.arange(ind)
        
        for i in range(len(testData_PTT)):
            PTT_regr[i,:]=PTT_temp[ind_temp]
            PTT_temp=np.roll(PTT_temp,1)
            HR_regr[i,:]=HR_temp[ind_temp]
            HR_temp=np.roll(HR_temp,1)

        X_test=np.concatenate((PTT_regr,HR_regr), axis=1)
        
        # Regression process
        modelfit_SBP,modelfit_DBP,SBP_pred,DBP_pred,mae_sbp,mae_dbp,dev_sbp,dev_dbp=regression_process(model,X_train,X_test,ind_train,ind_test,sbp,dbp)
        
        N=N+1
        if N>11:
            t_min=0
            break
        
    return t_min

# ALGORITHM START
#==============
# DATA LOADING
#==============
ecg_mat=scipy.io.loadmat('Prova_Sofia_Session1_Shimmer_6C0E_Calibrated_SD.mat')
ecg=ecg_mat['Shimmer_6C0E_ECG_LA_RA_24BIT_CAL']
tsecg=ecg_mat['Shimmer_6C0E_TimestampSync_Unix_CAL']
ppg_mat=scipy.io.loadmat('Prova_Sofia_Session1_Shimmer_9404_Calibrated_SD.mat')
ppg=ppg_mat['Shimmer_9404_PPG_A13_CAL']
tsppg=ppg_mat['Shimmer_9404_TimestampSync_Unix_CAL']

# Creation of numpy arrays
ecg_head=np.zeros(len(ecg))
ts_ecg=np.zeros(len(ecg))
ppg_head=np.zeros(len(ppg))
ts_ppg=np.zeros(len(ppg))
for i in range(len(ecg)):
    ecg_head[i]=ecg[i]
    ts_ecg[i]=tsecg[i]/1000
    
for i in range(len(ppg)):
    ppg_head[i]=ppg[i]
    ts_ppg[i]=tsppg[i]/1000
    
fs=504.12  # Sampling frequency (Hz)
rec_time_mins_ppg = ((len(ppg_head)-1)/fs)/60 
t_ppg = np.arange(0,len(ppg_head))/fs
rec_time_mins_ecg = ((len(ecg_head)-1)/fs)/60
t_ecg = np.arange(0,len(ecg_head))/fs

#=================
# SIGNAL PREPARING
#=================
# Cut of the first noisy samples (first 20 seconds)
cut_int=round(20/(t_ppg[-1]/len(t_ppg)))
ind=np.arange(0,cut_int)
ppg_head=np.delete(ppg_head,ind)
ts_ppg=np.delete(ts_ppg,ind)
t_ppg=np.delete(t_ppg,ind)
cut_int=round(20/(t_ecg[-1]/len(t_ecg)))
ind=np.arange(0,cut_int)
ecg_head=np.delete(ecg_head,ind)
ts_ecg=np.delete(ts_ecg,ind)
t_ecg=np.delete(t_ecg,ind)

# Signal alignment
if ts_ecg[0]<ts_ppg[0]:
    for i in range(len(ts_ecg)):
        if ts_ecg[i]>ts_ppg[0]:
            ind=np.arange(0,i-1)
            ecg_head=np.delete(ecg_head,ind)
            ts_ecg=np.delete(ts_ecg,ind)
            t_ecg=np.delete(t_ecg,ind)
            break
else:
    for i in range(len(ts_ppg)):
        if ts_ppg[i]>ts_ecg[0]:
            ind=np.arange(0,i-1)
            ppg_head=np.delete(ppg_head,ind)
            ts_ppg=np.delete(ts_ppg,ind)
            t_ppg=np.delete(t_ppg,ind)
            break

# Signals cut at the same length
if len(ts_ecg)>len(ts_ppg):
    t=t_ppg
    ind=np.arange(0,len(ts_ppg))
    ecg_head=ecg_head[ind]
    ts_ecg=ts_ecg[ind]         
else: 
    t=t_ecg
    ind=np.arange(0,len(ts_ecg))
    ppg_head=ppg_head[ind]
    ts_ppg=ts_ppg[ind]
    
# Signals synchronization
int_t=ts_ppg[0]-ts_ecg[0]
for i in range(len(ts_ecg)):
    if abs(ts_ppg[i]-ts_ecg[i])>int_t:
        ts_ppg[i]=ts_ecg[i]-int_t

#=================
# SIGNAL FILTERING
#=================
# ECG 
ecg_filt = hl_envelopes_idx(ecg_head)   # Baseline removal
      
# plt.plot(t,ecg_head)  
# plt.plot(t,ecg_filt)

# PPG 
fNy = fs/2    # Nyquist frequency (Hz)
ft = 50        # Cut off frequency (Hz) (experimental)
ws=0.1         # Passaband ripple (dB) (experimental)
wp=15          # Stopband attenuation (dB) (experimental)
fa=30          # Attenuation frequenzy (Hz) (experimental)
n,wn=scipy.signal.buttord(ft/fNy,fa/fNy,ws,wp)
b,a=scipy.signal.butter(n+1,wn)                 # 7-orer low-pass Butterworth filter
ppg_filt1=scipy.signal.filtfilt(b,a,ppg_head)
ppg_filt = hl_envelopes_idx(ppg_filt1)          # Baseline removal

# plt.plot(t,ppg_head)  
# plt.plot(t,ppg_filt)

#================
# PEAKS DETECTION
#================
# ECG
th_ecg=1.25     # Change the threshold eventually
vect_R, ts_R=peaks_detection(ecg_filt, ts_ecg, t, th_ecg)

plt.plot(t,ecg_filt)
plt.plot(t,vect_R,'o')

# PPG
th_ppg=25     # Change the threshold eventually
vect_P, ts_P=peaks_detection(ppg_filt, ts_ppg, t, th_ppg)

plt.plot(t,ppg_filt)
plt.plot(t,vect_P,'o')

#===================
# FEATURE EXTRACTION
#===================
n=0
T=0
found=0
ptt=np.zeros(len(ecg_filt))         # PTT array
hr=np.zeros(len(ecg_filt))          # HR array
timetable=np.zeros(len(ecg_filt))   # Timestamp array
for i in range(len(vect_R)):
    if i>=T:
        if vect_R[i]>0:
            found=0
            for j in range(i+1,len(vect_R)):
                if found==1:
                    break
                if vect_R[j]>0:
                    break
                else:
                    if vect_P[j]>0:
                        ptt[n]=ts_P[j]-ts_R[i]
                        for k in range(i+1,len(vect_R)):
                            if vect_R[k]>0:
                                hr[n]=60/(ts_R[k]-ts_R[i])
                                timetable[n]=ts_R[i]   
                                n=n+1
                                T=k
                                found=1
                                break    
                        
# Zero elements deletion and arrays cut at the same length
ind=np.array(np.where(ptt==0))
ptt=np.delete(ptt,ind)
ind=np.array(np.where(hr==0))
hr=np.delete(hr,ind)
ind=np.array(np.where(timetable==0))
timetable=np.delete(timetable,ind)

if len(ptt)>len(hr):
    ind=np.arange(0,len(hr))
    ptt=ptt[ind]
    timetable=timetable[ind]
else:
    ind=np.arange(0,len(ptt))
    hr=hr[ind]
    timetable=timetable[ind]
    
# Arrays cleaning
mean_PTT=np.mean(ptt)
dev_PTT=np.std(ptt)
mean_HR=np.mean(hr)
dev_HR=np.std(hr)

for i in range(len(timetable)):
    if ptt[i]>mean_PTT+dev_PTT or ptt[i]<mean_PTT-dev_PTT or hr[i]>mean_HR+dev_HR or hr[i]<mean_HR-dev_HR:
        ptt[i]=0
        hr[i]=0
        timetable[i]=0

ind=np.array(np.where(ptt==0))
ptt=np.delete(ptt,ind)
ind=np.array(np.where(hr==0))
hr=np.delete(hr,ind)
ind=np.array(np.where(timetable==0))
timetable=np.delete(timetable,ind)

#===================
# OMRON DATA LOADING
#===================
with open("Prova_Sofia.csv") as filecsv:
    reader=csv.reader(filecsv,delimiter=";")
    ts_omron=np.array(list(map(float,[(line[0]) for line in reader])))
with open("Prova_Sofia.csv") as filecsv:
    reader=csv.reader(filecsv,delimiter=";")
    sbp=np.array(list(map(float,[(line[1]) for line in reader])))
with open("Prova_Sofia.csv") as filecsv:
    reader=csv.reader(filecsv,delimiter=";")
    dbp=np.array(list(map(float,[(line[2]) for line in reader]))) 

# Omron's time values correspond to the time in which the device returns the 
# pressure values
ts_omron=ts_omron-60*np.ones(len(ts_omron)) 

# Creation of the interpolating time array
n=np.array(np.where(ts_ecg==timetable[0]))
m=np.array(np.where(ts_ecg==timetable[-1]))
ind=np.arange(n,m)
t_fit=ts_ecg[ind] 

# Interpolation of the Omron's data
SBP_fit=np.interp(t_fit,ts_omron,sbp)
DBP_fit=np.interp(t_fit,ts_omron,dbp)

# Interpolation of PTT and HR values
HR_fit=np.interp(t_fit,timetable,hr)
PTT_fit=np.interp(t_fit,timetable,ptt)

#==================
# FEATURE REDUCTION
#==================
row1=feat_reduction(PTT_fit,t_fit)     # PTT
row2=feat_reduction(HR_fit,t_fit)      # HR
row3=feat_reduction(SBP_fit,t_fit)     # SBP
row4=feat_reduction(DBP_fit,t_fit)     # DBP

# Arrays resampling
row1=np.interp(timetable,t_fit,row1)
row2=np.interp(timetable,t_fit,row2)
row3=np.transpose(np.interp(timetable,t_fit,row3))
row4=np.transpose(np.interp(timetable,t_fit,row4))

# PLOT 2 Y-AXES (HR AND PTT) 
# fig, ax1 = plt.subplots()

# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('HR [Hz]', color='red')
# ax1.plot(timetable, row2, color='red')
# ax1.tick_params(axis='y', labelcolor='red')

# ax2 = ax1.twinx()  
# ax2.set_ylabel('PTT [s]', color='blue')  
# ax2.plot(timetable, row1, color='blue')
# ax2.tick_params(axis='y', labelcolor='blue')

# fig.tight_layout()
# plt.show()

#===================
# REGRESSION METHODS
#===================
# Preparing data
# Training set contains the 70% of the whole dataset, the test set the remaining 30%
sz_train=round(0.7*len(row1))
ind_train=np.arange(0,sz_train)
ind_test=np.arange(sz_train,len(row1))
trainData_PTT=row1[ind_train] 
testData_PTT = row1[ind_test]
trainData_HR=row2[ind_train] 
testData_HR = row2[ind_test]
X_train=np.transpose(np.array([trainData_PTT,trainData_HR]))
X_test=np.transpose(np.array([testData_PTT,testData_HR]))
perc=round(0.2*len(ind_test))

# LINEAR REGRESSION
regr = linear_model.LinearRegression()             # Parameters definition
MLR_modelfit_SBP,MLR_modelfit_DBP,MLR_SBP_pred,MLR_DBP_pred,MLR_mae_SBP,MLR_mae_DBP,MLR_dev_SBP,MLR_dev_DBP=regression_process(regr,X_train,X_test,ind_train,ind_test,row3,row4)
MLR_coeff_SBP=MLR_modelfit_SBP.coef_
MLR_coeff_DBP=MLR_modelfit_DBP.coef_

# SBP plot
plt.plot(np.arange(0,len(ind_test)),MLR_SBP_pred,'r',label="Predicted SBP")
plt.plot(np.arange(0,len(ind_test)),row3[ind_test],'b',label="Real SBP")
plt.xlabel('Samples')
plt.ylabel('Amplitude (mmHg)')
plt.legend()

# DBP plot
plt.plot(np.arange(0,len(ind_test)),MLR_DBP_pred,'r',label="Predicted DBP")
plt.plot(np.arange(0,len(ind_test)),row4[ind_test],'b',label="Real DBP")
plt.xlabel('Samples')
plt.ylabel('Amplitude (mmHg)')
plt.legend()

# RANDOM FOREST REGRESSION
regr = RandomForestRegressor(n_estimators=100,random_state=7,criterion='mae') # Parameters definition
RFR_modelfit_SBP,RFR_modelfit_DBP,RFR_SBP_pred,RFR_DBP_pred,RFR_mae_SBP,RFR_mae_DBP,RFR_dev_SBP,RFR_dev_DBP=regression_process(regr,X_train,X_test,ind_train,ind_test,row3,row4)

# SBP plot
plt.plot(np.arange(0,len(ind_test)),RFR_SBP_pred,'r',label="Predicted SBP")
plt.plot(np.arange(0,len(ind_test)),row3[ind_test],'b',label="Real SBP")
plt.xlabel('Samples')
plt.ylabel('Amplitude (mmHg)')
plt.legend()

# DBP plot
plt.plot(np.arange(0,len(ind_test)),RFR_DBP_pred,'r',label="Predicted DBP")
plt.plot(np.arange(0,len(ind_test)),row4[ind_test],'b',label="Real DBP")
plt.xlabel('Samples')
plt.ylabel('Amplitude (mmHg)')
plt.legend()

# RIDGE REGRESSION
regr = Ridge(alpha=.01)    
RR_modelfit_SBP,RR_modelfit_DBP,RR_SBP_pred,RR_DBP_pred,RR_mae_SBP,RR_mae_DBP,RR_dev_SBP,RR_dev_DBP=regression_process(regr,X_train,X_test,ind_train,ind_test,row3,row4)

# SBP plot
plt.plot(np.arange(0,len(ind_test)),RR_SBP_pred,'r',label="Predicted SBP")
plt.plot(np.arange(0,len(ind_test)),row3[ind_test],'b',label="Real SBP")
plt.xlabel('Samples')
plt.ylabel('Amplitude (mmHg)')
plt.legend()

# DBP plot
plt.plot(np.arange(0,len(ind_test)),RR_DBP_pred,'r',label="Predicted DBP")
plt.plot(np.arange(0,len(ind_test)),row4[ind_test],'b',label="Real DBP")
plt.xlabel('Samples')
plt.ylabel('Amplitude (mmHg)')
plt.legend()

# SUPPORT VECTOR REGRESSION
regr = SVR(C=50, epsilon=0.2)      
SVR_modelfit_SBP,SVR_modelfit_DBP,SVR_SBP_pred,SVR_DBP_pred,SVR_mae_SBP,SVR_mae_DBP,SVR_dev_SBP,SVR_dev_DBP=regression_process(regr,X_train,X_test,ind_train,ind_test,row3,row4)

# SBP plot
plt.plot(np.arange(0,len(ind_test)),SVR_SBP_pred,'r',label="Predicted SBP")
plt.plot(np.arange(0,len(ind_test)),row3[ind_test],'b',label="Real SBP")
plt.xlabel('Samples')
plt.ylabel('Amplitude (mmHg)')
plt.legend()

# DBP plot
plt.plot(np.arange(0,len(ind_test)),SVR_DBP_pred,'r',label="Predicted DBP")
plt.plot(np.arange(0,len(ind_test)),row4[ind_test],'b',label="Real DBP")
plt.xlabel('Samples')
plt.ylabel('Amplitude (mmHg)')
plt.legend()

# 4 SUBPLOTS (%matplotlib qt)
# fig, ax=plt.subplots(4)

# # HR E PTT
# # ax[0].set_xlabel('Samples')
# ax[0].set_ylabel('HR [Hz]', color='tab:red')
# ax[0].plot(row2, color='tab:red')
# ax[0].tick_params(axis='y', labelcolor='tab:red')
# ax[0].set_facecolor('#E6EAF2')
# ax[0].set_title('Prediction vs. Test for measurement 3')
# ax[0].spines[:].set_visible(False)
# ax02=ax[0].twinx()  
# ax02.set_ylabel('PTT [s]', color='tab:blue')  
# ax02.plot(row1, color='tab:blue')
# ax02.tick_params(axis='y', labelcolor='tab:blue')
# ax02.spines[:].set_visible(False)

# #SBP
# ax[1].plot(ind_train,row3[ind_train], color='tab:blue',label='Train')
# ax[1].plot(ind_test,row3[ind_test],'b--',color='tab:blue',label='Test')
# ax[1].plot(ind_test,RFR_SBP_pred,color='tab:red',label='RF: n_trees=100')
# ax[1].plot(ind_test,RR_SBP_pred,color='tab:green',label='Ridge: alpha=0.01')
# ax[1].plot(ind_test,MLR_SBP_pred,color='m',label='Linear')
# ax[1].plot(ind_test,SVR_SBP_pred,color='#ff7514',label='SVR: C=50')
# ax[1].set_facecolor('#E6EAF2')
# ax[1].spines[:].set_visible(False)
# # ax[1].set_xlabel('Samples')
# ax[1].set_ylabel('SBP [mmHg]')
# ax[1].grid(linewidth=1,color='#FFFFFF',linestyle='-')
# ax[1].legend(loc='upper left',ncol=3,frameon=False)

# #DBP
# ax[2].plot(ind_train,row4[ind_train], color='tab:blue',label='Train')
# ax[2].plot(ind_test,row4[ind_test],'b--',color='tab:blue',label='Test')
# ax[2].plot(ind_test,RFR_DBP_pred,color='tab:red',label='RF: n_trees=100')
# ax[2].plot(ind_test,RR_DBP_pred,color='tab:green',label='Ridge: alpha=0.01')
# ax[2].plot(ind_test,MLR_DBP_pred,color='m',label='Linear')
# ax[2].plot(ind_test,SVR_DBP_pred,color='#ff7514',label='SVR: C=50')
# ax[2].set_facecolor('#E6EAF2')
# ax[2].set_xlabel('Samples')
# ax[2].set_ylabel('DBP [mmHg]')
# ax[2].grid(linewidth=1,color='#FFFFFF',linestyle='-')
# ax[2].legend(loc='upper left',ncol=3,frameon=False)
# ax[2].spines[:].set_visible(False)

# # ISTOGRAMMA
# a=np.array([SVR_mae_SBP, RR_mae_SBP, RFR_mae_SBP, MLR_mae_SBP])
# b=np.array([SVR_mae_DBP, RR_mae_DBP, RFR_mae_DBP, MLR_mae_DBP]) 
# xticks = [0,1,2,3]
# width=0.35
# axis = np.arange(len(xticks))
# labels=['SVR: C=50', 'Ridge: alpha=0.01', 'RF: n_trees=100', 'Linear']
# ax[3].bar(axis+width/2,b, width, label='DBP')
# ax[3].bar(axis-width/2,a, width, label='SBP')
# ax[3].set_facecolor('#E6EAF2')
# ax[3].grid(linewidth=1,color='#FFFFFF',linestyle='-')
# ax[3].set_title("MAE for each algorithm")
# ax[3].set_xticks(xticks)
# ax[3].set_xticklabels(labels) 
# ax[3].set_ylim(0,15)
# ax[3].set_ylabel('MAE [mmHg]')
# ax[3].legend(frameon=False)
# ax[3].spines[:].set_visible(False)

#=======================================
# REGRESSION METHODS WITHIN A 10s WINDOW
#=======================================
# Preparing data
# Definition of a 10 seconds window 
for i in range (1,len(row1)):
    if abs(timetable[i]-timetable[0])>=15:
        ind=i  # Samples
        break
    
# Windows concatenation in a single training matrix
PTT_temp=trainData_PTT
HR_temp=trainData_HR
PTT_regr=np.zeros((len(ind_train),np.dtype('int64').type(ind)))
HR_regr=np.zeros((len(ind_train),np.dtype('int64').type(ind)))
ind_temp=np.arange(ind)
for i in range(len(trainData_PTT)):
    PTT_regr[i,:]=PTT_temp[ind_temp]
    PTT_temp=np.roll(PTT_temp,1)
    HR_regr[i,:]=HR_temp[ind_temp]
    HR_temp=np.roll(HR_temp,1)

X_train=np.concatenate((PTT_regr,HR_regr), axis=1)

# Windows concatenation in a single test matrix
PTT_temp=testData_PTT
HR_temp=testData_HR
PTT_regr=np.zeros((len(ind_test),np.dtype('int64').type(ind)))
HR_regr=np.zeros((len(ind_test),np.dtype('int64').type(ind)))
ind_temp=np.arange(ind)
for i in range(len(testData_PTT)):
    PTT_regr[i,:]=PTT_temp[ind_temp]
    PTT_temp=np.roll(PTT_temp,1)
    HR_regr[i,:]=HR_temp[ind_temp]
    HR_temp=np.roll(HR_temp,1)

X_test=np.concatenate((PTT_regr,HR_regr), axis=1)

# LINEAR REGRESSION
regr = linear_model.LinearRegression()             # Parameters definition
wind_MLR_modelfit_SBP,wind_MLR_modelfit_DBP,wind_MLR_SBP_pred,wind_MLR_DBP_pred,wind_MLR_mae_SBP,wind_MLR_mae_DBP,wind_MLR_dev_SBP,wind_MLR_dev_DBP=regression_process(regr,X_train,X_test,ind_train,ind_test,row3,row4)

# SBP plot
plt.plot(np.arange(0,len(ind_test)),wind_MLR_SBP_pred,'r',label="Predicted SBP")
plt.plot(np.arange(0,len(ind_test)),row3[ind_test],'b',label="Real SBP")
plt.xlabel('Samples')
plt.ylabel('Amplitude (mmHg)')
plt.legend()

# DBP plot
plt.plot(np.arange(0,len(ind_test)),wind_MLR_DBP_pred,'r',label="Predicted DBP")
plt.plot(np.arange(0,len(ind_test)),row4[ind_test],'b',label="Real DBP")
plt.xlabel('Samples')
plt.ylabel('Amplitude (mmHg)')
plt.legend()

# RANDOM FOREST REGRESSION
regr = RandomForestRegressor(n_estimators=100,random_state=7,criterion='mae') # Parameters definition
wind_RFR_modelfit_SBP,wind_RFR_modelfit_DBP,wind_RFR_SBP_pred,wind_RFR_DBP_pred,wind_RFR_mae_SBP,wind_RFR_mae_DBP,wind_RFR_dev_SBP,wind_RFR_dev_DBP=regression_process(regr,X_train,X_test,ind_train,ind_test,row3,row4)

# SBP plot
plt.plot(np.arange(0,len(ind_test)),wind_RFR_SBP_pred,'r',label="Predicted SBP")
plt.plot(np.arange(0,len(ind_test)),row3[ind_test],'b',label="Real SBP")
plt.xlabel('Samples')
plt.ylabel('Amplitude (mmHg)')
plt.legend()

# DBP plot
plt.plot(np.arange(0,len(ind_test)),wind_RFR_DBP_pred,'r',label="Predicted DBP")
plt.plot(np.arange(0,len(ind_test)),row4[ind_test],'b',label="Real DBP")
plt.xlabel('Samples')
plt.ylabel('Amplitude (mmHg)')
plt.legend()

# RIDGE REGRESSION
regr = Ridge(alpha=.01)    
wind_RR_modelfit_SBP,wind_RR_modelfit_DBP,wind_RR_SBP_pred,wind_RR_DBP_pred,wind_RR_mae_SBP,wind_RR_mae_DBP,wind_RR_dev_SBP,wind_RR_dev_DBP=regression_process(regr,X_train,X_test,ind_train,ind_test,row3,row4)

# SBP plot
plt.plot(np.arange(0,len(ind_test)),wind_RR_SBP_pred,'r',label="Predicted SBP")
plt.plot(np.arange(0,len(ind_test)),row3[ind_test],'b',label="Real SBP")
plt.xlabel('Samples')
plt.ylabel('Amplitude (mmHg)')
plt.legend()

# DBP plot
plt.plot(np.arange(0,len(ind_test)),wind_RR_DBP_pred,'r',label="Predicted DBP")
plt.plot(np.arange(0,len(ind_test)),row4[ind_test],'b',label="Real DBP")
plt.xlabel('Samples')
plt.ylabel('Amplitude (mmHg)')
plt.legend()

# SUPPORT VECTOR REGRESSION
regr = SVR(C=50, epsilon=0.2)      
wind_SVR_modelfit_SBP,wind_SVR_modelfit_DBP,wind_SVR_SBP_pred,wind_SVR_DBP_pred,wind_SVR_mae_SBP,wind_SVR_mae_DBP,wind_SVR_dev_SBP,wind_SVR_dev_DBP=regression_process(regr,X_train,X_test,ind_train,ind_test,row3,row4)

# SBP plot
plt.plot(np.arange(0,len(ind_test)),wind_SVR_SBP_pred,'r',label="Predicted SBP")
plt.plot(np.arange(0,len(ind_test)),row3[ind_test],'b',label="Real SBP")
plt.xlabel('Samples')
plt.ylabel('Amplitude (mmHg)')
plt.legend()

# DBP plot
plt.plot(np.arange(0,len(ind_test)),wind_SVR_DBP_pred,'r',label="Predicted DBP")
plt.plot(np.arange(0,len(ind_test)),row4[ind_test],'b',label="Real DBP")
plt.xlabel('Samples')
plt.ylabel('Amplitude (mmHg)')
plt.legend()

#=========================
# MINIMUM CALIBRATION TIME
#=========================
# Without time interval dataset division
min_calib_t=300

for i in range(len(timetable)):
    if timetable[i]-timetable[0]>=min_calib_t:  # Time (s)
        ind_train=np.arange(0,i)
        ind_test=np.arange(i,len(timetable))
        break
    
trainData_PTT=ptt[ind_train] 
testData_PTT = ptt[ind_test]
trainData_HR=hr[ind_train] 
testData_HR = hr[ind_test]
X_train=np.transpose(np.array([trainData_PTT,trainData_HR]))
X_test=np.transpose(np.array([testData_PTT,testData_HR]))

regr = linear_model.LinearRegression()          # Change eventually
MLR_modelfit_SBP,MLR_modelfit_DBP,MLR_SBP_pred,MLR_DBP_pred,MLR_mae_SBP,MLR_mae_DBP,MLR_dev_SBP,MLR_dev_DBP=regression_process(regr,X_train,X_test,ind_train,ind_test,row3,row4)
calib_time=calibration_time(min_calib_t,timetable,MLR_mae_SBP,MLR_mae_DBP,MLR_dev_SBP,MLR_dev_DBP,row3,row4,regr)

# Within time interval dataset division
min_calib_wind_t=300

for i in range(len(timetable)):
    if timetable[i]-timetable[0]>=min_calib_wind_t:    # Time (s)
        ind_train=np.arange(0,i)
        ind_test=np.arange(i,len(timetable))
        break

trainData_PTT=row1[ind_train] 
testData_PTT = row1[ind_test]
trainData_HR=row2[ind_train] 
testData_HR = row2[ind_test]
X_train=np.array([trainData_PTT,trainData_HR])
X_test=np.array([testData_PTT,testData_HR])
perc=round(0.2*len(ind_test))

# Windows concatenation in a single training matrix
PTT_temp=trainData_PTT
HR_temp=trainData_HR
PTT_regr=np.zeros((len(ind_train),np.dtype('int64').type(ind)))
HR_regr=np.zeros((len(ind_train),np.dtype('int64').type(ind)))
ind_temp=np.arange(ind)
for i in range(len(trainData_PTT)):
    PTT_regr[i,:]=PTT_temp[ind_temp]
    PTT_temp=np.roll(PTT_temp,1)
    HR_regr[i,:]=HR_temp[ind_temp]
    HR_temp=np.roll(HR_temp,1)

X_train=np.concatenate((PTT_regr,HR_regr), axis=1)

# Windows concatenation in a single test matrix
PTT_temp=testData_PTT
HR_temp=testData_HR
PTT_regr=np.zeros((len(ind_test),np.dtype('int64').type(ind)))
HR_regr=np.zeros((len(ind_test),np.dtype('int64').type(ind)))
ind_temp=np.arange(ind)
for i in range(len(testData_PTT)):
    PTT_regr[i,:]=PTT_temp[ind_temp]
    PTT_temp=np.roll(PTT_temp,1)
    HR_regr[i,:]=HR_temp[ind_temp]
    HR_temp=np.roll(HR_temp,1)

X_test=np.concatenate((PTT_regr,HR_regr), axis=1)

regr = linear_model.LinearRegression()              # Change eventually
#regr = SVR(C=50, epsilon=0.2)    
wind_MLR_modelfit_SBP,wind_MLR_modelfit_DBP,wind_MLR_SBP_pred,wind_MLR_DBP_pred,wind_MLR_mae_SBP,wind_MLR_mae_DBP,wind_MLR_dev_SBP,wind_MLR_dev_DBP=regression_process(regr,X_train,X_test,ind_train,ind_test,row3,row4)
calib_time_wind=calibration_time_window(min_calib_wind_t,timetable,wind_MLR_mae_SBP,wind_MLR_mae_DBP,wind_MLR_dev_SBP,wind_MLR_dev_DBP,row3,row4,regr,row1,row2,ind)
