# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 16:56:25 2021

@author: Yuting
"""
import numpy as np
import os
import csv
import pandas as pd
import lightkurve as lk
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.ndimage import gaussian_filter1d
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scipy.optimize as optimize
from sklearn.metrics import r2_score
import matplotlib.gridspec as gridspec
def read_tar(name):
    p = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize'
    p = os.path.join(p,'tar_%s.csv'%name)
    df = pd.read_csv(p)
    tar = np.array(df['target'])
    return tar
#%%
############# Section:light curve
def lightcurve_full(name,target):
    #read path#
    #if name == 'KOI':
    #    rpath = r'C:/Users/Yuting/Desktop/UTokyo/research/discussion/paper/organize/lightcurve/KOI/data'
    #elif name == 'nonKOI':
    #    rpath = r'C:/Users/Yuting/Desktop/UTokyo/research/discussion/paper/organize/lightcurve/nonKOI/data'
    rpath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/lightcurve/%s/data'%name
    rpath = os.path.join(rpath,'KIC%s.csv'%target)
    df = pd.read_csv(rpath)
    lctime = np.array(df['time'])
    lcflux = np.array(df['flux'])
    lcerr = np.array(df['flux_err'])
    lc = lk.LightCurve(time = lctime, flux = lcflux, flux_err = lcerr)
    return lc

def prelim(lc,sigma = 5,width = 50):
    lc2 =lc.remove_outliers(sigma) #remove 
    lc3,R,c = highpass(lc2, width)
    lc4 = fill_gap(lc3)
    return lc4

def prelim2(lc,sigma = 5,width = 50):
    lc2 =lc.remove_outliers(sigma) #remove 
    lc3,R,c = highpass(lc2, width)
    lc4 = interpolate(lc3)
    return lc4

def highpass(lc, width):
     ###CAN INCREASE  SPEED OF COMPUTATION
    #lc = lightcurve_concatenate(target)
    lc = lc
    
    dt = 29.43/(60*24)
    half_wday = round(width/2)
    
    #w = 2200#49*45#50 #width about 50 days
    last = max(lc.time.value) 
    front = np.arange(0,half_wday+0.0001,dt)*(-1)
    front_t = front[::-1]
    back_t = np.arange(last+dt,last+half_wday,dt)
    time_padded = np.append(front_t, lc.time.value)
    time_padded = np.append(time_padded, back_t)
    
    front_f = np.ones(len(front_t))*lc.flux.value[0]
    back_f = np.ones(len(back_t))*lc.flux.value[-1]
    flux_padded = np.append(front_f,lc.flux.value)
    flux_padded =  np.append(flux_padded,back_f)
    last_ind = np.where(time_padded == last)[0][0]
    
    c = np.zeros(len(lc.flux.value))
    ##for loop
    for i in range(len(front_t),last_ind+1):
        ind = np.logical_and(time_padded> time_padded[i]-half_wday, time_padded < time_padded[i]+half_wday)
        j = i-len(front_t)
        c[j] = np.sum(flux_padded[ind])/len(flux_padded[ind])
   

    R = lc.flux.value - c
    lcn = lk.LightCurve(time = lc.time.value,flux = R, flux_err = lc.flux_err)
    
    return lcn,R,c  #return only the deducted light curve


    
def no_outlier(lc,sigma=5):
    lc_no_outlier =lc.remove_outliers(sigma)
     
    return lc_no_outlier
    
    
def boundary(lc,limitsigma):
    if limitsigma ==3:
        sigma = 3
    sigma = limitsigma
    lc_withedge = lc.remove_outliers(sigma)
    return lc_withedge

def fill_gap(lc):
    n,bins= np.histogram(lc.flux.value,100, density=True)
    
    ###PATH---sfig:where to save white noise distri. of lc
    #sfig = r'C:\Users\Luyuting\Project\photovariation\lightcurve\type3\nosigma\noise'
    #sfig = os.path.join(sfig,'KIC%s.png'%target)
    #plt.savefig(sfig)
    #plt.close()
    x,y = three_val(n,bins)
    noise_sig = (abs(x[0]-x[1])+abs(x[1]-x[2]))/2
    lctime = lc.time.value - lc.time.value[ 0]
    lcflux = lc.flux.value
    flux = np.array([])
    time = np.arange(0, lctime[-1]+0.01, 29.4/(60*24))
    half_width = 0.5*29.4/(24*60)
    for i in range(0,len(time)):
        upperbound = time[i] + half_width
        lowerbound = time[i] - half_width
        condition1 = lctime <=upperbound
        condition2 = lctime > lowerbound
        index = np.where(condition1 & condition2) #tuple to expand
        if len(index[0])>0:
            sample_flux = lcflux[index]
            value = np.mean(sample_flux)/len(sample_flux)
            flux = np.insert(flux, i, value)
        else:
            flux = np.insert(flux, i, 0+np.random.normal(0, noise_sig, 1))    ###changed 0 to 1  &&  +np.random.normal(0, 0.05, 1)
    lcn = lk.LightCurve(time = time, flux = flux)
    #lc.flux.value = flux
    #lc.time.value = time
    #lc.flux_err = flux_err
    return lcn

def interpolate(lc):
    sep = lc.time.value[1]-lc.time.value[0]
    start = lc.time.value[0]
    end = lc.time.value[-1]
    lctime = lc.time.value
    lcflux = lc.flux.value
    if sep*60*24>30:
        print('Padding problem')
    else:
        ok = 1
    ###############################
    time = np.arange(start,end,sep)
    flux = []
    ref = []
    n = 0
    for i in range(len(time)):
        ind = np.logical_and(lctime>=(time[i]-sep/2),lctime<(time[i]+sep/2))
        if len(lcflux[ind])>0:
            flux.append(np.mean(lcflux[ind]))
            ref.append(-1)
        else:
            flux.append(np.nan)
            ref.append(n)
            if i <len(time)-1:
                testind = np.logical_and(lctime>=(time[i+1]-sep/2),lctime<(time[i+1]+sep/2))
                if len(lcflux[testind])>0:
                    n = n+1
            else:
                popo = 0
    #############################
    ref_list = list(dict.fromkeys(ref))[1:]
    ref = np.array(ref)
    for item in ref_list:
        idx_nf = np.argwhere(ref==item).reshape(np.argwhere(ref==item).shape[0],) # where to put new flux
        if len(flux)-1 in idx_nf:
            for each in idx_nf:
                flux[each] = flux[int(min(idx_nf)-1)]+np.random.normal(0,10,1)
        else:
            before = idx_nf[0]-1
            after = idx_nf[-1]+1
            val = (flux[after]-flux[before])/len(idx_nf)
            ct = 1
            for each in idx_nf:
                flux[each] = flux[before] + val*ct +np.random.normal(0,10,1)
                ct = ct + 1
    lcn = lk.LightCurve(time = time, flux = flux)
    #lc.flux.value = flux
    #lc.time.value = time
    #lc.flux_err = flux_err
    return lcn

def noise(lc):
    sep = lc.time.value[1]-lc.time.value[0]
    start = lc.time.value[0]
    end = lc.time.value[-1]
    lctime = lc.time.value
    lcflux = lc.flux.value
    std = np.std(lc.flux.value)
    if sep*60*24>30:
        print('Padding problem')
    else:
        ok = 1
    ###############################
    time = np.arange(start,end,sep)
    flux = []
    ref = []
    n = 0
    for i in range(len(time)):
        ind = np.logical_and(lctime>=(time[i]-sep/2),lctime<(time[i]+sep/2))
        if len(lcflux[ind])>0:
            flux.append(np.mean(lcflux[ind]))
            ref.append(-1)
        else:
            flux.append(np.nan)
            ref.append(n)
            if i <len(time)-1:
                testind = np.logical_and(lctime>=(time[i+1]-sep/2),lctime<(time[i+1]+sep/2))
                if len(lcflux[testind])>0:
                    n = n+1
            else:
                popo = 0
    #############################
    ref_list = list(dict.fromkeys(ref))[1:]
    ref = np.array(ref)
    for item in ref_list:
        idx_nf = np.argwhere(ref==item).reshape(np.argwhere(ref==item).shape[0],) # where to put new flux
        if len(flux)-1 in idx_nf:
            for each in idx_nf:
                flux[each] = np.random.normal(0,100,1)
        else:
            before = idx_nf[0]-1
            after = idx_nf[-1]+1
            val = (flux[after]-flux[before])/len(idx_nf)
            ct = 1
            for each in idx_nf:
                flux[each] = np.random.normal(0,100,1)
                ct = ct + 1
    lcn = lk.LightCurve(time = time, flux = flux)
    #lc.flux.value = flux
    #lc.time.value = time
    #lc.flux_err = flux_err
    return lcn
def three_val(n,b):
    y = []
    x = []
    add = 0
    d = b[2]-b[1]
    for t in range(0,len(n)):
        add = sum(n[0:t])*d
        if add>0.16:
            lower_i = t
            break
    #y.append(n[lower_i])
    #x.append((b[lower_i]+b[lower_i + 1])/2)
    y.append(n[lower_i-1])
    x.append(b[lower_i-1])
    
    for t in range(0,len(n)):
        add = sum(n[0:t])*d
        if add>0.5:
            peak_i = t
            break
    #y.append(n[peak_i])
    #x.append((b[peak_i]+b[peak_i + 1])/2)
    y.append(n[peak_i-1])
    x.append(b[peak_i-1])

    for t in range(0,len(n)):
        add = sum(n[0:t])*d
        if add>0.84:
            upper_i = t
            break
    #y.append(n[upper_i])
    #x.append((b[upper_i]+b[upper_i + 1])/2)
    y.append(n[upper_i-1])
    x.append(b[upper_i-1])   ####
    return x,y

def int_str(arr):
    for i in range(len(arr)):
        arr[i] = int(arr[i])
    return arr

def read_lc(name,target):
    rpath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/result/LS/%s/KIC%s'%(name,target)
    rpath = os.path.join(rpath,'lc.csv')
    df = pd.read_csv(rpath)
    time,flux = np.array(df['time']),np.array(df['flux'])
    lc = lk.LightCurve(time = time, flux = flux)
    return lc
#%% # Section: Lomb Scargle
###################LombScargle#################
def periodogram(lc,way='ls'):
    #########Astropy########
    if way =='acf':
        ls = LombScargle(lc.time.value,lc.flux.value,normalization='standard',
            fit_mean=True)
    else:
        ls = LombScargle(lc.time.value,lc.flux.value, normalization='standard',
            fit_mean=True)
    pgfrequency, pgpower = ls.autopower(normalization='standard',
            minimum_frequency=0.014,maximum_frequency=4,samples_per_peak=20)

    #print(pgpower.max())  
    #fap = ls.false_alarm_level([0.01/100]) 
    #print(fap)
    pgpower = pgpower
    pgperiod = 1/pgfrequency
    
    #########Smoothing########
    scale = 24 * 3600 
    fmin = 0.1 * 10**(-6) * scale   #in 1/d 
    df = pgfrequency[1]-pgfrequency[0]
    add = np.zeros(208)
    f_start = (pgfrequency[0]-np.arange(208)*df-df)[::-1]
    f_end = pgfrequency[-1]+np.arange(208)*df+df
    freq = np.append(f_start,pgfrequency)
    freq = np.append(freq,f_end)
    y = np.append(add,pgpower)     
    y = np.append(y,add)     
    half_width = fmin/2
    c = np.zeros(len(pgpower))

    for i in range(0, len(c)):
        j = i+208
        ind = np.logical_and(freq> freq[j]-half_width, freq < freq[j]+half_width)
        c[i] = np.sum(y[ind])/len(y[ind])
    pgpower_smoothed = c
    ##Note that 1 Hz = 24*3600 1/d
    return pgfrequency,pgperiod,pgpower, pgpower_smoothed

def plot_LS(pgpower,period,power,way = 'ls',pic = 'only'):
    ##Compute max
    index = np.argmax(power)
    period_at_maxpower = period[index]
    maxpower = power[index]
    halfmax = power[index]/2
    
    i,maxlen = index,len(period)-1
    while i > 0:
        if  power[i] < halfmax:
            upper_t = (period[i] + period[i + 1]) / 2
            lower_indice = i
            i = index
            break
        else:
            i = i - 1
    while i < maxlen:
        if  power[i] < halfmax:
            lower_t = (period[i] + period[i - 1]) / 2
            upper_indice = i
            break
        else:
            i = i + 1

    ##Text
    py = max(pgpower)
    pmax = period_at_maxpower
    ut = upper_t - period_at_maxpower #upper bound
    lt = period_at_maxpower - lower_t #lower bound

    plt.plot(period, pgpower,'gray')
    plt.plot(period, power,color = 'k',linewidth = 3)
    fonts = [10,15,20,25,30,35,40]
    plt.text(32,py,r'$P_{\rm ACF}=%.2f^{+%.2f}_{-%.2f}(\rm days)$'%(pmax,ut,lt),fontsize = fonts[4],color = 'k') #30
    if way =='acf':
        plt.text(0.5,py,'Periodogram of ACF',fontsize = fonts[4],color = 'k')
    else:
        plt.text(0.5,py,'LS Periodogram',fontsize = fonts[4],color = 'k')
    plt.xlabel('period(days)',fontsize = fonts[3])
    if pic == 'only':
        plt.ylabel('Normalized Power',fontsize = fonts[3])
    else:
        plt.ylabel('')
    plt.rc('xtick',labelsize= fonts[2])
    plt.rc('ytick',labelsize= fonts[2])
    #plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    ##Fill halfmaximum area
    x_fill = period[int(lower_indice):int(upper_indice)+1]
    y_fill = power[int(lower_indice):int(upper_indice)+1]
    plt.fill_between(x_fill, y_fill, y2=0,color = 'lightcyan')#'lightcyan'
    
    ##Fill maximum line
    x_max = np.ones(2)*period[np.argmax(power)]
    y_max = np.array([max(power),0])
    plt.plot(x_max,y_max,'k--', linewidth = 2.5)
    plt.xlim(0,70)
    plt.ylim((0, py*5/4)) 
#    for i in range(0,len(planet_p)):
#        plt.annotate('planet', xy=(planet_p[i], py*6/10), xytext=(planet_p[i], py*8/10),
#                arrowprops=dict(facecolor='blue',edgecolor = 'b',width = 2, headwidth = 4),fontsize = 15)

    return pmax, ut, lt
#%% # Section: ACF
###################ACF#################
def ACF_co(lc,portion = 1/2,std = 30): ##smooth1 = 50, smooth2 = 30
    s = pd.Series(lc.flux.value)
    length = int(round(len(lc.flux.value)*portion,1))
    acf = np.array([])
    for x in range(1,length):
        c = s.autocorr(lag = x) #compute autocorrelation coefficient
        acf = np.append(acf, c)
    
    lag = lc.time.value[ :len(acf)]
    #acf_smooth = gaussian_filter1d(acf, std)
    acf_smooth = acf #need to be changed
    return lag, acf, acf_smooth

def crestloc(arr,time):
    maxt = []
    maxp = []
    i,maxlen = 0,len(arr)
    while i < maxlen-1:
        if arr[i] > arr[i+1]:
            if arr[i-1] < arr[i]:
                maxt.append((time[i]))
                maxp.append((arr[i]))
                i = i + 1
            else:
                i = i + 1
        else:
            i = i + 1
        if len(maxt)>=9: #5
            break
    return maxt[1:],maxp[1:] 

def trough(arr,time):
    maxt = []
    maxp = []
    i,maxlen = 0,len(arr)
    while i < maxlen-1:
        if arr[i] < arr[i+1]:
            if arr[i-1] > arr[i]:
                maxt.append((time[i]))
                maxp.append((arr[i]))
                i = i + 1
            else:
                i = i + 1
        else:
            i = i + 1
        if len(maxt)>=5:
            break
    return maxt,maxp

def ACF_plot(name,target):
    #read path
    rpath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/result/ACF/%s/KIC%s'%(name,target)
    path = os.path.join(rpath,'ACF.csv')
    df = pd.read_csv(path)
    lag = np.array(df['lag'])
    acf = np.array(df['acf'])
    acfsmooth = np.array(df['acfsmooth'])
    ##find period
    maxt,maxp = crestloc(acfsmooth,lag)
    mint,minp = trough(acfsmooth,lag)
    ind2 = np.argmax(maxp)
    ACFp2 = maxt[ind2] #max peak
    ACFp3 = maxt[0] #first peak
    
    ##plot
    plt.figure(figsize = (18,16))
    plt.subplot(311)
    plt.title('KIC%s'%target,fontsize = 30)
    plt.plot(lag,acf,color = 'k',linewidth = 3)
    plt.ylabel('ACF',fontsize = 25)
    plt.xlim(0,70)
    plt.xticks(color = 'w')
    plt.subplot(312)
    plt.xticks(color = 'w')
    plt.plot(lag,acfsmooth,color = 'k',linewidth = 3)
    plt.ylabel('Smoothed ACF',fontsize = 25)
    plt.scatter(maxt,maxp,s = 4,color = 'r')
    plt.scatter(mint,minp,s = 4,color = 'r')
    plt.text(maxt[0]*0.7,maxp[0]*1.2,'$P_{0}$=%.2fd'%maxt[0], fontsize = 25)
    plt.axvline(x=maxt[0],color = 'r', linestyle = '--')
    if ind2 != 0:
        plt.text(maxt[ind2],maxp[ind2]*1.2,'$P_{max}$=%.2fd'%maxt[ind2], fontsize = 25)
        plt.axvline(x=maxt[ind2],color = 'r', linestyle = '--')
    plt.xlim(0,70)
    plt.subplot(313)
    rpath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/result/ACF/%s/KIC%s'%(name,target)
    rpath = os.path.join(rpath,'spectrum.csv') 
    df = pd.read_csv(rpath)
    pgpower,pgperiod,pgpower_smoothed = np.array(df['power']),np.array(df['period']),np.array(df['powersmoothed'])
    pmax, upper, lower = plot_LS(pgpower,pgperiod,pgpower_smoothed,'acf')
    plt.subplots_adjust(hspace=0) 
    return pmax, upper, lower, ACFp2, ACFp3 

def ACF_plot_v2(name,target):
    #read path
    rpath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/result/ACF/%s/KIC%s'%(name,target)
    path = os.path.join(rpath,'ACF.csv')
    df = pd.read_csv(path)
    lag = np.array(df['lag'])
    acf = np.array(df['acf'])
    acfsmooth = np.array(df['acfsmooth'])
    ##find period
    maxt,maxp = crestloc(acfsmooth,lag)
    mint,minp = trough(acfsmooth,lag)
    ind2 = np.argmax(maxp)
    ACFp2 = maxt[ind2] #max peak
    ACFp3 = maxt[0] #first peak
    
    ##plot
    plt.figure(figsize = (18,18))
    gs1 = gridspec.GridSpec(3, 1)
    gs1.update(left=0.1, right=0.9, wspace=0.5, hspace=0)
    ax1 = plt.subplot(gs1[0, 0])
    #plt.subplot(311)
    plt.title('KIC%s'%target,fontsize = 30)
    plt.plot(lag,acf,color = 'k',linewidth = 3)
    plt.ylabel('ACF',fontsize = 25)
    plt.xlim(0,70)
    plt.xticks(color = 'w')
    
    #plt.subplot(312)
    ax2 = plt.subplot(gs1[1:2, 0], sharex=ax1)
    #plt.xticks(color = 'w')
    plt.plot(lag,acfsmooth,color = 'k',linewidth = 3)
    plt.ylabel('Smoothed ACF',fontsize = 25)
    plt.xlabel(r'$\tau$(days)',fontsize = 25)
    plt.scatter(maxt,maxp,s = 4,color = 'r')
    plt.scatter(mint,minp,s = 4,color = 'r')
    plt.text(maxt[0]*0.7,maxp[0]*1.2,'$P_{0}$=%.2fd'%maxt[0], fontsize = 25)
    plt.axvline(x=maxt[0],color = 'r', linestyle = '--')
    if ind2 != 0:
        plt.text(maxt[ind2],maxp[ind2]*1.2,'$P_{max}$=%.2fd'%maxt[ind2], fontsize = 25)
        plt.axvline(x=maxt[ind2],color = 'r', linestyle = '--')
    plt.xlim(0,70)
    
    #plt.subplot(313)
    gs2 = gridspec.GridSpec(3, 1)
    gs2.update(left=0.1, right=0.9, hspace=0.7)
    #ax2 = plt.subplot(gs2[0, 0], sharey=ax2)
    ax3 = plt.subplot(gs2[2, 0], sharex=ax2)
    rpath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/result/ACF/%s/KIC%s'%(name,target)
    rpath = os.path.join(rpath,'spectrum.csv') 
    df = pd.read_csv(rpath)
    pgpower,pgperiod,pgpower_smoothed = np.array(df['power']),np.array(df['period']),np.array(df['powersmoothed'])
    pmax, upper, lower = plot_LS(pgpower,pgperiod,pgpower_smoothed,'acf')
    plt.subplots_adjust(hspace=0) 
    return pmax, upper, lower, ACFp2, ACFp3 


#%% # Section: Wavelet
###################Wavelet#################
def plot_wavelet(name,target,pic = 'only'):
    fonts = [10,15,20,25,30,35,40]
    rpath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/result/wavelet/%s/KIC%s'%(name,target)
    path = os.path.join(rpath,'gwspectrum.csv')
    df = pd.read_csv(path)
    period = np.array(df['period'])[::-1]
    power = np.array(df['global_ws'])[::-1]
    sig = np.array(df['signif'])
    ##Compute max
    index = np.argmax(power)
    period_at_maxpower = period[index]
    maxpower = power[index]
    halfmax = power[index]/2
    
    i,maxlen = index,len(period)-1
    while i > 0:
        if  power[i] < halfmax:
            upper_t = (period[i] + period[i + 1]) / 2
            lower_indice = i
            i = index
            break
        else:
            i = i - 1
    while i < maxlen:
        if  power[i] < halfmax:
            lower_t = (period[i] + period[i - 1]) / 2
            upper_indice = i
            break
        else:
            i = i + 1

    ##Text
    py = max(power)
    pmax = period_at_maxpower
    ut = upper_t - period_at_maxpower #upper bound
    lt = period_at_maxpower - lower_t #lower bound

    plt.plot(period, power,'k',linewidth = 3)
    #plt.plot(period, sig,'r',linestyle = '--')
    fonts = [10,15,20,25,30,35,40]
    plt.text(32,py,r'$P_{\rm rot,phot}=%.2f^{+%.2f}_{-%.2f}(\rm days)$'%(pmax,ut,lt),fontsize = fonts[4],color = 'k') #30
    plt.text(0.5,py,'Global Wavelet Spectrum',fontsize = fonts[4],color = 'k')
    plt.xlabel('period(days)',fontsize = fonts[3])
    if pic == 'only':
        plt.ylabel('Normalized Power',fontsize = fonts[3])
    else:
        plt.ylabel('')
    plt.rc('xtick',labelsize= fonts[2])
    plt.rc('ytick',labelsize= fonts[2])
    #plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    ##Fill halfmaximum area
    x_fill = period[int(lower_indice):int(upper_indice)+1]
    y_fill = power[int(lower_indice):int(upper_indice)+1]
    plt.fill_between(x_fill, y_fill, y2=0,color = 'lightcyan')#'lightcyan'sliver)
    
    ##Fill maximum line
    x_max = np.ones(2)*period[np.argmax(power)]
    y_max = np.array([max(power),0])
    plt.plot(x_max,y_max,'k--', linewidth = 2.5) #'b-'
    plt.xlim(0,65)
    plt.ylim((0, py*5/4)) 
#    for i in range(0,len(planet_p)):
#        plt.annotate('planet', xy=(planet_p[i], py*6/10), xytext=(planet_p[i], py*8/10),
#                arrowprops=dict(facecolor='blue',edgecolor = 'b',width = 2, headwidth = 4),fontsize = 15)

    return pmax, ut, lt
#%% # Section: read values
################Read Values##############
def read_val(tar):
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/LS_value.csv'
    df = pd.read_csv(spath)
    number = np.array(df['No.'])
    alltarget = np.array(df['target'])
    LS_p = np.array(df['LS_p'])
    LS_up = np.array(df['LS_up'])
    LS_low = np.array(df['LS_low'])
    tag = np.array(df['tag'])
    KOI = np.array(df['KOI'])
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/ACF_value.csv'
    df = pd.read_csv(spath)
    ACF_p = np.array(df['ACF_p'])
    ACF_up = np.array(df['ACF_up'])
    ACF_low = np.array(df['ACF_low'])
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/Wavelet_value.csv'
    df = pd.read_csv(spath)
    W_p = np.array(df['W_p'])
    W_up = np.array(df['W_up'])
    W_low = np.array(df['W_low'])
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/Astero_value.csv'
    df = pd.read_csv(spath)
    AS_p = np.array(df['AS_p'])
    AS_up = np.array(df['AS_up'])
    AS_low = np.array(df['AS_low'])
    asi_val = np.array(df['asi'])
    asi_up = np.array(df['asiup'])
    asi_low = np.array(df['asilow'])
    
    #Changed from quarter to Q_noise
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/MAD/Q_noise_smooth.csv'#spath' = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/quarter.csv'
    df = pd.read_csv(spath)
    med = np.array(df['Median'])#np.array(df['median'])
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/MAD/MAD(group8).csv'
    df = pd.read_csv(spath)
    med_up = np.array(df['up'])
    med_low = np.array(df['low'])
    ######
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/widthxratio.csv'
    df = pd.read_csv(spath)
    d_val = np.array(df['div'])
    meana_back = np.array(df['MEAD'])  #olF:MEANAr
    
    qnum_back = np.array(df['qnum'])
    scatter_back = np.array(df['scatter'])
    KID = np.array(df['Kepler'])
    Reference = np.array(df['Ref'])
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/Sph.csv'
    df = pd.read_csv(spath)
    globalS = np.array(df['globalS']) 
    localS = np.array(df['localS'])
    
    ###Dnu Ratio###
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/AS_dnur.csv'
    df = pd.read_csv(spath)
    dnur_back = np.array(df['dnur'])
    
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/other.csv'
    df = pd.read_csv(spath)
    S_T = np.array(df['S_T'])
    S_Terr = np.array(df['S_Terr'])
    #######compute spectro sini######
    sini_val, sini_up, sini_low=compute_spectro(LS_p,LS_up,LS_low)
    #######compute astero i######
    ###
    ind = []
    group = []
    LS = []
    ACF = []
    W = []
    AS = []
    div = []
    meanar_group = []
    dnur = []
    spec_sini = []
    asi = []
    T = []
    KOIval = []
    Kepler = []
    newtag = []
    Sph= []
    medall = []
    qnumall = []
    scatterall = []
    Refs = []
    for item in tar:
        #print(item)
        group.append(tar)
        ind = np.argwhere(alltarget==item)[0][0]
        LS = LS + [LS_p[ind],LS_low[ind],LS_up[ind]]
        ACF = ACF + [ACF_p[ind],ACF_low[ind],ACF_up[ind]]
        W = W + [W_p[ind],W_low[ind],W_up[ind]]
        AS = AS + [AS_p[ind],AS_low[ind],AS_up[ind]]
        spec_sini = spec_sini + [sini_val[ind],sini_low[ind],sini_up[ind]]
        asi = asi + [asi_val[ind],asi_low[ind],asi_up[ind]] 
        T = T + [S_T[ind],S_Terr[ind],S_Terr[ind]]
        medall = medall + [med[ind],med_low[ind],med_up[ind]]
        #medall.append(med[ind])
        newtag.append(tag[ind])
        div.append(d_val[ind])
        meanar_group.append(meana_back[ind])
        dnur.append(dnur_back[ind])
        Sph.append(globalS[ind])
        Sph.append(localS[ind]) 
        qnumall.append(qnum_back[ind])
        scatterall.append(scatter_back[ind])
        Refs.append(Reference[ind])
        #################
        if np.isnan(KOI[ind]):
            KOIval.append('') 
            Kepler.append('')
        else:
            KOIval.append('KOI-%d'%(KOI[ind]))
            try:
                int(KID[ind])
            except ValueError:
                Kepler.append('') 
            else:
                Kepler.append('$Kepler$ - %s'%KID[ind])
        ################
    count = len(tar)    
    LS = np.array(LS).reshape(count,3)    
    ACF = np.array(ACF).reshape(count,3) 
    W = np.array(W).reshape(count,3)   
    AS = np.array(AS).reshape(count,3)      
    spec_sini = np.array(spec_sini).reshape(count,3)   
    asi = np.array(asi).reshape(count,3) 
    T = np.array(T).reshape(count,3)
    medall = np.array(medall).reshape(count,3)
    Sph= np.array(Sph).reshape(count,2)
    
    
    return LS, ACF, W, AS, newtag, div, meanar_group, dnur, count, spec_sini, asi,T,KOIval,Sph,medall,scatterall,qnumall,Kepler,Refs   # To add, T, i,cosi

def read_special(tar):
    special = np.array([8866102,6278762])
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/special_KOI.csv'
    df = pd.read_csv(spath)
    alltarget = np.array(df['target'])
    LS_p = np.array(df['LS_p'])
    LS_up = np.array(df['LS_up'])
    LS_low = np.array(df['LS_low'])
    ACF_p = np.array(df['ACF_p'])
    ACF_up = np.array(df['ACF_up'])
    ACF_low = np.array(df['ACF_low'])
    W_p = np.array(df['W_p'])
    W_up = np.array(df['W_up'])
    W_low = np.array(df['W_low'])
    AS_p = np.array(df['AS_p'])
    AS_up = np.array(df['AS_up'])
    AS_low = np.array(df['AS_low'])
    med = np.array(df['Median'])
    LS = []
    ACF = []
    W = []
    AS = []
    for ind in range(2):
        #print(item)
        ind = np.argwhere(special==tar[ind])[0][0]
        LS = LS + [LS_p[ind],LS_low[ind],LS_up[ind]]
        ACF = ACF + [ACF_p[ind],ACF_low[ind],ACF_up[ind]]
        W = W + [W_p[ind],W_low[ind],W_up[ind]]
        AS = AS + [AS_p[ind],AS_low[ind],AS_up[ind]]
    count = len(tar)    
    LS = np.array(LS).reshape(count,3)    
    ACF = np.array(ACF).reshape(count,3) 
    W = np.array(W).reshape(count,3)   
    AS = np.array(AS).reshape(count,3)     
    return LS, ACF, W, AS, med
#### read planets
def read_planet(target):

    spath = r'F:/Home/Project/photovariation/LS/33KOI.csv'
    df = pd.read_csv(spath)
    tl = np.array(df['target'])
    p1 = np.array(df['exo1'])
    pup1 = np.array(df['exoup1'])
    plow1 = np.array(df['exolow1'])
    p2 = np.array(df['exo2'])
    pup2 = np.array(df['exoup2'])
    plow2 = np.array(df['exolow2'])
    p3 = np.array(df['exo3'])
    pup3 = np.array(df['exoup3'])
    plow3 = np.array(df['exolow3'])
    p4 = np.array(df['exo4'])
    pup4 = np.array(df['exoup4'])
    plow4 = np.array(df['exolow4'])
    p5 = np.array(df['exo5'])
    pup5 = np.array(df['exoup5'])
    plow5 = np.array(df['exolow5'])
    ind = np.argwhere(tl == target)[0][0]
    prob = []
    if p1[ind]>0:
        prob.append([p1[ind],plow1[ind],pup1[ind]])
    if p2[ind]>0:
        prob.append([p2[ind],plow2[ind],pup2[ind]])
    if p3[ind]>0:
        prob.append([p3[ind],plow3[ind],pup3[ind]])
    if p4[ind]>0:
        prob.append([p4[ind],plow4[ind],pup4[ind]])
    if p5[ind]>0:
        prob.append([p5[ind],plow5[ind],pup5[ind]])
    return prob



def read_other(tar):
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/asterovalue.csv'
    df = pd.read_csv(spath)
    alltarget = np.array(df['target'])
    HBR = np.array(df['HBR'])
    nuratio = np.array(df['nuratio'])
    HBR_val = []
    nuratio_val = []
    for item in tar:
        #print(item)
        ind = np.argwhere(alltarget==item)[0][0]
        HBR_val.append(HBR[ind])
        nuratio_val.append(nuratio[ind])        
    
    return HBR_val, nuratio_val
#### read astero values
def compute_spectro(LS_p,LS_up,LS_low): #use LS P
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/other.csv'
    df = pd.read_csv(spath)
    vsin = np.array(df['vsin'])
    vsinerr = np.array(df['vsinerr'])
    specR = np.array(df['specR'])
    specRerr = np.array(df['specRup'])
    fac = 24*2600/696340
    sini_val = fac*vsin*LS_p/(2*np.pi*specR)
    sini_up = np.sqrt((LS_up/LS_p)**2+(specRerr/specR)**2+(vsinerr/vsin)**2)*sini_val
    sini_low = np.sqrt((LS_low/LS_p)**2+(specRerr/specR)**2+(vsinerr/vsin)**2)*sini_val
    
    sini_val = np.arcsin(fac*vsin*LS_p/(2*np.pi*specR))*180/np.pi
    sini_up = np.sqrt((LS_up/LS_p)**2+(specRerr/specR)**2+(vsinerr/vsin)**2)*sini_val
    sini_low = np.sqrt((LS_low/LS_p)**2+(specRerr/specR)**2+(vsinerr/vsin)**2)*sini_val
    return sini_val, sini_up, sini_low


def sintoi(spec_sini):
    i = np.arcsin()
    return i

####read previous literature
def read_PL(tar):
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/value_PL.csv'
    df = pd.read_csv(spath)
    alltarget = np.array(df['target'])
    G0 = np.array(df['G'])

    Gerr0 = np.array(df['Gerr'])

    C0 = np.array(df['C'])

    Cerr0 = np.array(df['Cerr'])

    M0 = np.array(df['M'])
    Merr0 = np.array(df['Merr'])
    K0 = np.array(df['K'])
    Kerr0 = np.array(df['Kerr'])
    N0 = np.array(df['N13'])
    Nerr0 = np.array(df['Nerr13'])
    G = []
    Gerr = []
    C = []
    Cerr = []
    M = []
    Merr = []
    K = []
    Kerr = []
    N = []
    Nerr = []
    for item in tar:
        ind = np.argwhere(alltarget==item)[0][0]
        G.append(G0[ind])
        Gerr.append(Gerr0[ind])
        C.append(C0[ind])
        Cerr.append(Cerr0[ind])
        M.append(M0[ind])
        Merr.append(Merr0[ind])
        K.append(K0[ind])     
        Kerr.append(Kerr0[ind])  
        N.append(N0[ind])     
        Nerr.append(Nerr0[ind]) 
    return G, Gerr, C, Cerr, M, Merr, K, Kerr, N, Nerr
#%%Classification: area under curve
def sort_arr(arr1,arr2,alltarget):
    group1 = []
    tag1 = []
    group2 = []
    tag2 = []
    list2 = []
    arr1 = np.sort(arr1)
    for item in arr1:
        ind = np.argwhere(alltarget==item)[0][0]
        list2.append(arr2[ind])
    list1 = arr1.tolist()
    list1, list2 = zip(*sorted(zip(list1, list2)))
    list1 = np.asarray(list1)
    list2 = np.asarray(list2)
    for i in range(len(list1)):
        if list2[i] == 'nonKOI':
            group2.append(list1[i])
            tag2.append(list2[i])
        if list2[i] == 'KOI':
            group1.append(list1[i])    
            tag1.append(list2[i])
    return group1, group2, tag1, tag2

def classification(criteria,alltarget):
    A = []
    B = []
    C = []
    D = []
    for i in range(len(alltarget)):
        if criteria[i] < 1:
            A.append(alltarget[i])
        elif criteria[i] >= 1 and criteria[i] < 1.7:
            B.append(alltarget[i])
        elif criteria[i] >= 1.7 and criteria[i] < 2.4:
            C.append(alltarget[i])
        elif criteria[i] >= 2.4:
            D.append(alltarget[i])
    return A, B, C, D

def group():
    gpath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/class.csv' 
    df2 = pd.read_csv(gpath)
    alltarget = np.array(df2['target'])
    grouping = np.array(df2['group'])
    A = []
    B = []
    C = []
    D = []
    if alltarget[0] == 8077137:
        for i in range(len(alltarget)):
            if grouping[i] == 'A':
                A.append(alltarget[i])
            if grouping[i] == 'B':
                B.append(alltarget[i])
            if grouping[i] == 'C':
                C.append(alltarget[i])
            if grouping[i] == 'D':
                D.append(alltarget[i])                
    else:
        print('ERROR: grouping target disorder!')
    return A, B, C, D

def group_photo():
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/widthxratio.csv'
    df = pd.read_csv(spath)
    alltarget = np.array(df['target'])
    grouping = np.array(df['class2'])
    single,multi = group_multiple()
    R = []
    U = []
    if alltarget[0] == 8077137:
        for i in range(len(alltarget)):
            if grouping[i] == 'R':
                R.append(alltarget[i])
            if grouping[i] == 'U':
                if alltarget[i] not in multi:
                    U.append(alltarget[i])            
    else:
        print('ERROR: grouping target disorder!')
    return R,U

def group_multiple():
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/widthxratio.csv'
    df = pd.read_csv(spath)
    alltarget = np.array(df['target'])
    #grouping = np.array(df['companion'])
    group2 = np.array(df['group7'])
    S = []
    M = []
    if alltarget[0] == 8077137:
        for i in range(len(alltarget)):
            if group2[i] != 'E':
                S.append(alltarget[i])
            else:
                M.append(alltarget[i])  
            #if grouping[i]<1 and group2[i] != 'E':
            #    S.append(alltarget[i])
            #elif grouping[i] > 0:
            #    M.append(alltarget[i])            
    else:
        print('ERROR: grouping target disorder!')
    return S,M

def group_T():
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/widthxratio.csv'
    df = pd.read_csv(spath)
    alltarget = np.array(df['target'])
    LS, ACF, W, AS, tag, div, meanar, dnur, count, spec_sini, asi,T,KOIval,Sph, med, scatter, qnum, kep, ref = read_val(alltarget)
    F = []
    G = []
    if alltarget[0] == 8077137:
        for i in range(len(alltarget)):
            if ~np.isnan(T[i,0]):
                if T[i,0] <=6000:
                    G.append(alltarget[i])
                elif T[i,0] > 6000:
                    F.append(alltarget[i])            
    else:
        print('ERROR: grouping target disorder!')
    return G,F

def group_P():
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/widthxratio.csv'
    df = pd.read_csv(spath)
    alltarget = np.array(df['target'])
    LS, ACF, W, AS, tag, div, meanar, dnur, count, spec_sini, asi,T,KOIval,Sph, med, scatter, qnum, kep, ref = read_val(alltarget)
    L = []
    S = []
    if alltarget[0] == 8077137:
        for i in range(len(alltarget)):
            if ~np.isnan(T[i,0]):
                if LS[i,0] <=30:
                    S.append(alltarget[i])
                elif LS[i,0] > 30:
                    L.append(alltarget[i])            
    else:
        print('ERROR: grouping target disorder!')
    return S,L



def group_asteronu():
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/widthxratio.csv'
    df = pd.read_csv(spath)
    alltarget = np.array(df['target'])
    grouping = np.array(df['kmk'])
    HBR, nuratio = read_other(alltarget)
    A = []
    Apoor = []
    B = []
    Bpoor = []
    if alltarget[0] == 8077137:
        for i in range(len(alltarget)):
            if grouping[i] == 'A':
                if nuratio[i]>=0.5:
                    A.append(alltarget[i])
                else:
                    Apoor.append(alltarget[i])
            elif grouping[i] == 'B':
                if nuratio[i]>=0.5:
                    B.append(alltarget[i])    
                else:
                    Bpoor.append(alltarget[i])
    else:
        print('ERROR: grouping target disorder!')
    return A,Apoor, B, Bpoor

def group_simple():
    gpath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/class.csv' 
    df2 = pd.read_csv(gpath)
    alltarget = np.array(df2['target'])
    LS, ACF, W, AS, tag, div, meanar, dnur, count, spec_sini, asi,T,KOIval,Sph, med, scatter, qnum, kep, ref = read_val(alltarget)
    koi = []
    non = []
    if alltarget[0] == 8077137:
        for i in range(len(alltarget)):
            if tag[i] == 'KOI':
                koi.append(alltarget[i])
            elif tag[i] == 'nonKOI':
                non.append(alltarget[i])             
    else:
        print('ERROR: grouping target disorder!')
    return koi, non

def group_singlekoi():
    gpath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/class.csv' 
    df2 = pd.read_csv(gpath)
    alltarget = np.array(df2['target'])
    S,M = group_multiple()
    LS, ACF, W, AS, tag, div, meanar, dnur, count, spec_sini, asi,T,KOIval,Sph, med, scatter, qnum, kep, ref = read_val(S)
    koi = []
    non = []
    for i in range(len(S)):
        if tag[i] == 'KOI':
            koi.append(alltarget[i])
        elif tag[i] == 'nonKOI':
            non.append(alltarget[i])       
    return koi, non

def group_planet(tag):
    gpath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/class.csv' 
    df2 = pd.read_csv(gpath)
    alltarget = np.array(df2['target'])
    grouping = np.array(df2['group7'])
    A = []
    B = []
    C = []
    D = []
    Anon = []
    Bnon = []
    Cnon = []
    Dnon = []
    if alltarget[0] == 8077137:
        for i in range(len(alltarget)):
            if grouping[i] == 'A':
                if tag[i] == 'KOI':
                    A.append(alltarget[i])
                else:
                    Anon.append(alltarget[i])
            if grouping[i] == 'B':
                if tag[i] == 'KOI':
                    B.append(alltarget[i])
                else:
                    Bnon.append(alltarget[i])
            if grouping[i] == 'C':
                if tag[i] == 'KOI':
                    C.append(alltarget[i])
                else:
                    Cnon.append(alltarget[i])
            if grouping[i] == 'D':
                if tag[i] == 'KOI':
                    D.append(alltarget[i])
                else:
                    Dnon.append(alltarget[i])                    
    else:
        print('ERROR: grouping target disorder!')
    return A,B,C,D,Anon,Bnon,Cnon,Dnon

def group_wkmk():
    gpath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/widthxratio.csv' 
    df2 = pd.read_csv(gpath)
    alltarget = np.array(df2['target'])
    grouping = np.array(df2['group2']) 
    LS, ACF, W, AS, tag, div, meanar, dnur, count, spec_sini, asi,T,KOIval,Sph, med, scatter, qnum, kep, ref = read_val(alltarget)
    A = []
    B = []
    C = []
    D = []
    Anon = []
    Bnon = []
    Cnon = []
    Dnon = []
    if alltarget[0] == 8077137:
        for i in range(len(alltarget)):
            if grouping[i] == 'A':
                if tag[i] == 'KOI':
                    A.append(alltarget[i])
                else:
                    Anon.append(alltarget[i])
            if grouping[i] == 'B':
                if tag[i] == 'KOI':
                    B.append(alltarget[i])
                else:
                    Bnon.append(alltarget[i])
            if grouping[i] == 'C':
                if tag[i] == 'KOI':
                    C.append(alltarget[i])
                else:
                    Cnon.append(alltarget[i])
            if grouping[i] == 'D':
                if tag[i] == 'KOI':
                    D.append(alltarget[i])
                else:
                    Dnon.append(alltarget[i])                    
    else:
        print('ERROR: grouping target disorder!')
    return A,B,C,D,Anon,Bnon,Cnon,Dnon

def group_notkmk():
    gpath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/widthxratio.csv' 
    df2 = pd.read_csv(gpath)
    alltarget = np.array(df2['target'])
    grouping = np.array(df2['group7']) #group4#group3
    starno = np.array(df2['companion'])
    LS, ACF, W, AS, tag, div, meanar, dnur, count, spec_sini, asi,T,KOIval,Sph, med, scatter, qnum, kep, ref = read_val(alltarget)
    A = []
    B = []
    C = []
    D = []
    Anon = []
    Bnon = []
    Cnon = []
    Dnon = []
    incon = [5773345,8006161,7970740,6225718,3544595,9965715,6225718,3656476,4349452,6521045,9955598]
    if alltarget[0] == 8077137:
        for i in range(len(alltarget)):
            if grouping[i] == 'A':
                if tag[i] == 'KOI':
                    A.append(alltarget[i])
                else:
                    Anon.append(alltarget[i])
            if grouping[i] == 'B':
                if tag[i] == 'KOI':
                    B.append(alltarget[i])
                else:
                    Bnon.append(alltarget[i])
            if grouping[i] == 'C':
                if tag[i] == 'KOI':
                    C.append(alltarget[i])
                else:
                    Cnon.append(alltarget[i])
            if grouping[i] == 'D':
                if tag[i] == 'KOI':
                    D.append(alltarget[i])
                else:
                    Dnon.append(alltarget[i])                    
    else:
        print('ERROR: grouping target disorder!')
    return A,B,C,D,Anon,Bnon,Cnon,Dnon



def group_astero():
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/widthxratio.csv'
    df = pd.read_csv(spath)
    alltarget = np.array(df['target'])
    grouping = np.array(df['Aserror'])
    A = []
    B = []
    if alltarget[0] == 8077137:
        for i in range(len(alltarget)):
            if grouping[i] == 'A':
                A.append(alltarget[i])
            if grouping[i] == 'B':
                B.append(alltarget[i])            
    else:
        print('ERROR: grouping target disorder!')
    return A,B

def group_medcon():
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/widthxratio.csv'
    df = pd.read_csv(spath)
    alltarget = np.array(df['target'])
    grouping = np.array(df['allmedcon2'])
    con = []
    incon = []
    for i in range(len(alltarget)):
        if grouping[i] == 1:
            con.append(alltarget[i])
        if grouping[i] == 0:
            incon.append(alltarget[i])  
    return con, incon

def group_period(p):
    gpath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/class.csv' 
    df2 = pd.read_csv(gpath)
    alltarget = np.array(df2['target'])
    list1 = p.tolist()
    list2 = alltarget.tolist()
    list1, list2 = zip(*sorted(zip(list1, list2)))
    list1 = np.asarray(list1)
    list2 = np.asarray(list2)
    LS, ACF, W, AS, tag, div, meana, dnur, count, spec_sini, asi,T,KOIval,Sph, med, scatter, qnum, kep, ref = read_val(list2)
    A = []
    B = []
    C = []
    Anon = []
    Bnon = []
    Cnon = []   
    allt = list2
    periods = W[:,0]
    print(periods)
    for i in range(len(allt)):
        if periods[i]<15:
            if tag[i] == 'KOI':
                A.append(allt[i])
            else:
                Anon.append(allt[i])
        if periods[i]>=15 and periods[i]<30:
            if tag[i] == 'KOI':
                B.append(allt[i])
            else:
                Bnon.append(allt[i])
        if periods[i]>=30:
            if tag[i] == 'KOI':
                C.append(allt[i])
            else:
                Cnon.append(allt[i])
                 
    return A,B,C,Anon,Bnon,Cnon

def group_any():
    spath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/widthxratio.csv'
    df = pd.read_csv(spath)
    alltarget = np.array(df['target'])
    LS, ACF, W, AS, tag, div, meana, dnur, count, spec_sini, asi,T,KOIval,Sph, med, scatter, qnum, kep, ref = read_val(alltarget)
    A = []
    B = []
    if alltarget[0] == 8077137:
        for i in range(len(alltarget)):
            if LS[i,0] <= 25:
                A.append(alltarget[i])
            elif LS[i,0]> 25:
                B.append(alltarget[i])    
    else:
        print('ERROR: grouping target disorder!')
    return A, B
#%% # PLOTTING SESSION

#Figure 1
def error(up,low):
    err = np.zeros(2*len(up))
    err = err.reshape(2,len(up))
    err[0,:] = low
    err[1,:] = up
    return err

def plot(col,shape,x,y,xup=None,xlow=None,yup = None,ylow = None,col2 = None,zorder = 1):
    if col2 == None:
        col2 == col
        
    try:
        xerr = error(xup,xlow)
    except TypeError:
        print('Neglecting X error bar')
   
    try:
        yerr = error(yup,ylow)
    except TypeError:
        print('Neglecting Y error bar')
   
    #print(x)
        ############## Assign color ################

    ################# PLOT #############
    for i in range(len(x)):
        if xup is None and yup is None:
            plt.scatter(x[i],y[i],marker = shape, color = col, s = 50,facecolors = col2,zorder = 1)
        elif xup is None and yup is not None:
            plt.errorbar(x[i],y[i],yerr = yerr[:,i].reshape(2,1),
                     fmt= shape, ecolor=col, c=col,elinewidth=1.2,
                     capsize=2,markersize = 4, mfc = col2,zorder = zorder)
                    # fmt= shape,ecolor=col,c=col,elinewidth=0.9,
                    #capsize=2,markersize = 5,mfc = col2,zorder = zorder)
        elif xup is not None and yup is None:
            plt.errorbar(x[i],y[i],xerr = xerr[:,i].reshape(2,1),
                     fmt= shape, ecolor=col, c=col,elinewidth=1.2,
                     capsize=2,markersize = 4, mfc = col2,zorder = zorder)
            #fmt= shape,ecolor=col,c=col,elinewidth=0.7,
            #         capsize=2,markersize = 5,label='Wavelet',mfc = col2,zorder = zorder)
        else:
            plt.errorbar(x[i],y[i],xerr = xerr[:,i].reshape(2,1),
            yerr = yerr[:,i].reshape(2,1),fmt= shape,ecolor=col, 
            c=col,elinewidth=1.2, capsize=2,markersize = 4, mfc = col2,zorder = zorder)
# Figure 2
def fig2(select,alltarget,num = 0):
    LS, ACF, W, AS, tag, div, meana, dnur, count = read_val(alltarget)
    y = np.linspace(0.7,(len(select)+1)*1.3,len(select))
    g_KOI, g_nonKOI,tag_KOI, tag_nonKOI = sort_arr(select,tag,alltarget)
    new_sel = g_KOI + g_nonKOI
    new_tag = tag_KOI + tag_nonKOI
    #for ind in range(len(new_sel)):
    for i in range(len(new_sel)):
        if new_tag[i] == 'KOI':
            shape = 'o'
        elif new_tag[i] == 'nonKOI':
            shape = '^'
        ind = np.argwhere(alltarget==new_sel[i])[0][0]
        plt.errorbar(LS[ind,0],y[i]+0.2,xerr=LS[ind,1:].reshape(2,1),fmt=shape,ecolor='g',color='g',elinewidth=3,capsize=7,markersize = 7)
        plt.errorbar(ACF[ind,0],y[i],xerr=ACF[ind,1:].reshape(2,1),fmt=shape,ecolor='r',color='r',elinewidth=3,capsize=7,markersize = 7)
        plt.errorbar(W[ind,0],y[i]-0.2,xerr=W[ind,1:].reshape(2,1),fmt=shape,ecolor='b',color='b',elinewidth=3,capsize=7,markersize = 7)
        plt.errorbar(AS[ind,0],y[i]-0.4,xerr=AS[ind,1:].reshape(2,1),fmt=shape,ecolor='k',color='k',elinewidth=3,capsize=7,markersize = 7)
    #plt.errorbar(asp,y-0.4,xerr=asperr,fmt='o',ecolor='k',color='k',elinewidth=4,capsize=9,markersize = 7, label = 'Astero')
    indice = len(g_KOI)
    print(indice)
    text_line = 0.5*(y[indice]+y[indice-1])
    plt.axhline(text_line,linestyle = '-',color = 'gray')
    
    yname = []
    for i in new_sel:
        name = str(i)
        yname.append(name)
    
    my_yticks = yname
    plt.yticks(y, my_yticks, fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.xlabel('Period (Day)',fontsize = 15)
    plt.ylabel('KIC',fontsize = 15)
    plt.xlim(xmin = 0)
    ##legend
    green_patch = mpatches.Patch(color='g', label='LS')
    red_patch = mpatches.Patch(color='r', label='ACF')
    blue_patch = mpatches.Patch(color='b', label='Wavelet')
    if num == 1:
        plt.legend(handles=[red_patch, green_patch, blue_patch],fontsize = 13)
        plt.text(0.8,y[0]+0.2,'KOI', color = 'gray', fontsize = 20)
        plt.text(0.8,y[-1]-0.12,'nonKOI', color = 'gray', fontsize = 20)
    if num == 4:
        plt.xlim(0,55)
    return

def fig2_ver2(select,alltarget,num = 0):
    LS, ACF, W, AS, tag, div, meana_group, dnur, count, spec_sini, asi,T,KOIval,Sph, med, scatter, qnum, kep, ref = read_val(alltarget)
    y = np.linspace(0.7,(len(select)+1)*1.3,len(select))
    g_KOI, g_nonKOI,tag_KOI, tag_nonKOI = sort_arr(select,tag,alltarget)
    new_sel = g_KOI + g_nonKOI
    new_tag = tag_KOI + tag_nonKOI
    yname = []
    for i in range(len(new_sel)):
        if new_tag[i] == 'KOI':
            shape = 'o'
        elif new_tag[i] == 'nonKOI':
            shape = '^'
            
        ind = np.argwhere(alltarget==new_sel[i])[0][0]
        
        if tag[ind] == 'KOI':
            name = KOIval[ind]
            yname.append(name) 
        elif tag[ind] == 'nonKOI':
            name = str(alltarget[ind])
            yname.append(name)  
       
        ind = np.argwhere(alltarget==new_sel[i])[0][0]
        plt.errorbar(LS[ind,0],y[i]+0.2,xerr=LS[ind,1:].reshape(2,1),fmt=shape,ecolor='g',color='g',elinewidth=3,capsize=7,markersize = 7)
        plt.errorbar(ACF[ind,0],y[i],xerr=ACF[ind,1:].reshape(2,1),fmt=shape,ecolor='r',color='r',elinewidth=3,capsize=7,markersize = 7)
        plt.errorbar(W[ind,0],y[i]-0.2,xerr=W[ind,1:].reshape(2,1),fmt=shape,ecolor='b',color='b',elinewidth=3,capsize=7,markersize = 7)
        plt.errorbar(AS[ind,0],y[i]-0.4,xerr=AS[ind,1:].reshape(2,1),fmt=shape,ecolor='k',color='k',elinewidth=3,capsize=7,markersize = 7)
    #plt.errorbar(asp,y-0.4,xerr=asperr,fmt='o',ecolor='k',color='k',elinewidth=4,capsize=9,markersize = 7, label = 'Astero')
    indice = len(g_KOI)
    print(indice)
    print(new_sel)

   # for j in range(len(new_sel)):
   #     if new_tag[j] == 'KOI':
   #         name = KOIval[j]
   #         yname.append(name)  
   #         print(name)
   #     elif new_tag[j] == 'nonKOI':
   #         name = str(new_sel[j])
   #         yname.append(name)  
    
   # for i in new_sel:
   #     name = str(i)
   #     yname.append(name)
    
    my_yticks = yname
    plt.yticks(y, my_yticks, fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.xlabel('Period (Day)',fontsize = 20)
    for item in y:
        diff = y[2]-y[1]
        plt.axhline(item-0.5*diff,linestyle = '--',color = 'gray')
        
    plt.xlim(xmin = 0)
    if len(new_sel)<2:
        plt.ylim(y[0]-3,y[0]+3)
    ##legend
    green_patch = mpatches.Patch(color='g', label='LS')
    red_patch = mpatches.Patch(color='r', label='ACF')
    blue_patch = mpatches.Patch(color='b', label='Wavelet')
    black_patch = mpatches.Patch(color='k', label='Astero')
    if num == 1:
        plt.xlim(0,33)
        plt.ylabel('KIC',fontsize = 20)
        #plt.legend(handles=[red_patch, green_patch, blue_patch],fontsize = 13)
        plt.legend(handles=[red_patch, green_patch, blue_patch,black_patch],fontsize = 13)        
       # plt.text(22,y[0],'nonKOI', color = 'k', fontsize = 20)
    elif num == 1.5:
        plt.ylabel('KIC',fontsize = 20)
        plt.xlim(0,33)
        plt.legend(handles=[red_patch, green_patch, blue_patch],fontsize = 13)
        plt.legend(handles=[red_patch, green_patch, blue_patch,black_patch],fontsize = 13)
        #plt.text(22,y[0],'KOI', color = 'k', fontsize = 20)
    elif num == 3:
        plt.xlim(0,28)
    elif num == 3.5:
        plt.xlim(0,28)
    elif num == 2:
        plt.xlim(0,52)
    else:
        plt.xlim(0,60)
    return

def fig2_ver2_colchange(select,alltarget,num = 0):
    LS, ACF, W, AS, tag, div, meana_group, dnur, count, spec_sini, asi,T,KOIval,Sph, med, scatter, qnum, kep, ref = read_val(select)
    y = np.linspace(0.7,(len(select)+1)*1.3,len(select))
    list1 = LS[:,0]
    print(list1)
    list2 = select.tolist()
    list1, list2 = zip(*sorted(zip(list1, list2)))
    #reorder of target according to median P
    select = list2
    #LS, ACF, W, AS, tag, div, meana_group, dnur, count, spec_sini, asi,T,KOIval,Sph, med, scatter, qnum, kep, ref = read_val(select)
    LS, ACF, W, AS, tag, div, meana_group, dnur, count, spec_sini, asi,T,KOIval,Sph, med, scatter, qnum, kep, ref = read_val(alltarget)
    yname = []
    for i in range(len(select)):
        ind = np.argwhere(alltarget==select[i])[0][0]
        if tag[ind] == 'KOI':
            name = KOIval[ind]
            yname.append(name) 
            shape = 'o'
            print(name)
        elif tag[ind] == 'nonKOI':
            name = str(alltarget[ind])
            yname.append(name)  
            shape = '^'


       
        #ind = np.argwhere(alltarget==new_sel[i])[0][0]
        plt.errorbar(LS[ind,0],y[i]+0.2,xerr=LS[ind,1:].reshape(2,1),fmt=shape,ecolor='r',color='r',elinewidth=3,capsize=7,markersize = 7)
        plt.errorbar(ACF[ind,0],y[i],xerr=ACF[ind,1:].reshape(2,1),fmt=shape,ecolor='g',color='g',elinewidth=3,capsize=7,markersize = 7)
        plt.errorbar(W[ind,0],y[i]-0.2,xerr=W[ind,1:].reshape(2,1),fmt=shape,ecolor='b',color='b',elinewidth=3,capsize=7,markersize = 7)
        plt.errorbar(AS[ind,0],y[i]-0.4,xerr=AS[ind,1:].reshape(2,1),fmt=shape,ecolor='k',color='k',elinewidth=3,capsize=7,markersize = 7)
    #plt.errorbar(asp,y-0.4,xerr=asperr,fmt='o',ecolor='k',color='k',elinewidth=4,capsize=9,markersize = 7, label = 'Astero')

    my_yticks = yname
    plt.yticks(y, my_yticks, fontsize = 20)
    plt.xticks(fontsize = 12)
    plt.xlabel('Period (days)',fontsize = 13)
    for item in y:
        diff = y[2]-y[1]
        plt.axhline(item-0.5*diff,linestyle = '--',color = 'gray')
        
    plt.xlim(xmin = 0)
    if len(select)<2:
        plt.ylim(y[0]-3,y[0]+3)
    ##legend
    green_patch = mpatches.Patch(color='g', label='ACF')
    red_patch = mpatches.Patch(color='r', label='LS')
    blue_patch = mpatches.Patch(color='b', label='Wavelet')
    black_patch = mpatches.Patch(color='k', label='Astero')
    if num == 1:
        plt.xlim(0,33)
        #plt.ylabel('KIC',fontsize = 12)
        #plt.legend(handles=[red_patch, green_patch, blue_patch],fontsize = 13)
        plt.legend(handles=[red_patch, green_patch, blue_patch,black_patch],fontsize = 13)        
       # plt.text(22,y[0],'nonKOI', color = 'k', fontsize = 20)
    elif num == 1.5:
        #plt.ylabel('KIC',fontsize = 20)
        plt.xlim(0,33)
        plt.legend(handles=[red_patch, green_patch, blue_patch],fontsize = 13)
        plt.legend(handles=[red_patch, green_patch, blue_patch,black_patch],fontsize = 13)
        #plt.text(22,y[0],'KOI', color = 'k', fontsize = 20)
    elif num == 3:
        plt.xlim(0,28)
    elif num == 3.5:
        plt.xlim(0,28)
    elif num == 2:
        plt.xlim(0,52)
    else:
        plt.xlim(0,60)
    return


def fig2_ver3(select,alltarget,num = 0):
    LS, ACF, W, AS, tag, div, meana_group, dnur, count, spec_sini, asi,T,KOIval = read_val(select)
    y = np.linspace(0.7,(len(select)+1)*1.3,len(select))
    yname = []
    for ind in range(len(select)):
        if tag[ind] == 'KOI':
            shape = 'o'
            name = KOIval[ind]
            yname.append(name)
        elif tag[ind] == 'nonKOI':
            shape = '^'
            name = str(select[ind])
            yname.append(name) 
      #  ind = np.argwhere(alltarget==new_sel[i])[0][0]

        plt.errorbar(LS[ind,0],y[ind]+0.2,xerr=LS[ind,1:].reshape(2,1),fmt=shape,ecolor='g',color='g',elinewidth=3,capsize=7,markersize = 7)
        plt.errorbar(ACF[ind,0],y[ind],xerr=ACF[ind,1:].reshape(2,1),fmt=shape,ecolor='r',color='r',elinewidth=3,capsize=7,markersize = 7)
        plt.errorbar(W[ind,0],y[ind]-0.2,xerr=W[ind,1:].reshape(2,1),fmt=shape,ecolor='b',color='b',elinewidth=3,capsize=7,markersize = 7)
        plt.errorbar(AS[ind,0],y[ind]-0.4,xerr=AS[ind,1:].reshape(2,1),fmt=shape,ecolor='k',color='k',elinewidth=3,capsize=7,markersize = 7)
    #plt.errorbar(asp,y-0.4,xerr=asperr,fmt='o',ecolor='k',color='k',elinewidth=4,capsize=9,markersize = 7, label = 'Astero')

#    yname = []
#    for j in range(len(new_sel)):
#        if new_tag[j] == 'KOI':
#            name = KOIval[j]
#            yname.append(name)  
#            print(name)
#        elif new_tag[j] == 'nonKOI':
#            name = str(new_sel[j])
#            yname.append(name)  
    
    #for i in new_sel:
    #    name = str(i)
    #    yname.append(name)
    
    my_yticks = yname
    plt.yticks(y, my_yticks, fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.xlabel('Period (Day)',fontsize = 20)

    plt.xlim(xmin = 0)
    if len(select)<2:
        plt.ylim(y[0]-3,y[0]+3)
    ##legend
    green_patch = mpatches.Patch(color='g', label='LS')
    red_patch = mpatches.Patch(color='r', label='ACF')
    blue_patch = mpatches.Patch(color='b', label='Wavelet')
    black_patch = mpatches.Patch(color='k', label='Astero')
    if num == 1:
        plt.xlim(0,25)
        plt.ylabel('KIC',fontsize = 20)
        #plt.legend(handles=[red_patch, green_patch, blue_patch],fontsize = 13)
        plt.legend(handles=[red_patch, green_patch, blue_patch,black_patch],fontsize = 13)
       # plt.text(22,y[0],'nonKOI', color = 'k', fontsize = 20)
    elif num == 1.5:
        plt.ylabel('KIC',fontsize = 20)
        plt.xlim(0,25)
        #plt.legend(handles=[red_patch, green_patch, blue_patch],fontsize = 13)
        plt.legend(handles=[red_patch, green_patch, blue_patch,black_patch],fontsize = 13)
     #   plt.text(22,y[0],'KOI', color = 'k', fontsize = 20)
    plt.xlim(0,50)
    return

def fig2_ver4(select,alltarget,num = 0):
    LS, ACF, W, AS, tag, div, meana_group, dnur, count, spec_sini, asi,T,KOIval,Sph, med, scatter, qnum, kep, ref = read_val(alltarget)
    y = np.linspace(0.7,(len(select)+1)*1.3,len(select))
    g_KOI, g_nonKOI,tag_KOI, tag_nonKOI = sort_arr(select,tag,alltarget)
    new_sel = g_KOI + g_nonKOI
    new_tag = tag_KOI + tag_nonKOI
    yname = []
    for i in range(len(new_sel)):
        if new_tag[i] == 'KOI':
            shape = 'o'
        elif new_tag[i] == 'nonKOI':
            shape = '^'
            
        ind = np.argwhere(alltarget==new_sel[i])[0][0]
        
        if tag[ind] == 'KOI':
            name = KOIval[ind]
            yname.append(name) 
        elif tag[ind] == 'nonKOI':
            name = str(alltarget[ind])
            yname.append(name)  
       
        ind = np.argwhere(alltarget==new_sel[i])[0][0]
        plt.errorbar(LS[ind,0],y[i]+0.2,xerr=LS[ind,1:].reshape(2,1),fmt=shape,ecolor='g',color='g',elinewidth=3,capsize=7,markersize = 7)
        plt.errorbar(ACF[ind,0],y[i],xerr=ACF[ind,1:].reshape(2,1),fmt=shape,ecolor='r',color='r',elinewidth=3,capsize=7,markersize = 7)
        plt.errorbar(W[ind,0],y[i]-0.2,xerr=W[ind,1:].reshape(2,1),fmt=shape,ecolor='b',color='b',elinewidth=3,capsize=7,markersize = 7)
        plt.errorbar(AS[ind,0],y[i]-0.4,xerr=AS[ind,1:].reshape(2,1),fmt=shape,ecolor='k',color='k',elinewidth=3,capsize=7,markersize = 7)
        plt.scatter(med[ind],y[i]+0.4,color = 'C1')
    #plt.errorbar(asp,y-0.4,xerr=asperr,fmt='o',ecolor='k',color='k',elinewidth=4,capsize=9,markersize = 7, label = 'Astero')
    indice = len(g_KOI)
    print(indice)
    print(new_sel)

   # for j in range(len(new_sel)):
   #     if new_tag[j] == 'KOI':
   #         name = KOIval[j]
   #         yname.append(name)  
   #         print(name)
   #     elif new_tag[j] == 'nonKOI':
   #         name = str(new_sel[j])
   #         yname.append(name)  
    
   # for i in new_sel:
   #     name = str(i)
   #     yname.append(name)
    
    my_yticks = yname
    plt.yticks(y, my_yticks, fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.xlabel('Period (Day)',fontsize = 20)
    for item in y:
        diff = y[2]-y[1]
        plt.axhline(item-0.5*diff,linestyle = '--',color = 'gray')
        
    plt.xlim(xmin = 0)
    if len(new_sel)<2:
        plt.ylim(y[0]-3,y[0]+3)
    ##legend
    green_patch = mpatches.Patch(color='g', label='LS')
    red_patch = mpatches.Patch(color='r', label='ACF')
    blue_patch = mpatches.Patch(color='b', label='Wavelet')
    black_patch = mpatches.Patch(color='k', label='Astero')
    if num == 1:
        plt.xlim(0,33)
        plt.ylabel('KIC',fontsize = 20)
        #plt.legend(handles=[red_patch, green_patch, blue_patch],fontsize = 13)
        plt.legend(handles=[red_patch, green_patch, blue_patch,black_patch],fontsize = 13)        
       # plt.text(22,y[0],'nonKOI', color = 'k', fontsize = 20)
    elif num == 1.5:
        plt.ylabel('KIC',fontsize = 20)
        plt.xlim(0,33)
        plt.legend(handles=[red_patch, green_patch, blue_patch],fontsize = 13)
        plt.legend(handles=[red_patch, green_patch, blue_patch,black_patch],fontsize = 13)
        #plt.text(22,y[0],'KOI', color = 'k', fontsize = 20)
    elif num == 3:
        plt.xlim(0,28)
    elif num == 3.5:
        plt.xlim(0,28)
    elif num == 2:
        plt.xlim(0,52)
    else:
        plt.xlim(0,60)
    return
#%% Plot consistency within error bar
def plot_compare(tar, x,y,xup=None,xlow=None,yup = None,ylow = None):
    count = 0
    incon = []
    new_incon = []
    try:
        xerr = error(xup,xlow)
    except TypeError:
        print('Neglecting X error bar')
   
    try:
        yerr = error(yup,ylow)
    except TypeError:
        print('Neglecting Y error bar')
   
    print(x)
    ############## Assign color ################
    for i in range(len(x)):
        if y[i] > x[i]:
            #print('%.2f is larger than %.2f'%(y[i],x[i]))
            if x[i] + xerr[1,i] > y[i] - yerr[0,i]:
                col = 'gray'
                shape = '.'
                #print('%.2f is larger than %.2f'%(x[i] + xerr[1,i],y[i] - yerr[0,i]))
            else:                
                col = 'r'
                shape = '^'
                count = count+1
                incon.append(tar[i])
                #print('%.2f is smaller than %.2f'%(x[i] + xerr[1,i],y[i] - yerr[0,i]))
        elif x[i] > y[i]:
            #print('%.2f is smaller than %.2f'%(y[i],x[i]))
            if x[i] - xerr[0,i] < y[i] + yerr[1,i]:
                col = 'gray'
                shape = '.'
               # print('%.2f is larger than %.2f'%(x[i] - xerr[0,i], y[i] + yerr[1,i]))
            else:
                col = 'r'
                shape = '^'
                count = count+1
                incon.append(tar[i])
                #print('%.2f is smaller than %.2f'%(x[i] - xerr[0,i], y[i] + yerr[1,i]))
       
        plt.errorbar(x[i],y[i],xerr = xerr[:,i].reshape(2,1),
            yerr = yerr[:,i].reshape(2,1),fmt= shape,ecolor=col, 
            color=col,elinewidth=2, capsize=2,markersize = 14)
    return incon

#########plot planet error bar#####################
def plot_planet(target,yval):
    gpath = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/KOIplanet.csv' 
    df = pd.read_csv(gpath)
    alltar= np.array(df['kepid'])
    p = np.array(df['period'])
    up = np.array(df['up'])
    low = np.array(df['low'])
    koi, non = group_simple()
    for item in koi:
        inds = [x[0] for x in np.argwhere(alltar == target)]
        for idx in inds:
            plt.errorbar(p[idx], yval-0.35, xerr=np.array([low[idx],up[idx]]).reshape(2,1), capsize=3, fmt='o', markersize=3, ecolor='cyan',markeredgecolor = "cyan", markerfacecolor = 'none')
###################################################

def check_con(X, Y):
    if X[0] > Y[0]:
        if X[0]-X[1]<Y[0]+Y[2]:
            val = 1
        else:
            val = -1
    else:
        if X[0]+X[2]>Y[0]-Y[1]:
            val = 1
        else:
            val = -1 
    return val


def read_vsin(name,author):
    if name =='KOI':
        path = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/specKOI.csv'
        df = pd.read_csv(path)
        alltarget = np.array(df['target'])
        vsin = np.array(df['vsin'])
        vsinerr = np.array(df['vsinerr'])
        R = np.array(df['specR'])
        Rup = np.array(df['specRerr'])
        Rlow = np.array(df['specRerr'])
        comment = 0
    elif name == 'nonKOI':
        path = r'F:/Home/UTokyo/research/discussion/paper/organize_new/organize/value/specnonKOI.csv'
        df = pd.read_csv(path)
        alltarget = np.array(df['target'])
        R = np.array(df['R'])
        Rup = np.array(df['Rup'])
        Rlow = np.array(df['Rlow'])
        if author == 'Brunt':
            vsin = np.array(df['Brunt_vsin'])
            vsinerr = np.array(df['Brunt_vsinerr'])
            comment = np.array(df['Bcom'])
        elif author == 'M':
            vsin = np.array(df['vsin'])
            vsinerr = np.array(df['vsinerr'])
            comment = np.array(df['Comment'])
    ############form error bar###############
    vsini = []
    R_f = []
    for i in range(len(alltarget)):
        vsini =  vsini + [vsin[i],vsinerr[i],vsinerr[i]]
        R_f =  R_f + [R[i],Rlow[i],Rup[i]]
    vsini = np.array(vsini).reshape(len(alltarget),3)  
    R_f = np.array(R_f).reshape(len(alltarget),3)  
    return vsini, R_f, alltarget,comment
##################################################
def spectro_vsin(rawselect,author,name):
    #KOI    
    vsin1, R1, reftar1,comment1 = read_vsin('KOI','no')
    #nonKOI-B
    vsin2, R2, reftar2,comment2 = read_vsin('nonKOI','Brunt')
    #nonKOI-M
    vsin3, R3, reftar3,comment3 = read_vsin('nonKOI','M')
    #read period
    LS, ACF, W, AS, tag, div, meana, dnur, count, spec_sini, asi,T,KOIval,Sph, med, scatter, qnum, kep, ref = read_val(rawselect)
    select = []
    for i in range(len(rawselect)):
        if tag[i]==name:
            select.append(rawselect[i])
    #################################
    LS, ACF, W, AS, tag, div, meana, dnur, count, spec_sini, asi,T,KOIval,Sph, med, scatter, qnum, kep, ref = read_val(select)
    vsin = []
    R = []
    comment = []
    pho_p = med
    for i in range(len(select)):
        if tag[i]=='KOI':
            idx = np.argwhere(reftar1==select[i])[0][0]
            vsin.append(vsin1[idx])
            R.append(R1[idx])
            comment.append(comment1)
        elif tag[i]== 'nonKOI':
            if author == 'Brunt':
                idx = np.argwhere(reftar2==select[i])[0][0]
                vsin.append(vsin2[idx])
                R.append(R2[idx])
                comment.append(comment2[idx])
            elif author == 'M':
                idx = np.argwhere(reftar3==select[i])[0][0]
                vsin.append(vsin3[idx])
                R.append(R3[idx])
                comment.append(comment3[idx])
    vsin = np.array(vsin)
    R = np.array(R)
    ############################################
    fac = 696340 # solar radii
    unit = 3600*24
    mid = vsin[:,0] * pho_p[:,0]*unit/(np.pi*2*R[:,0]*fac)
    up = mid*np.sqrt((vsin[:,2]/vsin[:,0])**2+(R[:,2]/R[:,0])**2+ (pho_p[:,2]/pho_p[:,0])**2)
    low =mid*np.sqrt((vsin[:,1]/vsin[:,0])**2+(R[:,1]/R[:,0])**2+ (pho_p[:,1]/pho_p[:,0])**2)
    sini = np.array([mid,low,up])
    sini = sini.transpose()
    #num = np.count_nonzero(~np.isnan(mid))
    #nan_name = select[np.isnan(mid)]
    #print('%d out of %d targets are nonzero and non nan values for %s'%(num,len(select),name))
    #print(nan_name)
    #print(' has no value')
    return sini,select
##8 nan###
#####################################################