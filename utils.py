# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 13:47:50 2020

@author: nnordhol
"""

import numpy as np
from ODEvolution import *
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

def ODE_scanner(murange=(0,1.4), survfracs=(-7,0), steps = 100, dilution = 100, mut_start=0.1, cycles=15):
    
    simmus=np.linspace(murange[0],murange[1],steps)
    simfracs=np.logspace(survfracs[0],survfracs[-1],steps, base=10)
    sample_volume = 10/dilution
    
    wts=np.zeros((len(simmus),len(simfracs)))
    evs=np.zeros((len(simmus),len(simfracs)))
    wt_times = np.zeros((len(simmus),len(simfracs)))
    ev_times = np.zeros((len(simmus),len(simfracs)))
    rs=np.zeros((len(simmus),len(simfracs)))
    ss=np.zeros((len(simmus),len(simfracs)))
    scs = rs=np.zeros((len(simmus),len(simfracs)))
    for i,simmu in enumerate(simmus):
        for j,simfrac in enumerate(simfracs):
            wt = Genotype(mumax = 1.2, survival = 6.7*10**-5, n0=10**4, gid='WT')
            mut = Genotype(mumax = simmu, survival = simfrac, n0 = mut_start, gid='Mutant')
            
            exp = Experiment([wt, mut], poisson=False, sample_volume=sample_volume)
            exp.conduct_steps(cycles=cycles, stop_after_first_extinct=True)
            wt.survivors[-1]
            mut.survivors[-1]
            #ratio = mut.n[-2]/(wt.n[-2]+mut.n[-2])
            logdiffs = np.diff(np.log(mut.survivors))-np.diff(np.log(wt.survivors))
            logdiffs = logdiffs[(logdiffs!=np.inf)&(logdiffs!=np.nan)]
            sc=np.nanmean(logdiffs)
            wts[i,j]=wt.n[-2]
            evs[i,j]=mut.n[-2]
            wt_times[i,j] = wt.ts[-1]
            ev_times[i,j] = mut.ts[-1]
            #rs[i,j]=ratio
            scs[i,j] = sc
    rs=evs/(evs+wts)
    savestr=f'dil-{dilution}_mut-{mut_start}'
    np.save(f'rs_{savestr}.npy', rs)
    np.save(f'wts_{savestr}.npy', wts)
    np.save(f'evs_{savestr}.npy', evs)
    np.save(f'scs_{savestr}.npy', scs)
    np.save(f'simmus_{savestr}.npy', simmus)
    np.save(f'simfracs_{savestr}.npy', simfracs)
    np.save(f'ev_times_{savestr}.npy', ev_times)
    np.save(f'wt_times_{savestr}.npy', wt_times)
    
    f=plt.figure()
    cs=plt.contourf(simfracs,simmus,np.where(np.nan_to_num(rs)>0.5, ev_times, 0)/24,5)
    plt.colorbar()
    plt.contour(cs, colors='k')
    plt.title('Cycles until mutant takes over population')
    plt.ylabel('Growth rate [h$^{-1}$]')
    plt.xlabel('Surviving fraction')
    plt.xlim([10**survfracs[0], 10**survfracs[1]])
    plt.ylim([murange[0],murange[1]])
    f.savefig(f'Cycles_mutant_{savestr}.svg')
    
    f=plt.figure()
    cs=plt.contourf(simfracs,simmus,np.where(np.nan_to_num(rs)>0.5, ev_times, 0)/24,4)
    plt.colorbar()
    plt.contour(cs, colors='k')
    plt.title('Cycles until mutant takes over population')
    plt.ylabel('Growth rate [h$^{-1}$]')
    plt.xlabel('Surviving fraction')
    plt.xlim([10**survfracs[0], 10**survfracs[1]])
    plt.ylim([murange[0],murange[1]])
    plt.xscale('log')
    f.savefig(f'Cycles_mutant_log_{savestr}.svg')
    
    f=plt.figure()
    cs=plt.contourf(simfracs,simmus, rs, 5)
    plt.title('Fraction of mutant in the population after 15 cycles')
    plt.colorbar()
    plt.contour(cs, colors='k')
    plt.ylabel('Growth rate [1/h]')
    plt.xlabel('Surviving fraction')
    plt.xlim([10**survfracs[0], 10**survfracs[1]])
    plt.ylim([murange[0],murange[1]])
    plt.xscale('log')
    f.savefig(f'Fraction_mutant_{savestr}.svg')
    
    f=plt.figure()
    cs=plt.contourf(simfracs,simmus,rs,10, locator=ticker.LogLocator())
    plt.title('Fraction of mutant in the population after 15 cycles')
    plt.colorbar()
    plt.contour(cs, colors='k')
    plt.ylabel('Growth rate [1/h]')
    plt.xlabel('Survival fraction')
    plt.xlim([10**survfracs[0], 10**survfracs[1]])
    plt.ylim([murange[0],murange[1]])
    plt.xscale('log')
    f.savefig(f'Fraction_mutant_log_ticker{savestr}.svg')

if __name__=="__main__":
    ds = [1,2,5,10,100,1000,10**4, 10**6]
    mut_starts= [0.01, 0.1, 1, 100, 10**4, 10**6]
    for ms in mut_starts:
        for d in ds:
            ODE_scanner(dilution=d, mut_start=ms)