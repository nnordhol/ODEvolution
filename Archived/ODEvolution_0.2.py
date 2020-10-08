#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 09:41:22 2020

@author: nnordholt
"""
from scipy.integrate import odeint, solve_ivp, ode
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numba import jit



# Define bacterial growth with Monod kinetics. n = number of bacteria, r = resource concentrtion:
# This can be extended to include growth inhibition, kill kinetics
def growth(n, r, mumax, km):
    return mumax*r/(r+km)*n

# Define ODE function.
    
def pop_ivp(t, ys, ode_args):
    # This if-statement is necessary because otherwise the resource runs into negative values and returns crap.
    #if ys[-1] <10**-10:
    #    return np.zeros(len(ys))
    gen_vec = [growth(ys[i],ys[-1],ar[0],ar[1]) for i,ar in enumerate(ode_args)]
    r_vec = -np.sum([gen_vec[i]*ar[2] for i,ar in enumerate(ode_args)])
    #dNdt = mumax*r/(r+km)*N
    #dN1dt = mumax1*r/(r+km)*N1
    #drdt = -e*(dNdt+dN1dt)
    dydt = np.append(np.transpose(gen_vec),r_vec)
    return dydt

def pop_ivp_mut(t, ys, ode_args):
    # This if-statement is necessary because otherwise the resource runs into negative values and returns crap.
    #if ys[-1] <10**-10:
    #    return np.zeros(len(ys))
    gen_vec = [growth(ys[i],ys[-1],ar[0],ar[1]) for i,ar in enumerate(ode_args)]
    r_vec = -np.sum([gen_vec[i]*ar[2] for i,ar in enumerate(ode_args)])
    #dNdt = mumax*r/(r+km)*N
    #dN1dt = mumax1*r/(r+km)*N1
    #drdt = -e*(dNdt+dN1dt)
    dydt = np.append(np.transpose(gen_vec),r_vec)
    return dydt


# DFE from Robert et al 2018 for fitness cost of mutations: stats.beta(a = 0.0074, b = 2.4)
dfe_roberts = stats.beta(a = 0.0074, b = 2.4)


# Change Genotype.n to dict with cycle as key?
class Genotype:
    """ Genotype class for competition and evolution experiments """
    
    def __init__(self, mumax = 1.11, e = 5*10**-9, km = 0.25, n0 = 10**4, gid = 0, mut_rate = 10**-10, survival = 0.0004, mr=10**-3):
        
        self.gid = gid
        self.mumax = mumax
        self.e = e
        self.km = km
        self.n = np.array([n0])
        self.mutations = dict()
        self.survival = survival
        self.survivors = np.zeros((0,1))
        self.ts = np.zeros((0,1))
        self.mr = 10**-3
        self.mutants=np.zeros((0,1))
        #self.y0 = np.array([])
        
    def get_pars(self):
        return [self.n, self.mumax, self.km, self.e]
        
    
    def mutate(self):
        return Genotype()
    
        
        
        
        
# Include volume and sample volume, and Poisson sampling!
# Volume and Poisson sampling are important for emergence and propagation of mutants (absolute numbers are important here!)
# [stats.poisson(g.n[-1]*(sample_vol/vol)).rvs() for g in self.genotypes]

# Include calculation of selection coefficient matrix
        
class Experiment:
    """ Experiment class for competition and evolution experiments """
    
    def __init__(self, genotypes, volume = 10, sample_volume = 0.1, predilution = 4,
                 gluc_start = 20, t_span = [0,24], killing = True):
        
        self.genotypes = genotypes
        self.cycles = 0
        self.volume = volume
        self.sample_volume = sample_volume
        self.dilution = volume/sample_volume
        self.predilution = predilution
        self.gluc_start = gluc_start
        self.gluc = np.array([gluc_start])
        self.t_span = t_span
        #self.ode_args = [[g.mumax, g.km, g.e] for g in self.genotypes]
        #self.y0 = np.array([g.n[-1] for g in self.genotypes]+[gluc_start])
        self.ts = np.zeros((0,1))
        self.sols = np.zeros((0,len(self.genotypes)+1))
        self.killing = killing
        self.survivors = np.zeros((0,len(self.genotypes)))
        self.fractions = np.zeros((0,len(self.genotypes)))
        self.total = np.zeros((0,len(self.genotypes)))
    
    def pop_ivp_mut(self, t, ys, ode_args):
        # This if-statement is necessary because otherwise the resource runs into negative values and returns crap.
        #if ys[-1] <10**-10:
            #    return np.zeros(len(ys))
            gen_vec = np.array([growth(ys[i],ys[-1],ar[0],ar[1]) for i,ar in enumerate(ode_args)])
            r_vec = -np.sum([gen_vec[i]*ar[2] for i,ar in enumerate(ode_args)])
            #dNdt = mumax*r/(r+km)*N
            #dN1dt = mumax1*r/(r+km)*N1
            #drdt = -e*(dNdt+dN1dt)
            mutants = gen_vec*self.from_genos('mr')
            #print(t, np.where(mutants>=1, mutants, 0))
            #self.append_genos('mutants', np.where(mutants>=1, mutants, 0))
            #self.append_genos('ts', np.array([t]*len(self.genotypes)))
            dydt = np.append(np.transpose(gen_vec),r_vec)
            return dydt
    
    def append_genos(self, geno_att, toappend):
        
        if geno_att in Genotype().__dict__:
            for i,g in enumerate(self.genotypes):
                g.__dict__[geno_att] = np.append(g.__dict__[geno_att], toappend[i])
                #map(lambda x: g.__dict__[geno_att] = np.append(g.__dict__[geno_att], toappend), self.genotypes)
    
    def from_genos(self, geno_att):
       if geno_att in Genotype().__dict__:
           return np.array([g.__dict__[geno_att] for g in self.genotypes]).T
       else:
           print(f'{geno_att} not an attribute of class Genotype().')
    
    def dilute(self, last_n=None):
        if last_n is None:
            last_n = np.array([g.n[-1] for g in self.genotypes])
        
        lam = (last_n*self.sample_volume)
        sample = stats.poisson(lam).rvs()
        diluted = np.where(sample>=1, sample/self.volume, 0)
        #diluted = np.array([g.n[-1]/self.dilution if g.n[-1]>=self.dilution else 0 for g in self.genotypes]+[self.gluc_start])
        #diluted = np.array([g.n[-1]/self.dilution if g.n[-1]>=self.dilution else 0 for g in self.genotypes]+[self.gluc_start])
        self.append_genos('n', diluted)
        self.gluc = np.append(self.gluc, self.gluc_start)
        
    def kill(self):
        
        survivors = np.array([(g.n[-1]/self.predilution)*g.survival for g in self.genotypes])
        self.append_genos('survivors', survivors)
        #[(g.survivors = survivors) for g in self.genotypes]
    
        # if not len(self.survivors):
        #     self.survivors = np.array([survivors])
        # else:
        #     self.survivors = np.append(self.survivors, [survivors], axis = 0)
        self.dilute(survivors)
        #self.y0 = np.append(np.where(diluted>=1, diluted, 0), [self.gluc_start])
        #self.y0 = np.array([(g.n[-1]/self.predilution)*g.survival/self.dilution if (g.n[-1]/self.predilution)*g.survival/self.dilution>=1 else 0 for g in self.genotypes]+[self.gluc_start])
    
    
    def create_ode_vec(self):
        return np.transpose([g.n[-1] for g in self.genotypes])
    
    @jit
    def conduct_once(self):
        
        
        
        #def doubling(t,y,*args): 
        #    return (y-y[:,-1])
        #doubling_event = lambda t,y: doubling(t, y)
        #doubling.terminal = False
        #doubling.direction = -1
        
        #def no_gluc(t,y, *args): return y[-1]
        #no_gluc.terminal = False
        #no_gluc.direction = -1
        
        y0 = np.append([g.n[-1] for g in self.genotypes], self.gluc[-1])
        
        ode_args = [[g.mumax, g.km, g.e] for g in self.genotypes]

        self.sol = solve_ivp(self.pop_ivp_mut, self.t_span, y0=y0, args=([ode_args]),
                                 dense_output=True, 
                                 # events =no_gluc,
                                 #t_eval = np.linspace(0,24,101),
                                 #min_step = 0.001,
                                 #max_step = 0.001,
                                 method = 'LSODA',
                                 atol=10**-9,
                                 rtol=10**-8)
        
        # This needs adaptation once new Genotypes can emerge: dimensions of sols will change! So put solutions on Genotype objects, not here? Or append solution objects in array, not solutions
        self.append_genos('n',  self.sol.y[:,1:])
        #self.append_genos('ts',  [self.sol.t]*len(self.genotypes))
        
        #self.sols = np.append(self.sols, self.sol.y.T, axis = 0)
        self.gluc = np.append(self.gluc, self.sol.y[-1])
        self.append_genos('ts',  [(self.ts[-1] if self.ts.any() else 0)+self.sol.t]*len(self.genotypes))
        self.ts = np.append(self.ts, (self.ts[-1] if self.ts.any() else 0)+self.sol.t)
        self.days = self.ts/self.t_span[-1]
        #self.calc_fractions()
        #self.total = np.append(self.total, np.sum(self.sol.y[:-1],axis=0))#np.array([np.sum(sol) for sol in self.sol.y[:-1].T]))
        self.cycles +=1
        #for i,g in enumerate(self.genotypes):
        #    g.n = np.concatenate((g.n,self.sol.y[i]))
            
        
    def conduct(self, cycles = 15):
        
        for c in range(cycles):
            self.conduct_once()
            self.kill() if self.killing else self.dilute()
        #self.total_survivors = np.sum(self.survivors, axis =1)
                
        
    def calc_fractions(self):
        self.fractions =  [sol/np.sum(sol) for sol in self.sols[:,:-1]]
        
    
    def add_genotype(self, genotype):
        self.genotypes.append(genotype)
        self.y0 = np.array([g.n[-1] for g in self.genotypes]+[self.gluc_start])
    
    def calc_t(self):
        taus = np.log(2)/self.from_genos('mumax')
