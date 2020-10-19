#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 09:41:22 2020

@author: nnordholt
"""
from scipy.integrate import solve_ivp
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
#from copy import deepcopy
from random import random

from random import choice

#Sfrom numba import jit


def no_gluc(t,y, *args): return y[-1]
no_gluc.terminal = True
no_gluc.direction = -1

##### TODO #######

# Return array of ts and ns for each cycle
## This makes it easier to include lag, and calculate selection coefficients
# Possibility to have concentrations change over time
## For antibiotic (dosing), but also for BAC through exhaustion model?
# Make possible to let defined genotype emerge at certain timepoint. Then fit model to survival data. (Alternatively: wrapper with add_genotype after cycle)
# Calculate total survivors from genotypes
## This should be done by returning a dataframe. Makes it easier for plotting and other operations
### Problem : duplicate time entries, so time cant be used as index. Works for survivors, though
# Option to easily set fitness effects and target size (e.g. effects of the mutants we measure, plus target size from number of mutations)
## Then we can use it in fitting
# Make it possible to construct lineages (which genotype comes from which?)
# Calculate selection coefficients (matrix?)
# Incorporate trade-off between growth rate and survival: emergence of tolerant strain, which then loses survival to growth rate?
## Trade off in both directions
# Add ts and ns as array structures for each cycle to genotypes --> calculate number of mutants in each step
## Alternatively, only append first and last value of each step?
# Include target size, specific mutation in simulations
# Use distribution of growth rates for each genotype to simulate phenotypes?


# Define bacterial growth with Monod kinetics. n = number of bacteria, r = resource concentration:
# This can be extended to include growth inhibition, kill kinetics 
def growth(n, r, mumax, km, a = 0, ic50 = 1, kappa = 1):
    return mumax*r/(r+km)*n*(1-a/(a+ic50))


# DFE from Robert et al 2018 for fitness cost of mutations: stats.beta(a = 0.0074, b = 2.4)
dfe_roberts = stats.beta(a = 0.0074, b = 2.4)


# Change Genotype.n to dict with cycle as key?
class Genotype:
    """ Genotype class for competition and evolution experiments """
    
    def __init__(self, mumax = 1.11, e = 5*10**-9, km = 0.25, n0 = 10**4, gid = 0, survival = 0.0004, mr=2*10**-10, ic50 = 1, kappa = 1):
        
        self.gid = gid
        self.mumax = mumax
        self.e = e
        self.km = km
        self.n = np.array([n0])
        self.mutations = np.zeros((0,1))
        self.survival = survival
        self.survivors = np.zeros((0,1))
        self.ts = np.zeros((0,1))
        self.mr = mr
        self.mutation_times = np.zeros((0,1))
        self.extinct = False
        self.emerged = 0
        self.stepdy = np.zeros((0,1))
        self.ic50 = ic50
        self.kappa = 1
        #self.y0 = np.array([])
        
    def get_pars(self):
        return [self.n, self.mumax, self.km, self.e]
        
    
    def mutate(self, volume = 10):
        
        fitness_cutoff = 0
        target_size = 9000
        dndt = (self.stepdy[-1])*volume
        expected = dndt*self.mr*target_size
        mutants = stats.poisson(expected).rvs()
        if mutants == 0:
            return
        mu_effects = dfe_roberts.rvs(mutants)
        mu_effects = mu_effects[mu_effects>=fitness_cutoff]
        mwes= []
        #survival_dist = stats.loguniform(10**-6, 2*10**-1).rvs(len(mu_effects))
        pos_neg=1 if random() < 0.5 else -1
        trade_off = stats.norm(0.461, 0.04).rvs(len(mu_effects))## Observed trade-off in our experiments, plus some noise
        for i,me in enumerate(mu_effects):
            #mu_dist = stats.uniform(0.1,1).rvs()
            #survival = survival_dist[i]
            me=me*pos_neg
            survival = trade_off[i]*me
            new_mu = self.mumax-me
            mwe = Genotype(
                n0 = 0.1, 
                mumax = new_mu,
                survival = survival
                )
            mwe.mutation_times = np.append(mwe.mutation_times, self.ts[-1])
            #mwe.gid = mwe.gid+'_1'
            mwe.mutations = np.append(self.mutations, [mwe.mumax, mwe.survival])
            mwes.append(mwe)
    
        return mwes
    
        
        
        
        
# Include volume and sample volume, and Poisson sampling!
# Volume and Poisson sampling are important for emergence and propagation of mutants (absolute numbers are important here!)
# [stats.poisson(g.n[-1]*(sample_vol/vol)).rvs() for g in self.genotypes]

# Include calculation of selection coefficient matrix
        
class Experiment:
    """ Experiment class for competition and evolution experiments """
    
    def __init__(self, genotypes, volume = 10, sample_volume = 0.1, predilution = 4,
                 gluc_start = 20, t_span = [0,24], killing = True, poisson = True, antibiotic = 0):
        
        self.genotypes = genotypes
        self.cycles = 0
        self.volume = volume
        self.sample_volume = sample_volume
        self.dilution = volume/sample_volume
        self.predilution = predilution
        self.gluc_start = gluc_start
        self.gluc = np.array([gluc_start])
        self.antibiotic = antibiotic
        self.t_span = t_span
        #self.ode_args = [[g.mumax, g.km, g.e] for g in self.genotypes]
        #self.y0 = np.array([g.n[-1] for g in self.genotypes]+[gluc_start])
        self.ts = np.zeros((0,1))
        self.sols = np.zeros((0,len(self.genotypes)+1))
        self.killing = killing
        self.survivors = np.zeros((0,len(self.genotypes)))
        self.fractions = np.zeros((0,len(self.genotypes)))
        self.total = np.zeros((0,len(self.genotypes)))
        self.poisson = poisson
    
    def pop_ivp_mut(self, t, ys, ode_args):
        gen_vec = np.array([growth(ys[i],ys[-1],ar[0],ar[1]) for i,ar in enumerate(ode_args)])
        if (gen_vec<1/self.volume).all():
            return np.zeros(len(ys))
        r_vec = -np.sum([gen_vec[i]*ar[2] for i,ar in enumerate(ode_args)])
            #dNdt = mumax*r/(r+km)*N
            #dN1dt = mumax1*r/(r+km)*N1
            #drdt = -e*(dNdt+dN1dt)
        #mutants = gen_vec*self.from_genos('mr')
            #print(t, np.where(mutants>=1, mutants, 0))
            #self.append_genos('mutants', np.where(mutants>=1, mutants, 0))
            #self.append_genos('ts', np.array([t]*len(self.genotypes)))
        dydt = np.append(np.transpose(gen_vec),r_vec)
        
        return dydt
    
    def append_genos(self, geno_att, toappend):
        toappend=list(toappend)
        if geno_att in Genotype().__dict__:
            #if len(self.genotypes)>1:
                for g in self.genotypes:
                    if not g.extinct:
                        g.__dict__[geno_att] = np.append(g.__dict__[geno_att], toappend[0])
                        toappend.pop(0)
             #       else:
                        pass
            #else:
            #    g = self.genotypes[0]
            #    g.__dict__[geno_att] = np.append(g.__dict__[geno_att], toappend[0])
                #map(lambda x: g.__dict__[geno_att] = np.append(g.__dict__[geno_att], toappend), self.genotypes)
    
    
# =============================================================================
#     def append_genos(self, geno_att, toappend):
#         
#         if geno_att in Genotype().__dict__:
#             if len(self.genotypes)>1:
#                 for i,g in enumerate(self.genotypes):
#                     g.__dict__[geno_att] = np.append(g.__dict__[geno_att], toappend[i])
#             else:
#                 g = self.genotypes[0]
#                 g.__dict__[geno_att] = np.append(g.__dict__[geno_att], toappend)
#                 #map(lambda x: g.__dict__[geno_att] = np.append(g.__dict__[geno_att], toappend), self.genotypes)
# =============================================================================
    
    def set_all_geno_att(self, geno_att, vals):
        if geno_att in Genotype().__dict__:
            if len(self.genotypes)>1:
                for i,g in enumerate([self.genotypes]):
                    g.__dict__[geno_att] = vals[i] if len(vals)>1 else vals
            else:
                g = self.genotypes[0]
                g.__dict__[geno_att] = vals[0]
    
    def from_genos(self, geno_att):
       if geno_att in Genotype().__dict__:
           return np.array([g.__dict__[geno_att] for g in self.genotypes]).T
       else:
           print(f'{geno_att} not an attribute of class Genotype().')

    
    def dilute(self, last_n=None):
        
        if last_n is None:
            last_n = np.array([g.n[-1] for g in self.get_not_extinct()])
        self.append_genos('survivors', last_n)
        
        lam = (last_n*self.sample_volume)
        if self.poisson:
            sample = stats.poisson(lam).rvs()
            if type(sample)==int:
                sample=np.array([sample])
        else:
            sample = lam
        diluted = np.where(sample>=1, sample/self.volume, 0)
        #diluted = np.array([g.n[-1]/self.dilution if g.n[-1]>=self.dilution else 0 for g in self.genotypes]+[self.gluc_start])
        #diluted = np.array([g.n[-1]/self.dilution if g.n[-1]>=self.dilution else 0 for g in self.genotypes]+[self.gluc_start])
        #extinct = (diluted==0)
        #extinct_index = [i for i, val in enumerate(extinct) if val] 
        #for i in extinct_index:
        #    self.genotypes[i] = True
        #not_extinct = [g for g in self.genotypes if not g.extinct]
        #self.append_genos('n', diluted[~extinct], not_extinct)
        self.append_genos('n', diluted)
        self.append_genos('ts', [self.ts[-1]]*len(self.genotypes))
        self.gluc = np.append(self.gluc, self.gluc_start)
        
        for i,g in enumerate(self.get_not_extinct()):
            g.extinct = (diluted == 0)[i]
        
        #self.set_all_geno_att('extinct', (diluted == 0))
        
    def kill(self):
        survivors = np.array([(g.n[-1]/self.predilution)*g.survival for g in self.get_not_extinct()])
        survivors = np.where(survivors*self.volume>=1, survivors, 0)
        
        #[(g.survivors = survivors) for g in self.genotypes]
    
        # if not len(self.survivors):
        #     self.survivors = np.array([survivors])
        # else:
        #     self.survivors = np.append(self.survivors, [survivors], axis = 0)
        self.dilute(survivors)
        #self.y0 = np.append(np.where(diluted>=1, diluted, 0), [self.gluc_start])
        #self.y0 = np.array([(g.n[-1]/self.predilution)*g.survival/self.dilution if (g.n[-1]/self.predilution)*g.survival/self.dilution>=1 else 0 for g in self.genotypes]+[self.gluc_start])
    
    def get_not_extinct(self):
        return [g for g in self.genotypes if not g.extinct]
    
    def create_ode_vec(self):
        return np.transpose([g.n[-1] for g in self.genotypes])
    
    #@jit(cache=True) # jit kills the kernel with this function.
    def conduct_once(self, t_span = None) -> None:
        
        if t_span is None:
            t_span = self.t_span
        
        #def doubling(t,y,*args): 
        #    return (y-y[:,-1])
        #doubling_event = lambda t,y: doubling(t, y)
        #doubling.terminal = False
        #doubling.direction = -1
        
        def no_gluc(t,y, *args): return y[-1]
        no_gluc.terminal = True
        #no_gluc.direction = -1
        
        #not_extinct = np.array([g.n[-1] for g in self.genotypes if not g.extinct])

        # Somehow exclude genotypes that have gone extinct from this whole process! (maybe use .extinct flag?)
        # Stop simulation when gluc low
        last_n,params = map(np.array,zip(*[(g.n[-1], [g.mumax, g.km, g.e]) for g in self.get_not_extinct()]))
        #print(last_n)
        #print(extinct)
        #not_extinct = last_n[~extinct]
        
        y0 = np.append(last_n, self.gluc[-1])
        
        ode_args = params

        self.sol = solve_ivp(self.pop_ivp_mut, t_span, y0=y0, args=([ode_args]),
                                 #dense_output=True, 
                                 #events =no_gluc,
                                 #t_eval = np.linspace(0,24,101),
                                 #min_step = 0.001,
                                 #max_step = 0.001,
                                 #method = 'LSODA',
                                 atol=10**-9,
                                 rtol=10**-8
                                 )
        
        self.append_genos('n',  self.sol.y[:-1,:])
        self.append_genos('ts',  [(self.ts[-1] if self.ts.any() else 0)+self.sol.t]*(len(y0)-1))
        self.append_genos('stepdy', self.sol.y[:-1,-1]-self.sol.y[:-1,0])
        #self.append_genos('ts',  [self.sol.t]*len(self.genotypes))
        
        #self.sols = np.append(self.sols, self.sol.y.T, axis = 0)
        self.gluc = np.append(self.gluc, self.sol.y[-1])
        self.ts = np.append(self.ts, (self.ts[-1] if self.ts.any() else 0)+self.sol.t)
        self.days = self.ts/self.t_span[-1]
        #return self.sol
        #self.calc_fractions()
        #self.total = np.append(self.total, np.sum(self.sol.y[:-1],axis=0))#np.array([np.sum(sol) for sol in self.sol.y[:-1].T]))
        #for i,g in enumerate(self.genotypes):
        #    g.n = np.concatenate((g.n,self.sol.y[i]))
            
        
    def conduct(self, cycles = 15):
        
        for c in range(cycles):
            self.conduct_once()
            self.cycles +=1
            self.kill() if self.killing else self.dilute()
        #self.total_survivors = np.sum(self.survivors, axis =1)
    
    def conduct_steps(self, cycles = 15, steps = 25, mutate = False, stop_after_first_extinct = False):
        
        if steps is None:
            t_spans = [self.t_span]
        else:
            l,step = np.linspace(self.t_span[0], self.t_span[1], steps, retstep=True, dtype='float64')
            t_spans = [[0,step]]*(steps-1)
            
        for c in range(cycles):
            j = 0
            #print(f'Cycle: {self.cycles}')
            for t_span in t_spans:
                while (any([g.stepdy[-1]>0 if g.stepdy.size>0 else True for g in self.get_not_extinct()]) or j == 0) and (self.ts.size==0 or not np.isclose(self.ts[-1],self.t_span[1]*(self.cycles+1))):
                    self.conduct_once(t_span=t_span)
                    if mutate:
                        mutants = [g.mutate(volume = self.volume) for g in self.genotypes if g.stepdy[-1]>=(1/self.volume)]
                        mutants = list(filter(None.__ne__, mutants))
                        if len(mutants)>=1:
                            #mutants = [choice(mutants[0])] # Adds only one mutant
                            self.add_genotypes(np.concatenate(mutants))
                    j+=1
                else:
                    self.append_genos('n',  [g.n[-1] for g in self.get_not_extinct()])
                    self.append_genos('ts',  [(self.cycles*self.t_span[-1])+self.t_span[-1]]*len(self.get_not_extinct()))
                    self.append_genos('stepdy', [0]*len(self.get_not_extinct()))
                    self.ts = np.append(self.ts, [(self.cycles*self.t_span[-1])+self.t_span[-1]])
                    break
                    
            self.cycles +=1
            self.kill() if self.killing else self.dilute()
            if stop_after_first_extinct:
                if any([g.extinct for g in self.genotypes]):
                    break
        #self.total_survivors = np.sum(self.survivors, axis =1)
                
        
    def calc_fractions(self):
        self.fractions =  [sol/np.sum(sol) for sol in self.sols[:,:-1]]
        
    
    def add_genotypes(self, genotypes):
        for genotype in genotypes:
            genotype.emerged = self.ts[-1]
            self.genotypes.append(genotype)
        #self.y0 = np.array([g.n[-1] for g in self.genotypes]+[self.gluc_start])
    
    def calc_t(self):
        taus = np.log(2)/self.from_genos('mumax')


# Useful stuff:
# Plot survivors vs. cycles for genotypes that emerge during experiment:   
# plt.figure();[plt.plot(np.linspace(g.ts[0]/24, 15, len(g.survivors)),g.survivors) for g in exp.genotypes];plt.yscale('log')