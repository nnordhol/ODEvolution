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



# Define bacterial growth with Monod kinetics. n = number of bacteria, r = resource concentrtion:
def growth(n, r, mumax, km):
    return mumax*r/(r+km)*n

# Define ODE function. Takes ys and arguments. Could be done with np.vectorize() instead?
# There seems to be an issue with solve_ivp, because it can not solve the system (unless)
    
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


def pop(ys, t, ode_args):
    if ys[-1] <=0:
        return np.zeros(len(ys))
    gen_vec = [growth(ys[i],ys[-1],ar[0],ar[1]) for i,ar in enumerate(ode_args)]
    r_vec = -np.sum([ys[i]*ar[2] for i,ar in enumerate(ode_args)])
    #if abs(r_vec)>ys[-1]:
    #    return np.zeros(len(ys))
    #gen_vec = [growth(g.n[-1],r, g.mumax, g.km) for g in genos]
    #r_vec = -np.sum([g.n[-1]*g.e for g in genos]) # make individual e for each genotype
    #dNdt = mumax*r/(r+km)*N
    #dN1dt = mumax1*r/(r+km)*N1
    #drdt = -e*(dNdt+dN1dt)
    dydt = np.append(np.transpose(gen_vec),r_vec)
    return dydt


# DFE from Robert et al 2018 for fitness cost of mutations: stats.beta(a = 0.0074, b = 2.4)
# dfe_roberts = stats.beta(a = 0.0074, b = 2.4)

class Genotype:
    """ Genotype class for competition and evolution experiments """
    
    def __init__(self, mumax = 1.11, e = 5*10**-9, km = 0.25, n0 = 10**4, gid = 0, mut_rate = 10**-10, survival = 0.0004):
        
        self.id = gid
        self.mumax = mumax
        self.e = e
        self.km = km
        self.n = np.array([n0])
        self.mutations = dict()
        self.survival = survival
        self.survivors = np.array([])
        #self.y0 = np.array([])
        
    def get_pars(self):
        return [self.n, self.mumax, self.km, self.e]
    
    def mutate(self):
        g1=Genotype()
    
        
        
# Put data on genotypes and use getter and setter function in the experiment. This will allow addition of genotypes        
        
# Include volume and sample volume, and Poisson sampling!
# Volume and Poisson sampling are important for emergence and propagation of mutants (absolute numbers are important here!)
# [stats.poisson(g.n[-1]*(sample_vol/vol)).rvs() for g in self.genotypes]

# Include calculation of selection coefficient matrix

# Make carbon source objects? Diauxic shifts and persisters?

#
        
class Experiment:
    """ Experiment class for competition and evolution experiments """
    
    def __init__(self, genotypes, cycles = 15, volume = 10, sample_volume = 0.1,dilution = 100, predilution = 4,
                 gluc_start = 20, t_span = [0,24], killing = True):
        
        self.genotypes = genotypes
        self.cycles = cycles
        #self.volume = volume
        #self.sample_volume = sample_volume
        self.dilution = dilution
        self.predilution = predilution
        # Make self.gluc!
        self.gluc_start = gluc_start
        self.t_span = t_span
        self.ode_args = [[g.mumax, g.km, g.e] for g in self.genotypes]
        self.y0 = np.array([g.n[-1] for g in self.genotypes]+[gluc_start])
        self.ts = np.array([0])
        self.sols = np.array([self.y0])
        self.killing = killing
        self.survivors = np.array([])
        self.fractions = np.array([])
        self.total = np.array([np.sum(self.y0[:-1])])
    
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
    
    def dilute(self):
        self.y0 = np.array([g.n[-1]/self.dilution if g.n[-1]>=self.dilution else 0 for g in self.genotypes]+[self.gluc_start])
    
    def kill(self):
        
        survivors = np.array([(g.n[-1]/self.predilution)*g.survival for g in self.genotypes])
        self.append_genos('survivors', survivors)
        #[(g.survivors = survivors) for g in self.genotypes]
    
        if not len(self.survivors):
            self.survivors = np.array([survivors])
        else:
            self.survivors = np.append(self.survivors, [survivors], axis = 0)
            
        diluted = self.survivors[-1]/self.dilution
        self.y0 = np.append(np.where(diluted>=1, diluted, 0), [self.gluc_start])
        #self.y0 = np.array([(g.n[-1]/self.predilution)*g.survival/self.dilution if (g.n[-1]/self.predilution)*g.survival/self.dilution>=1 else 0 for g in self.genotypes]+[self.gluc_start])
    
    
    def create_ode_vec(self):
        return np.transpose([g.n[-1] for g in self.genotypes])
        
   
    def conduct_once(self):
        
        #def no_gluc(t,y, *args): return y[-1]
        #no_gluc.terminal = False
        #no_gluc.direction = -1
        
        #y0 = np.append(self.from_genos('n')[-1], self.gluc_start)
        
        self.sol = solve_ivp(pop_ivp,self.t_span, y0=self.y0,args=([self.ode_args]),
                                 #dense_output=True, events = no_gluc,
                                 #t_eval = np.linspace(0,24,101),
                                 atol=10**-9,rtol=10**-8)
        
        # This needs adaptation once new Genotypes can emerge: dimensions of sols will change! So put solutions on Genotype objects, not here? Or append solution objects in array, not solutions
        self.append_genos('n',  self.sol.y)
        self.sols = np.append(self.sols, self.sol.y.T, axis = 0)
        self.ts = np.append(self.ts, (self.ts[-1] if self.ts.any() else 0)+self.sol.t)
        self.days = self.ts/24
        self.calc_fractions()
        # SOmething is VERY wrong here with survivors (arrays and lists mixed together...)
        #self.survivors = np.append(self.survivors, [self.y0*self.dilution],axis=0)
        self.total = np.append(self.total, np.sum(self.sol.y[:-1],axis=0))#np.array([np.sum(sol) for sol in self.sol.y[:-1].T]))
        
        #for i,g in enumerate(self.genotypes):
        #    g.n = np.concatenate((g.n,self.sol.y[i]))
            
        
    def conduct(self):
        
        self.conduct_once()
        
        for c in range(self.cycles-1):
            self.kill() if self.killing else self.dilute()
            self.conduct_once()
        self.total_survivors = np.sum(self.survivors, axis =1)
                
        
    def calc_fractions(self):
        self.fractions =  [sol/np.sum(sol) for sol in self.sols[:,:-1]]
        
    def conduct_odeint(self):
        t = np.linspace(self.t_span[0],self.t_span[1],101)
        self.sol = odeint(pop_ivp, self.y0, t, args=(self.ode_args), tfirst = True)
        
    
    def conduct_ode(self):
        r = ode(pop_ivp).set_integrator('vode')
        r.set_initial_value(self.y0, self.t_span[0]).set_f_params(self.ode_args)
        t = np.linspace(self.t_span[0],self.t_span[1],501)
        self.sol = [r.integrate(r.t + (24/501)) for tt in t]
    
    def add_genotype(self, genotype):
        self.genotypes.append(genotype)
        self.y0 = np.array([g.n[-1] for g in self.genotypes]+[self.gluc_start])
