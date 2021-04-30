    # -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
from . import normal
from . import bsm
import pyfeng as pf
import scipy.integrate as spint

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        '''
        return 0
    
    def price(self, strike, spot, texp=None, sigma=None, cp=1, step=100, iter=10000, seed=12345):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        self.step = step  # number of time steps of MC
        self.iter = iter  # number of iteration of MC

        # np.random.seed(12345)
        np.random.seed(seed)
        # Generate correlated normal random variables W1, Z1
        z = np.random.normal(size=(self.iter, self.step))
        x = np.random.normal(size=(self.iter, self.step))
        w = self.rho * z + np.sqrt(1-self.rho**2) * x

        path_size = np.zeros([self.iter, self.step + 1])   # shape instrument for defining variables below
        delta_tk = texp / self.step                      # length of each time step
        log_sk = np.log(spot) * np.ones_like(path_size)  # log of price
        sk = spot * np.ones_like(path_size)              # price
        sigma_tk = self.sigma * np.ones_like(path_size)  # sigma
        for i in range(self.step):
            log_sk[:, i+1] = log_sk[:, i] + sigma_tk[:, i] * np.sqrt(delta_tk) * w[:, i] - 0.5 * (sigma_tk[:, i]**2) * delta_tk
            sigma_tk[:, i+1] = sigma_tk[:, i] * np.exp(self.vov * np.sqrt(delta_tk) * z[:, i] - 0.5 * (self.vov**2) * delta_tk)
            sk[:, i+1] = np.exp(log_sk[:, i+1])

        price_sabr_bsm_mc = np.zeros_like(strike)
        self.price_mc = np.zeros([self.iter, len(strike)])  # used for cpmputing MC variance
        for j in range(len(strike)):
            self.price_mc[:, j] = np.maximum(sk[:, -1] - strike[j], 0)
            price_sabr_bsm_mc[j] = np.mean(np.maximum(sk[:, -1] - strike[j], 0))

        return price_sabr_bsm_mc

'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None

    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model 
        '''
        return 0
        
    def price(self, strike, spot, texp=None, sigma=None, cp=1, step=100, iter=10000, seed=12345):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        self.step = step  # number of time steps of MC
        self.iter = iter  # number of iteration of MC

        # np.random.seed(12345)
        np.random.seed(seed)
        # Generate correlated normal random variables W1, Z1
        z = np.random.normal(size=(self.iter, self.step))
        x = np.random.normal(size=(self.iter, self.step))
        w = self.rho * z + np.sqrt(1-self.rho**2) * x

        path_size = np.zeros([self.iter, self.step + 1])   # shape instrument for defining variables below
        delta_tk = texp / self.step                      # length of each time step
        sk = spot * np.ones_like(path_size)              # price
        sigma_tk = self.sigma * np.ones_like(path_size)  # sigma
        for i in range(self.step):
            sk[:, i+1] = sk[:, i] + sigma_tk[:, i] * w[:, i] * np.sqrt(delta_tk)
            sigma_tk[:, i+1] = sigma_tk[:, i] * np.exp(self.vov * np.sqrt(delta_tk) * z[:, i] - 0.5 * (self.vov ** 2) * delta_tk)

        price_sabr_norm_mc = np.zeros_like(strike)
        for j in range(len(strike)):
            price_sabr_norm_mc[j] = np.mean(np.maximum(sk[:, -1] - strike[j], 0))

        return price_sabr_norm_mc


'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''

    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        return 0
    
    def price(self, strike, spot, texp=None, cp=1, step=100, iter=10000, seed=12345):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        self.step = step
        self.iter = iter

        # np.random.seed(12345)
        np.random.seed(seed)
        z = np.random.normal(size=(self.iter, self.step))  # Generate normal random variables Z1 driving sigma

        delta_tk = texp / self.step                      # length of each time step
        sigma_tk = self.sigma * np.ones([self.iter, self.step+1])  # sigma
        for i in range(self.step):
            sigma_tk[:, i+1] = sigma_tk[:, i] * np.exp(self.vov * np.sqrt(delta_tk) * z[:, i] - 0.5 * (self.vov ** 2) * delta_tk)

        I = spint.simps(sigma_tk * sigma_tk, dx=texp/self.step) / (self.sigma**2)  # compute I(T) using Simpson's rule
        # I = np.mean(sigma_tk * sigma_tk, axis=1) / (self.sigma**2)
        spot_cond_mc = spot * np.exp(self.rho * (sigma_tk[:, -1] - self.sigma) / self.vov - (self.rho*self.sigma)**2 * texp * I / 2)
        vol_cond_mc = self.sigma * np.sqrt((1 - self.rho**2) * I)

        price_sabr_bsm_cond_mc = np.zeros_like(strike)
        for j in range(len(strike)):
            price_sabr_bsm_cond_mc[j] = np.mean(bsm.price(strike[j], spot_cond_mc, texp, vol_cond_mc))

        return price_sabr_bsm_cond_mc


'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None

    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        return 0
        
    def price(self, strike, spot, texp=None, cp=1, step=100, iter=10000, seed=12345):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        self.step = step
        self.iter = iter

        # np.random.seed(12345)
        np.random.seed(seed)
        z = np.random.normal(size=(self.iter, self.step))  # Generate normal random variables Z1 driving sigma

        delta_tk = texp / self.step                      # length of each time step
        sigma_tk = self.sigma * np.ones([self.iter, self.step+1])  # sigma
        for i in range(self.step):
            sigma_tk[:, i+1] = sigma_tk[:, i] * np.exp(self.vov * np.sqrt(delta_tk) * z[:, i] - 0.5 * (self.vov ** 2) * delta_tk)

        I = spint.simps(sigma_tk * sigma_tk, dx=texp/self.step) / (self.sigma**2)  # compute I(T) using Simpson's rule
        # I = np.mean(sigma_tk * sigma_tk, axis=1) / (self.sigma**2)
        spot_cond_mc = spot + self.rho * (sigma_tk[:, -1] - self.sigma) / self.vov
        vol_cond_mc = self.sigma * np.sqrt((1 - self.rho**2) * I)

        price_sabr_norm_cond_mc = np.zeros_like(strike)
        for j in range(len(strike)):
            price_sabr_norm_cond_mc[j] = np.mean(normal.price(strike[j], spot_cond_mc, texp, vol_cond_mc))

        return price_sabr_norm_cond_mc

