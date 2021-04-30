# -*- coding: utf-8 -*-
"""
Created on Thur Apr 29
Conditional MC for Heston model with QE discretization scheme
@author: xueyang & xiaoyin
"""
import numpy as np
from option_models import bsm
import pyfeng as pf
import scipy.stats as st
import scipy.integrate as spint
from tqdm import tqdm

class HestonCondMC:
    def __init__(self, sigma=0.2, kappa=0.5, theta=0.04, vov=1, rho=-0.9, intr=0, divr=0):
        self.sigma = sigma
        self.kappa = kappa
        self.theta = theta
        self.vov = vov
        self.rho = rho

        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)

    def price(self, strike, spot, texp, delta, psi_c=1.5, path=10000, seed=None):
        '''
        Conditional MC routine goes here
        Generate paths for vol only using QE scheme by Andersen(2008).
        Then compute integrated variance and BSM price.
        Get prices (vector) for all strikes
        '''

        self.path = path
        self.delta = delta   # length of each time step
        self.step = int(texp / self.delta)

        np.random.seed(seed)
        vt = self.sigma ** 2 * np.ones([self.path, self.step + 1])
        u = np.random.uniform(size=(self.path, self.step))

        expo = np.exp(-self.kappa * self.delta)
        for i in tqdm(range(self.step)):
            # first compute m, s_square, psi
            m = self.theta + (vt[:, i] - self.theta) * expo
            s2 = vt[:, i] * (self.vov ** 2) * expo * (1 - expo) / self.kappa + self.theta * (self.vov ** 2) * ((1 - expo) ** 2) / (2 * self.kappa)
            psi = s2 / (m ** 2)

            below = np.where(psi <= psi_c)[0]
            if below != []:
                ins = 2 * psi[below] ** -1
                b2 = ins - 1 + np.sqrt(ins * (ins - 1))
                b = np.sqrt(b2)
                a = m[below] / (1 + b2)
                z = st.norm.ppf(u[below, i])
                vt[below, i+1] = a * (b + z) ** 2

            above = np.where(psi > psi_c)[0]
            if above != []:
                p = (psi[above] - 1) / (psi[above] + 1)
                beta = (1 - p) / m[above]
                for k in range(len(above)):
                    if u[above[k], i] > p[k]:
                       vt[above[k], i+1] = beta[k] ** -1 * np.log((1 - p[k]) / (1 - u[above[k], i]))
                    else:
                       vt[above[k], i+1] = 0

        VT = spint.simps(vt, dx=self.delta)   # compute VT using Simpson's rule
        spot_cmc = spot * np.exp(
            self.rho * (vt[:, -1] - vt[:, 0] - self.kappa * (self.theta - VT)) / self.vov - self.rho ** 2 * VT / 2)
        vol_cmc = np.sqrt((1 - self.rho ** 2) * VT / texp)

        price_heston_cmc = np.zeros_like(strike)
        for j in range(len(strike)):
            price_heston_cmc[j] = np.mean(bsm.price(strike[j], spot_cmc, texp, vol_cmc))

        return price_heston_cmc


'''Euler method
        vt[:, i + 1] = vt[:, i] * np.exp(
            self.vov * np.sqrt(delta) * z[:, i] - 0.5 * (self.vov ** 2) * delta)
        I = spint.simps(sigma_tk * sigma_tk, dx=texp / self.step) / (
                    self.sigma ** 2)  # compute I(T) using Simpson's rule

        spot_cond_mc = spot * np.exp(
            self.rho * (sigma_tk[:, -1] - self.sigma) / self.vov - (self.rho * self.sigma) ** 2 * texp * I / 2)
        vol_cond_mc = self.sigma * np.sqrt((1 - self.rho ** 2) * I)
'''