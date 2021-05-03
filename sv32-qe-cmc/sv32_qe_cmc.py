# -*- coding: utf-8 -*-
"""
Created on Mon, May 3, 2021
Conditional MC for 3/2 model based on QE discretization scheme by Andersen(2008)
@author: xueyang
"""
import numpy as np
import pyfeng as pf
import scipy.stats as st
import scipy.integrate as spint
from tqdm import tqdm


class SV32QECondMC:
    '''
    Conditional MC for 3/2 model based on QE discretization scheme by Andersen(2008)

    Underlying price is assumed to follow a geometric Brownian motion.

    Example:

    '''

    def __init__(self, vov=1, kappa=0.5, rho=-0.9, theta=0.04):
        '''
        Initiate a 3/2 model

        Args:
            vov: volatility of variance, strictly positive
            kappa: speed of variance's mean-reversion, strictly positive
            rho: correlation between BMs of price and vol
            theta: long-term mean (equilibirum level) of the variance, strictly positive
        '''
        self.vov = vov
        self.kappa = kappa
        self.rho = rho
        self.theta = theta

    def price(self, strike, spot, texp, sigma, delta, intr=0, divr=0, psi_c=1.5, path=10000, seed=None):
        '''
        Conditional MC routine for 3/2 model
        Generate paths for vol only using QE discretization scheme.
        Compute integrated variance and get BSM prices vector for all strikes.

        Args:
            strike: strike price
            spot: spot (or forward)
            texp: time to expiry
            sigma: initial volatility
            delta: length of each time step
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            psi_c: critical value for psi, lying in [1, 2]
            path: number of vol paths generated
            seed: random seed for rv generation

        Return:
            BSM price vector for all strikes
        '''
        self.sigma = sigma
        self.bsm_model = pf.Bsm(self.sigma, intr=intr, divr=divr)
        self.delta = delta
        self.path = int(path)
        self.step = int(texp / self.delta)

        xt = 1 / self.sigma**2 * np.ones([self.path, self.step + 1])  # xt = 1 / vt
        np.random.seed(seed)
        u = np.random.uniform(size=(self.path, self.step))

        # equivalent kappa and theta for xt to follow a Heston model
        kappa_new = self.kappa * self.theta
        theta_new = (self.kappa + self.vov**2) / (self.kappa * self.theta)
        expo = np.exp(-kappa_new * self.delta)
        for i in range(self.step):
            # compute m, s_square, psi given xt(i)
            m = theta_new + (xt[:, i] - theta_new) * expo
            s2 = xt[:, i] * (self.vov ** 2) * expo * (1 - expo) / kappa_new + theta_new * (self.vov ** 2) * \
                 ((1 - expo) ** 2) / (2 * kappa_new)
            psi = s2 / m ** 2

            # compute xt(i+1) given psi
            below = np.where(psi <= psi_c)[0]
            ins = 2 * psi[below] ** -1
            b2 = ins - 1 + np.sqrt(ins * (ins - 1))
            b = np.sqrt(b2)
            a = m[below] / (1 + b2)
            z = st.norm.ppf(u[below, i])
            xt[below, i+1] = a * (b + z) ** 2

            above = np.where(psi > psi_c)[0]
            p = (psi[above] - 1) / (psi[above] + 1)
            beta = (1 - p) / m[above]
            for k in range(len(above)):
                if u[above[k], i] > p[k]:
                    xt[above[k], i+1] = beta[k] ** -1 * np.log((1 - p[k]) / (1 - u[above[k], i]))
                else:
                    xt[above[k], i+1] = 0

        # compute integral of vt, equivalent spot and vol
        vt_int = spint.simps(1/xt, dx=self.delta)
        vt = 1 / xt
        spot_cmc = spot * np.exp(self.rho / self.vov * (np.log(vt[:, -1] / vt[:, 0]) - self.kappa * (self.theta * texp - vt_int * (1 + self.vov**2 * 0.5 / self.kappa))) - self.rho ** 2 * vt_int / 2)
        vol_cmc = np.sqrt((1 - self.rho ** 2) * vt_int / texp)

        # compute bsm price vector for the given strike vector
        price_cmc = np.zeros_like(strike)
        for j in range(len(strike)):
            price_cmc[j] = np.mean(self.bsm_model.price_formula(strike[j], spot_cmc, vol_cmc, texp, intr=intr, divr=divr))

        return price_cmc

