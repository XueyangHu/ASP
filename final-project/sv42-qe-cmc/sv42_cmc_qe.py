# -*- coding: utf-8 -*-
"""
Created on Sat, May 1, 2021
Conditional MC for 4/2 model based on QE discretization scheme by Andersen(2008)
@author: Xueyang & Xiaoyin
"""
import numpy as np
import pyfeng as pf
import scipy.stats as st
import scipy.integrate as spint
from tqdm import tqdm


class Sv42CondMcQE:
    '''
    Conditional MC for 4/2 model based on QE discretization scheme by Andersen(2008)

    Underlying price is assumed to follow a geometric Brownian motion.
    Volatility (variance) of the price is assumed to follow a CIR process.

    Example:




    '''

    def __init__(self, a, b, vov=1, kappa=0.5, rho=-0.9, theta=0.04):
        '''
        Initiate a 4/2 model

        Args:
            a: coefficient of sigma(t)
            b: coefficient of 1/sigma(t)
            vov: volatility of variance, strictly positive
            kappa: speed of variance's mean-reversion, strictly positive
            rho: correlation between BMs of price and vol
            theta: long-term mean (equilibirum level) of the variance, strictly positive
        '''
        self.a = a
        self.b = b
        self.vov = vov
        self.kappa = kappa
        self.rho = rho
        self.theta = theta

    def price(self, strike, spot, texp, sigma, delta, err, intr=0, divr=0, psi_c=1.5, path=10000, seed=None, scheme='Euler'):
        '''
        Conditional MC routine for 4/2 model
        Generate paths for vol only using QE discretization scheme.
        Compute integrated variance and get BSM prices vector for all strikes.

        Args:
            strike: strike price
            spot: spot (or forward)
            texp: time to expiry
            sigma: initial volatility
            delta: length of each time step
            err: approximation for vt near 0 (since vt should not hit 0 here)
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

        vt = self.sigma ** 2 * np.ones([self.path, self.step + 1])
        np.random.seed(seed)

        u = np.random.uniform(size=(self.path, self.step))

        expo = np.exp(-self.kappa * self.delta)
        # for i in tqdm(range(self.step)):
        for i in range(self.step):
            # compute m, s_square, psi given vt(i)
            m = self.theta + (vt[:, i] - self.theta) * expo
            s2 = vt[:, i] * (self.vov ** 2) * expo * (1 - expo) / self.kappa + self.theta * (self.vov ** 2) * \
                 ((1 - expo) ** 2) / (2 * self.kappa)
            psi = s2 / m ** 2

            # compute vt(i+1) given psi
            below = np.where(psi <= psi_c)[0]
            ins = 2 * psi[below] ** -1
            b2 = ins - 1 + np.sqrt(ins * (ins - 1))
            b = np.sqrt(b2)
            a = m[below] / (1 + b2)
            z = st.norm.ppf(u[below, i])
            vt[below, i+1] = a * (b + z) ** 2

            above = np.where(psi > psi_c)[0]
            p = (psi[above] - 1) / (psi[above] + 1)
            beta = (1 - p) / m[above]
            for k in range(len(above)):
                if u[above[k], i] > p[k]:
                    vt[above[k], i+1] = beta[k] ** -1 * np.log((1 - p[k]) / (1 - u[above[k], i]))
                else:
                    vt[above[k], i+1] = err

        # compute integral of vt and 1/vt, equivalent spot and vol
        # vt_int = spint.simps(vt, dx=self.delta)
        # yt_int = spint.simps(1/vt, dx=self.delta)
        vt_int = spint.trapz(vt, dx=self.delta)
        yt_int = spint.trapz(1/vt, dx=self.delta)


        x1 = self.a * self.rho / self.vov * (vt[:, -1] - vt[:, 0] - self.kappa * (self.theta * texp - vt_int))


        a1 = self.b * self.rho / self.vov
        ln = np.log(vt[:, -1] / vt[:, 0])
        a1 = a1 * ln
        a2 = (self.vov**2 * 0.5 - self.kappa * self.theta) * yt_int
        a3 = a1 * (np.log(vt[:, -1] / vt[:, 0]) + (self.vov**2 * 0.5 - self.kappa * self.theta) * yt_int + self.kappa * texp)

        x2 = self.b * self.rho / self.vov * (np.log(vt[:, -1] / vt[:, 0]) + (self.vov**2 * 0.5 - self.kappa * self.theta) * yt_int + self.kappa * texp)
        x3 = - self.rho**2 * 0.5 * (self.a**2 * vt_int + self.b**2 * yt_int + 2 * self.a * self.b * texp)
        spot_cmc = spot * np.exp(x1 + x2 + x3)
        vol_cmc = np.sqrt((1 - self.rho**2) / texp * (self.a**2 * vt_int + self.b**2 * yt_int +
                                                      2 * self.a * self.b * texp))

        # compute bsm price vector for the given strike vector
        price_cmc = np.zeros_like(strike)
        for j in range(len(strike)):
            price_cmc[j] = np.mean(self.bsm_model.price_formula(strike[j], spot_cmc, vol_cmc, texp))

        return price_cmc

