import numpy as np
import matplotlib.pyplot as plt
import pyfeng as pf
import option_models as opt

'''Test CMC for Heston model using QE scheme:'''
'''beta = 1, BSM model'''

# Parameters
# strike = np.linspace(75, 125, num=25)   # Generate an arithmetic sequence of 25 numbers from 75 to 125
strike = [100.0, 140.0, 70.0]
forward = 100
sigma = 0.2
texp = 10
vov = 1
rho = -0.9
kappa = 0.5
theta = 0.04
beta = 1

heston_cmc_qe = opt.heston.HestonCondMC(sigma, vov=vov, rho=rho, kappa=kappa, theta=theta)
delta = [1, 1/2, 1/4, 1/8, 1/16, 1/32]
price_cmc = np.zeros([len(delta), len(strike)])
for d in range(len(delta)):
    price_cmc[d, :] = heston_cmc_qe.price(strike, forward, texp, delta=delta[d], path=100000, seed=123456)

print('\n' + 'CMC price for Heston model using QE scheme:')
print(price_cmc)

