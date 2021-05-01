import numpy as np
import heston_qe_cmc as heston
import time
from tqdm import tqdm
import pyfeng as pf
import matplotlib.pyplot as plt

'''
Test CMC for Heston model using QE scheme
'''

# strike = np.linspace(75, 125, num=25)   # Generate an arithmetic sequence of 25 numbers from 75 to 125
strike = [100.0, 140.0, 70.0]
forward = 100
delta = [1, 1/2, 1/4, 1/8, 1/16, 1/32]

case = np.zeros([3, 6])
case[0] = [1,   0.5, -0.9, 10, 0.04, np.sqrt(0.04)]
case[1] = [0.9, 0.3, -0.5, 15, 0.04, np.sqrt(0.04)]
case[2] = [1,   1,   -0.3, 5,  0.09, np.sqrt(0.09)]

for i in range(3):
    start = time.time()
    vov, kappa, rho, texp, theta, sigma = case[i]

    heston_cmc_qe = heston.HestonQECondMC(vov=vov, kappa=kappa, rho=rho, theta=theta)
    price_cmc = np.zeros([len(delta), len(strike)])
    for d in range(len(delta)):
        price_cmc[d, :] = heston_cmc_qe.price(strike, forward, texp, sigma=sigma, delta=delta[d], path=1e5, seed=123456)

    end = time.time()
    np.set_printoptions(suppress=True)
    print('Case %s:' % i)
    print(price_cmc)
    print('Running time is %.3f seconds.' % (end - start))

# n = 50
# for i in range(3):
#     start = time.time()
#     vov, kappa, rho, texp, theta, sigma = case[i]
#
#     heston_cmc_qe = heston.HestonQECondMC(vov=vov, kappa=kappa, rho=rho, theta=theta)
#     price_cmc = np.zeros([len(delta), len(strike), n])
#     for j in tqdm(range(n)):
#         for d in range(len(delta)):
#             price_cmc[d, :, j] = heston_cmc_qe.price(strike, forward, texp, sigma=sigma, delta=delta[d], path=1e4)
#
#     end = time.time()
#     np.set_printoptions(suppress=True)
#     print('Case %s:' % i)
#     print(price_cmc.mean(axis=2))
#     print(price_cmc.std(axis=2))
#     print('Running time is %.3f seconds.' % (end - start) + '\n')