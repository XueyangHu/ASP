import numpy as np
import sv32_cmc_qe as sv32
import time
import pyfeng as pf
import matplotlib.pyplot as plt
from tqdm import tqdm

'''
Test CMC for 3/2 model using QE scheme
'''

# strike = np.linspace(75, 125, num=25)
strike = [100.0, 140.0, 70.0]
forward = 100
delta = [1, 1/2, 1/4, 1/8, 1/16, 1/32]

case = np.zeros([3, 6])
# Given by the paper
case[0] = [1,   0.5, -0.9, 10, 0.04, np.sqrt(0.04)]
case[1] = [0.9, 0.3, -0.5, 15, 0.04, np.sqrt(0.04)]
case[2] = [1,   1,   -0.3, 5,  0.09, np.sqrt(0.09)]

# Example
# case[0] = [8.56, 22.84, -0.99, 0.5,  0.218, 0.245]

for i in range(3):
    start = time.time()
    vov, kappa, rho, texp, theta, sigma = case[i]

    sv32_cmc_qe = sv32.Sv32CondMcQE(vov=vov, kappa=kappa, rho=rho, theta=theta)
    price_cmc = np.zeros([len(delta), len(strike)])
    for d in range(len(delta)):
        price_cmc[d, :] = sv32_cmc_qe.price(strike, forward, texp, sigma=sigma, delta=delta[d], path=1e4, seed=123456)

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
#     sv32_cmc_qe = sv32.Sv32CondMcQE(vov=vov, kappa=kappa, rho=rho, theta=theta)
#     price_cmc = np.zeros([len(delta), len(strike), n])
#     for j in tqdm(range(n)):
#         for d in range(len(delta)):
#             price_cmc[d, :, j] = sv32_cmc_qe.price(strike, forward, texp, sigma=sigma, delta=delta[d], path=1e4)
#
#     end = time.time()
#     np.set_printoptions(suppress=True)
#     print('Case %s:' % i)
#     print(price_cmc.mean(axis=2))
#     print(price_cmc.std(axis=2))
#     print('Running time is %.3f seconds.' % (end - start) + '\n')