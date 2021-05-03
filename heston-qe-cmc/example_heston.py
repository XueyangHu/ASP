import numpy as np
import heston_qe_cmc as heston

strike = [100.0, 140.0, 70.0]
forward = 100
delta = [1, 1/2, 1/4, 1/8, 1/16, 1/32]
vov, kappa, rho, texp, theta, sigma = [1, 0.5, -0.9, 10, 0.04, 0.2]
heston_cmc_qe = heston.HestonCondMC(vov=vov, kappa=kappa, rho=rho, theta=theta)
price_cmc = np.zeros([len(delta), len(strike)])

for d in range(len(delta)):
    price_cmc[d, :] = heston_cmc_qe.price(strike, forward, texp, sigma=sigma, delta=delta[d], path=10000, seed=123456)

np.set_printoptions(suppress=True)
print(price_cmc)