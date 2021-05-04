import numpy as np
import sv42_cmc_qe as sv42

strike = [100.0, 140.0, 70.0]
forward = 100
delta = [1, 1/2, 1/4, 1/8, 1/16, 1/32]
vov, kappa, rho, texp, theta, sigma = [1, 0.5, -0.9, 10, 0.04, 0.2]
sv42_cmc_qe = sv42.Sv42CondMcQE(a=1, b=1, vov=vov, kappa=kappa, rho=rho, theta=theta)
price_cmc = np.zeros([len(delta), len(strike)])

for d in range(len(delta)):
    price_cmc[d, :] = sv42_cmc_qe.price(strike, forward, texp, sigma=sigma, delta=delta[d], err=1e-6, path=1e4, seed=123456)

np.set_printoptions(suppress=True)
print(price_cmc)

