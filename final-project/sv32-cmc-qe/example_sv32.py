import numpy as np
import sv32_cmc_qe as sv32

strike = [100.0, 140.0, 70.0]
forward = 100
delta = [1, 1/2, 1/4, 1/8, 1/16, 1/32]
vov, kappa, rho, texp, theta, sigma = [1, 0.5, -0.9, 10, 0.04, np.sqrt(0.04)]
sv32_cmc_qe = sv32.Sv32CondMcQE(vov=vov, kappa=kappa, rho=rho, theta=theta)
price_cmc = np.zeros([len(delta), len(strike)])
for d in range(len(delta)):
    price_cmc[d, :] = sv32_cmc_qe.price(strike, forward, texp, sigma=sigma, delta=delta[d], path=1e5, seed=123456)

np.set_printoptions(suppress=True)
print(price_cmc)