import numpy as np
from optimize import grad_J, J
# TODO beta constant for now

# projection onto feasible set is just max of that value and 0


def projection_theta(val):
    return max(val, 0)

# projection is the same for beta and theta because feasible sets are the same


def Z(eps_n, vb, grad):
    return (1/eps_n)*(vb - eps_n*grad - projection_theta(vb - eps_n*grad))


def epsilon(n):
    return 1e-4


stopping_diff = 1e3


def projection_method():
    beta_n = 3000
    theta_n = 100  # try different starts
    last_j = J(theta_n, beta_n)
    new_j = last_j
    thetas = [theta_n]
    betas = [beta_n]
    Js = [last_j]
    n = 0
    while len(Js) == 1 or np.abs(last_j - new_j) > stopping_diff:
        # while True:
        eps = epsilon(n)
        grad_theta, grad_beta = grad_J(theta_n, beta_n)
        theta_n = theta_n - eps*(grad_theta + Z(eps, theta_n, grad_theta))
        beta_n = beta_n - eps*(grad_beta + Z(eps, beta_n, grad_beta))
        thetas.append(theta_n)
        betas.append(beta_n)
        last_j = new_j
        new_j = J(theta_n, beta_n)
        Js.append(new_j)
        n += 1
        if n % 10 == 0:
            print('grad_theta', grad_theta)
            print('grad_beta', grad_beta)
            print('new_j', new_j)
            print('theta_n', theta_n)
            print('beta_n', beta_n)

    return Js, thetas


Js, thetas = projection_method()

'''
Using initial values of beta = 3000, and theta = 100, stopping diff of 1e3, and stepsize .0001
Running these with real-world estimates, both theta and beta are immediately driven to 0. The up-front cost of 
electric generating capacity and storage far outweigh the cost savings over 25 years.

Setting initial beta to 0 and reducing the upfront generating capacity cost C_s to ~20%, 
a convergent value of 1102 up from the initial capacity of 100 is found. Saving ~$72 million.

'''
