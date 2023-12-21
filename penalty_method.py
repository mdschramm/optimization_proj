import numpy as np
from optimize import grad_theta_J, J
# Penalty method ===========
# TODO generation and capacity are hard constraints: ideally should use projection method

# theta >= 0


def g(theta_n):
    return -theta_n


grad_g_theta = -1

# penalty


def alpha(n):
    return 100*n

# number of gradient updates per alpha step


def T(n):
    return 200 + 10*n


def grad_theta_alpha_J(theta_nk, beta_nk, alph):
    penalty = alph*(g(theta_nk))*grad_g_theta if g(theta_nk) > 0 else 0
    return grad_theta_J(theta_n=theta_nk, beta_n=beta_nk) + penalty


def epsilon(k):
    return .0001


stopping_diff = 50

# this caused problems with values outside of the feasible set


def penalty_method():
    beta_n = 3000
    theta_nk = 0  # try different starts
    last_j = J(theta_nk, beta_n)
    new_j = last_j
    thetas = [theta_nk]
    Js = [last_j]
    n = 0
    while len(Js) == 1 or np.abs(last_j - new_j) > stopping_diff:
        print('N epochs:', n)
        alpha_n = alpha(n)
        print(theta_nk, new_j)
        for k in range(T(n)):
            theta_nk = theta_nk - \
                epsilon(k)*grad_theta_alpha_J(theta_nk, beta_n, alpha_n)
            thetas.append(theta_nk)
            last_j = new_j
            new_j = J(theta_nk, beta_n)
            Js.append(new_j)
        n += 1
        if n > 10:
            break

    return Js, thetas


Js, thetas = penalty_method()
