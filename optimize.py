import numpy as np

rng = np.random.default_rng(seed=123)
# cost constants estimated here https://www.nrel.gov/docs/fy22osti/80694.pdf
C_s = 5.56e6  # Estimated cost of solar installation/MWH output = $1.13 * 1 / .2 efficiency * 1e6 mw / w
C_b = 1.5e6  # Cost of storage/MW
C_g = 183  # Retail cost/MWH of grid electricity in nyc

# Tests - beta = 3000, theta = 100
# C_s = 1.1e6  # Interesting value of cs - leads to convergence of theta at 1102
# C_b = 200  # With above leads to beta=2997
# C_g = 3.5e4  # test

N = 365 * 20  # 20 year period

avg_load = 6000  # Daily average MWH electric load of NYC in 2022

# days are indexed by i
# theta and beta gradient updates are indexed by n

# Unfulfilled demand on given day


def X(theta_n, xi, storage_i):
    return xi - storage_i - theta_n


def grad_theta_x(grad_s_i):
    return -1 - grad_s_i


def grad_beta_x(grad_s_i):
    return -grad_s_i


def grad_theta_s_i(s_i_prev, grad_s_i_prev, theta_n, beta_n, xi_i):
    if s_i_prev == 0:
        if theta_n <= xi_i or theta_n >= beta_n + xi_i:
            return 0
        return 1
    elif s_i_prev == beta_n:
        if theta_n <= xi_i - beta_n or theta_n > xi_i:
            return 0
        return 1
    else:
        if theta_n <= xi_i - s_i_prev or theta_n >= beta_n - s_i_prev + xi_i:
            return 0
        return grad_s_i_prev + 1


def grad_beta_s_i(s_i_prev, theta_n, beta_n, xi_i):
    if s_i_prev == 0:
        if beta_n <= theta_n - xi_i:
            return 1
        return 0

    elif s_i_prev == beta_n:
        if beta_n > xi_i - theta_n or theta_n >= xi_i:
            return 1
        return 0
    else:
        if beta_n <= theta_n + s_i_prev - xi_i:
            return 1
        return 0


def h(x):
    return C_g*x if x > 0 else 0


def h_prime(x):
    return C_g if x > 0 else 0


# rate param lambda = 1/avg_load
xis = rng.exponential(scale=avg_load, size=N)

# sum over N days of h


def grid_purchase_cost(theta_n, beta_n, xis):
    total_purchased = 0
    S = []
    for i, xi in enumerate(xis):
        S_i_prev = 0 if i == 0 else S[i-1]
        xi_prev = 0 if i == 0 else xis[i-1]
        S_i = min(beta_n, max(S_i_prev + theta_n - xi_prev, 0))
        S.append(S_i)
        # basically X(theta, beta)
        mwh_purchased = xi - S_i - theta_n if xi - S_i - theta_n > 0 else 0
        total_purchased += mwh_purchased
    return C_g*total_purchased


def J(theta_n, beta_n):
    solar_cost = C_s*theta_n
    storage_cost = C_b*beta_n
    return solar_cost + storage_cost + grid_purchase_cost(theta_n, beta_n, xis)


# TODO combine computation of beta and theta gradients into same
# function to save loops
def grad_grid_purchase_cost(theta_n, beta_n, xis):
    S = []
    grad_theta = 0
    grad_beta = 0
    grad_theta_S_i_prev = 0
    for i, xi in enumerate(xis):
        S_i_prev = 0 if i == 0 else S[i-1]
        xi_prev = 0 if i == 0 else xis[i-1]
        S_i = min(beta_n, max(S_i_prev + theta_n - xi_prev, 0))
        S.append(S_i)
        x = xi - S_i - theta_n
        grad_theta_S_i = grad_theta_s_i(
            S_i_prev, grad_theta_S_i_prev, theta_n, beta_n, xi)
        grad_beta_S_i = grad_beta_s_i(S_i_prev, theta_n, beta_n, xi)
        grad_theta_S_i_prev = grad_theta_S_i
        grad_theta += h_prime(x)*grad_theta_x(grad_theta_S_i)
        grad_beta += h_prime(x)*grad_beta_x(grad_beta_S_i)
    return grad_theta, grad_beta


def grad_J(theta_n, beta_n):
    grad_theta_purch, grad_beta_purch = grad_grid_purchase_cost(
        theta_n, beta_n, xis)
    return C_s + grad_theta_purch, C_b + grad_beta_purch


assert (np.abs(C_g*np.sum(xis) -
        grid_purchase_cost(theta_n=0, beta_n=0, xis=xis)) < 1)
