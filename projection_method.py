import numpy as np
from optimize import grad_theta_J, J
# TODO beta constant for now

# projection onto feasible set is just max of that value and 0


def projection_theta(val):
    return max(val, 0)


def Z(eps_n, theta_n, grad):
    return (1/eps_n)*(theta_n - eps_n*grad - projection_theta(theta_n - eps_n*grad))


def epsilon(n):
    return .0001


stopping_diff = 5e2


def projection_method():
    beta_n = 3000
    theta_n = 100  # try different starts
    last_j = J(theta_n, beta_n)
    new_j = last_j
    thetas = [theta_n]
    Js = [last_j]
    n = 0
    while len(Js) == 1 or np.abs(last_j - new_j) > stopping_diff:
        # while True:
        eps = epsilon(n)
        grad = grad_theta_J(theta_n, beta_n)
        theta_n = theta_n - eps*(grad + Z(eps, theta_n, grad))
        thetas.append(theta_n)
        last_j = new_j
        new_j = J(theta_n, beta_n)
        Js.append(new_j)
        n += 1
        if n % 10 == 0:
            print('grad', grad)
            print('new_j', new_j)
            print('theta_n', theta_n)

    return Js, thetas


Js, thetas = projection_method()
print('Js', Js)
print('Thetas', thetas[-1])
