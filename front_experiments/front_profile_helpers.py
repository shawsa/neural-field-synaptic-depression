import numpy as np

def get_speed_upper(*, mu, alpha, theta, gamma):
    A = 2*theta*mu*alpha
    B = (2*theta-1)*alpha + 2*theta*mu/gamma
    C = 2*theta/gamma - 1
    return (-B + np.sqrt(B**2 - 4*A*C))/(2*A)

def get_speed_lower(*, mu, alpha, theta, gamma):
    A = 2*theta*mu*alpha
    B = (2*theta-1)*alpha + 2*theta*mu/gamma
    C = 2*theta/gamma - 1
    return (-B - np.sqrt(B**2 - 4*A*C))/(2*A)

def get_speed_regressive(*, mu, alpha, theta, gamma):
    assert (theta < gamma) and (gamma < 2*theta)
    return 1/2/mu*(1 - theta/(gamma - theta))

def Q_progressive(x, mu, alpha, gamma, theta, c=None):
    if c is None:
        c = get_speed_upper(mu=mu, alpha=alpha, gamma=gamma, theta=theta)
    return np.heaviside(x, 0.5) + np.heaviside(-x, 0.5)*(gamma + (1-gamma)*np.exp(x/c/alpha/gamma))

def U_progressvie(x, mu, alpha, gamma, theta, c=None):
    if c is None:
        c = get_speed_upper(mu=mu, alpha=alpha, gamma=gamma, theta=theta)
    K1 = (1-gamma)*c*alpha**2*gamma**2/mu/(1-c**2*alpha**2*gamma**2)
    K2 = -1/2/c/mu * ((1-gamma)*c*alpha*gamma/(1-c*alpha*gamma) - gamma)
    r = 1/c/alpha/gamma - 1/c/mu
    left = (theta-gamma - K1/r - K2*c*mu/(c*mu-1))*np.exp(x/c/mu) + gamma + K1/r*np.exp(x/c/alpha/gamma) + K2*c*mu/(c*mu-1)*np.exp(x)
    right = gamma*(1+c*alpha)/2/(1+c*alpha*gamma)/(1+c*mu) * np.exp(-x)
    return np.heaviside(-x, .5)*left + np.heaviside(x, .5)*right

def Q_regressive(x, mu, alpha, theta, gamma, c=None):
    if c is None:
        c = get_speed_regressive(mu=mu, alpha=alpha, gamma=gamma, theta=theta)
    return np.heaviside(-x, 0.5)*gamma + np.heaviside(x, 0.5)*(1 + (gamma-1)*np.exp(x/c/alpha))

def U_regressive(x, mu, alpha, theta, gamma, c=None):
    if c is None:
        c = get_speed_regressive(mu=mu, alpha=alpha, gamma=gamma, theta=theta)
    left = gamma - gamma/2/(1-c*mu)*np.exp(x)
    right = (theta-gamma/2/(1+c*mu))*np.exp(x/c/mu) + gamma/2/(1+c*mu)*np.exp(-x)
    return np.heaviside(-x, 0.5)*left + np.heaviside(x, 0.5)*right
