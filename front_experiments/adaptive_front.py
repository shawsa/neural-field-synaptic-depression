import numpy as np

def response(pulse_amp, *, mu, alpha, theta, gamma):
    beta = (1 - 1/gamma)/alpha
    return response_beta(pulse_amp, mu=mu, alpha=alpha, beta=beta, theta=theta)

def response_beta(pulse_amp, *, mu, alpha, beta, theta):
    c = get_speed_beta(mu=mu, alpha=alpha, beta=beta, theta=theta)
    denominator = theta*c*mu**2/(1+c*mu) - mu*alpha**2*beta*c/2/(1+alpha*beta+c*alpha)**2/(1+c*mu)
    numerator = pulse_amp*c*mu
    return numerator/denominator

# generated code

# def get_speed(mu, alpha, beta, theta):
#     return (1.0/4.0)*(-2*alpha*beta*mu*theta - 2*alpha*theta + alpha - 2*mu*theta + np.sqrt(4*( lambda base, exponent: base**exponent )(alpha, 2)*( lambda base, exponent: base**exponent )(beta, 2)*( lambda base, exponent: base**exponent )(mu, 2)*( lambda base, exponent: base**exponent )(theta, 2) - 8*( lambda base, exponent: base**exponent )(alpha, 2)*beta*mu*( lambda base, exponent: base**exponent )(theta, 2) - 4*( lambda base, exponent: base**exponent )(alpha, 2)*beta*mu*theta + 4*( lambda base, exponent: base**exponent )(alpha, 2)*( lambda base, exponent: base**exponent )(theta, 2) - 4*( lambda base, exponent: base**exponent )(alpha, 2)*theta + ( lambda base, exponent: base**exponent )(alpha, 2) + 8*alpha*beta*( lambda base, exponent: base**exponent )(mu, 2)*( lambda base, exponent: base**exponent )(theta, 2) - 8*alpha*mu*( lambda base, exponent: base**exponent )(theta, 2) + 4*alpha*mu*theta + 4*( lambda base, exponent: base**exponent )(mu, 2)*( lambda base, exponent: base**exponent )(theta, 2)))/(alpha*mu*theta)

def get_speed(*, mu, alpha, theta, gamma):
    A = 2*theta*mu*alpha
    B = (2*theta-1)*alpha + 2*theta*mu/gamma
    C = 2*theta/gamma - 1
    return (-B + np.sqrt(B**2 - 4*A*C))/(2*A)

def get_speed_beta(*, mu, alpha, theta, beta):
    gamma = 1/(1 + alpha*beta)
    return get_speed(mu=mu, alpha=alpha, theta=theta, gamma=gamma)


def Q_numeric(x, mu, alpha, beta, theta):
    c = get_speed_beta(mu=mu, alpha=alpha, beta=beta, theta=theta)
    return (lambda input: np.heaviside(input,0.5))(x) + (alpha*beta*np.exp(x*(alpha*beta + 1)/(alpha*c)) + 1)*(lambda input: np.heaviside(input,0.5))(-x)/(alpha*beta + 1)

def U_numeric(x, mu, alpha, beta, theta):
    c = get_speed_beta(mu=mu, alpha=alpha, beta=beta, theta=theta)
    gamma = 1/(1+alpha*beta)
    K1 = (1-gamma)*c*alpha**2*gamma**2/mu/(1-c**2*alpha**2*gamma**2)
    K2 = -1/2/c/mu * ((1-gamma)*c*alpha*gamma/(1-c*alpha*gamma) - gamma)
    r = 1/c/alpha/gamma - 1/c/mu
    left = (theta-gamma - K1/r - K2*c*mu/(c*mu-1))*np.exp(x/c/mu) + gamma + K1/r*np.exp(x/c/alpha/gamma) + K2*c*mu/(c*mu-1)*np.exp(x)
    right = gamma*(1+c*alpha)/2/(1+c*alpha*gamma)/(1+c*mu) * np.exp(-x)
    return np.heaviside(-x, .5)*left + np.heaviside(x, .5)*right
