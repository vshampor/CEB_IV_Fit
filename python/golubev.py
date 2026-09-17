import scipy.integrate as integrate
from numpy import sqrt, exp, pi, abs, log, heaviside, cosh, inf, tanh


def I_dless(v, tau):
    return sqrt(abs((2.0 * pi * tau * (1.0 + 0.375 * tau - 0.1171875 * tau * tau) ** 2) *
                    exp(-((abs(v) - 1.0 - (1.15 + tau) * tau) / tau)) /
                    (1.0 + exp(-((abs(v) - 1.0 - (1.15 + tau) * tau) / tau))) + (v * v - 1.0) *
                    exp(-(1.0 - v * v) / tau) /
                    (1.0 + exp(-(1.0 - v * v) / tau)))) * \
        (exp(-(1 - v) / tau) / (2 + exp(-(1 - v) / tau)) - exp(-(1 + v) / tau) / (2 + exp(-(1 + v) / tau)))


def I(u, T, Rn, Delta):
    v = 1.6e-19 * u / abs(Delta)
    tau = 1.38e-23 * abs(T / Delta)
    return Delta / (1.6e-19 * Rn) * I_dless(v, tau)


def Ilog(u, T, Rn, Delta):
    return log(I(u, T, Rn, Delta))


def Fp(v, tau):
    return sqrt(2 * pi * tau) * ((0.7147 * tau + 0.9419) * (1 - v) / (2 * exp((1 - v) / tau) + 1.187 + 0.09 * tanh(45 * (tau - 0.096))) \
            + (-2.135 * tau ** 2 + 0.9164 * tau + 0.4865) * tau / (2 * exp((1 - v) / tau) + 78845 * tau ** 5 - 35903 * tau ** 4 + 6050 * tau ** 3 - 469.66 * tau ** 2 + 17.27 * tau + 0.2927)) * \
           (1 / (exp(-2.5 * (1 + 2 * tau - v) / tau) + 1)) + 1 * heaviside(v - 1 - tau, 0) * (-0.5 * v * sqrt(abs(v ** 2 - 1)) + \
           0.5 * log(abs(v + sqrt(abs(v ** 2 - 1)))) + pi ** 2 * tau ** 2 / 6 * v / sqrt(abs(v ** 2 - 1))) * \
           (1 / (exp(2.5 * (1 + 2 * tau - v) / tau) + 1))


def Pcool(v, taun, tau):
    return Fp(v, taun) + Fp(-v, taun) - 2 * Fp(0, tau)


def Fp_integrand(x, v, tau):
    return (0.5 * log(x + sqrt(x ** 2 - 1)) + (0.5 * x - v) * sqrt(x ** 2 - 1)) / (4 * tau * (cosh(0.5 * (x - v) / tau)) ** 2)


def Fp_int(v, tau):
    return integrate.quad(Fp_integrand, 1.0, inf, args=(v, tau))[0]


def Pcool_integral(v, taun, tau):
    return Fp_int(v, taun) + Fp_int(-v, taun) - 2 * Fp_int(0, tau)


def Pcool_dim(u, Tn, Ts, Rn, Delta):
    v = 1.6e-19 * u / abs(Delta)
    tau = 1.38e-23 * abs(Ts / Delta)
    taun = 1.38e-23 * abs(Tn / Delta)
    return Delta ** 2 / (2.56e-38 * Rn) * Pcool_integral(v, taun, tau)


def Pcool_dim_approx(u, Tn, Ts, Rn, Delta):
    v = 1.6e-19 * u / abs(Delta)
    tau = 1.38e-23 * abs(Ts / Delta)
    taun = 1.38e-23 * abs(Tn / Delta)
    return Delta ** 2 / (2.56e-38 * Rn) * Pcool(v, taun, tau)