import numpy as np
from .vessel import pressure

def inlet_from_data(t: float, heart):
    idt = heart.input_data[:, 0]
    idq = heart.input_data[:, 1]

    t_hat = t // heart.cardiac_period
    t -= t_hat * heart.cardiac_period

    for i in range(len(idt) - 1):
        if idt[i] <= t <= idt[i + 1]:
            idx = i
            break
    else:
        idx = len(idt) - 2  # fallback if loop doesn't break

    return idq[idx] + (t - idt[idx]) * (idq[idx + 1] - idq[idx]) / (idt[idx + 1] - idt[idx])


def wave_speed(A, gamma):
    return np.sqrt(1.5 * gamma * np.sqrt(A))


def riemann_invariants(i: int, v):
    c = wave_speed(v.A[i], v.gamma[i + 1])
    return v.u[i] - 4 * c, v.u[i] + 4 * c


def inv_riemann_invariants(W1: float, W2: float):
    return 0.5 * (W1 + W2)


def incompat(v, dt: float):
    W11, W21 = riemann_invariants(0, v)
    W12, W22 = riemann_invariants(1, v)

    c1 = wave_speed(v.A[0], v.gamma[1])
    W11 += (W12 - W11) * (c1 - v.u[0]) * dt / v.dx
    W21 = 2 * v.Q[0] / v.A[0] - W11

    v.u[0] = inv_riemann_invariants(W11, W21)
    v.A[0] = v.Q[0] / v.u[0]


def inbc(v, t: float, dt: float, heart):
    v.Q[0] = inlet_from_data(t, heart)
    incompat(v, dt)


def outcompat(v, dt: float):
    W1M1, W2M1 = riemann_invariants(v.M - 2, v)
    W1M, W2M = riemann_invariants(v.M - 1, v)

    c = wave_speed(v.A[-1], v.gamma[-2])
    W2M += (W2M1 - W2M) * (v.u[-1] + c) * dt / v.dx
    W1M = v.W1M0 - v.Rt * (W2M - v.W2M0)

    v.u[-1] = inv_riemann_invariants(W1M, W2M)
    v.Q[-1] = v.A[-1] * v.u[-1]


def newtone(f, df, xn, max_iter=10):
    for _ in range(max_iter):
        xn -= f(xn) / df(xn)
    return xn


def wk3(v, dt: float, rho: float):
    if v.inlet_impedance_matching:
        v.R1 = rho * wave_speed(v.A[-1], v.gamma[-2]) / v.A[-1]
        v.R2 = abs(v.total_peripheral_resistance - v.R1)

    v.Pc += dt / v.Cc * (v.A[-1] * v.u[-1] - (v.Pc - v.Pout) / v.R2)

    As = v.A[-1]+ #.001*np.random.uniform(low=1, high=2, size=(1,))[0]
    ssAl = np.sqrt(np.sqrt(v.A[-1]))
    sgamma = 2 * np.sqrt(6 * v.gamma[-2])
    sA0 = np.sqrt(v.A0[-1])
    bA0 = v.beta[-1] / sA0

    def fun(As_):
        return (As_ * v.R1 * (v.u[-1] + sgamma * (ssAl - np.sqrt(np.sqrt(As_)))) -
                (v.Pext + bA0 * (np.sqrt(As_) - sA0)) + v.Pc)

    def dfun(As_):
        return (v.R1 * (v.u[-1] + sgamma * (ssAl - 1.25 * np.sqrt(np.sqrt(As_)))) -
                bA0 * 0.5 / np.sqrt(As_))

    As = newtone(fun, dfun, As)
    us = (pressure(As, v.A0[-1], v.beta[-1], v.Pext) - v.Pout) / (As * v.R1)
    print(v.A[-1] == As)
    v.A[-1] = As
    v.u[-1] = us


def outbc(v, dt: float, rho: float):
    if v.usewk3:
        wk3(v, dt, rho)
    else:
        outcompat(v, dt)


def update_ghost_cells(network):
    for v in network.vessels.values():
        v.U00A = v.A[0]
        v.U00Q = v.Q[0]
        v.UM1A = v.A[-1]
        v.UM1Q = v.Q[-1]

