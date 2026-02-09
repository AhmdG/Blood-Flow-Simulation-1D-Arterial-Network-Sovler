import numpy as np


def get_U_conj(v1, v2):
    return np.array([
        v1.u[-1],
        v2.u[0],
        np.sqrt(np.sqrt(v1.A[-1])),
        np.sqrt(np.sqrt(v2.A[0]))
    ])


def get_F_conj(v1, v2, U, k, W, rho):
    return np.array([
        U[0] + 4 * k[0] * U[2] - W[0],
        U[1] - 4 * k[1] * U[3] - W[1],
        U[0] * U[2]**4 - U[1] * U[3]**4,
        0.5 * rho * U[0]**2 + v1.beta[-1] * (U[2]**2 / np.sqrt(v1.A0[-1]) - 1) -
        (0.5 * rho * U[1]**2 + v2.beta[0] * (U[3]**2 / np.sqrt(v2.A0[0]) - 1))
    ])


def get_J_conj(v1, v2, U, k, rho):
    J = np.zeros((4, 4))

    J[0, 0] = 1.0
    J[1, 1] = 1.0

    J[0, 2] = 4 * k[0]
    J[1, 3] = -4 * k[1]

    J[2, 0] = U[2]**4
    J[2, 1] = -U[3]**4
    J[2, 2] = 4 * U[0] * U[2]**3
    J[2, 3] = -4 * U[1] * U[3]**3

    J[3, 0] = rho * U[0]
    J[3, 1] = -rho * U[1]
    J[3, 2] = 2 * v1.beta[-1] * U[2] / np.sqrt(v1.A0[-1])
    J[3, 3] = -2 * v2.beta[0] * U[3] / np.sqrt(v2.A0[0])

    return J


def NR_conj(U, W, J, F, k, v1, v2, rho, tol=1e-5, max_iter=20):
    for _ in range(max_iter):
        if np.linalg.norm(F) < tol:
            break
        delta = np.linalg.solve(J, -F)
        U += delta
        F = get_F_conj(v1, v2, U, k, W, rho)
        J = get_J_conj(v1, v2, U, k, rho)
    return U


def update_conj(v1, v2, U):
    v1.u[-1] = U[0]
    v2.u[0] = U[1]

    v1.A[-1] = U[2] ** 4
    v1.Q[-1] = v1.u[-1] * v1.A[-1]

    v2.A[0] = U[3] ** 4
    v2.Q[0] = v2.u[0] * v2.A[0]


def join_vessels_conj(v1, v2, rho):
    k = (
        np.sqrt(1.5 * v1.gamma[-1]),
        np.sqrt(1.5 * v2.gamma[0])
    )
    U = get_U_conj(v1, v2)
    W = (
        U[0] + 4 * k[0] * U[2],
        U[1] - 4 * k[1] * U[3]
    )
    F = get_F_conj(v1, v2, U, k, W, rho)
    J = get_J_conj(v1, v2, U, k, rho)

    U = NR_conj(U, W, J, F, k, v1, v2, rho)

    update_conj(v1, v2, U)

