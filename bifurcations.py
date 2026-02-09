import numpy as np


def get_U_bif(v1, v2, v3):
    return np.array([
        v1.u[-1],
        v2.u[0],
        v3.u[0],
        np.sqrt(np.sqrt(v1.A[-1])),
        np.sqrt(np.sqrt(v2.A[0])),
        np.sqrt(np.sqrt(v3.A[0]))
    ])


def get_J_bif(v1, v2, v3, U, k):
    J = np.zeros((6, 6))

    J[0, 0] = 1.0
    J[1, 1] = 1.0
    J[2, 2] = 1.0

    J[0, 3] = 4 * k[0]
    J[1, 4] = -4 * k[1]
    J[2, 5] = -4 * k[2]

    J[3, 0] = U[3] ** 4
    J[3, 1] = -U[4] ** 4
    J[3, 2] = -U[5] ** 4
    J[3, 3] = 4 * U[0] * (U[3] ** 3)
    J[3, 4] = -4 * U[1] * (U[4] ** 3)
    J[3, 5] = -4 * U[2] * (U[5] ** 3)

    J[4, 3] = 2 * v1.beta[-1] * U[3] / np.sqrt(v1.A0[-1])
    J[4, 4] = -2 * v2.beta[0] * U[4] / np.sqrt(v2.A0[0])

    J[5, 3] = 2 * v1.beta[-1] * U[3] / np.sqrt(v1.A0[-1])
    J[5, 5] = -2 * v3.beta[0] * U[5] / np.sqrt(v3.A0[0])

    return J


def get_F(v1, v2, v3, U, k, W):
    return np.array([
        U[0] + 4 * k[0] * U[3] - W[0],
        U[1] - 4 * k[1] * U[4] - W[1],
        U[2] - 4 * k[2] * U[5] - W[2],
        U[0] * U[3] ** 4 - U[1] * U[4] ** 4 - U[2] * U[5] ** 4,
        v1.beta[-1] * (U[3] ** 2 / np.sqrt(v1.A0[-1]) - 1) -
        v2.beta[0] * (U[4] ** 2 / np.sqrt(v2.A0[0]) - 1),
        v1.beta[-1] * (U[3] ** 2 / np.sqrt(v1.A0[-1]) - 1) -
        v3.beta[0] * (U[5] ** 2 / np.sqrt(v3.A0[0]) - 1),
    ])


def NR_bif(U, W, J, F, k, v1, v2, v3, tol=1e-5, max_iter=20):
    for _ in range(max_iter):
        if np.linalg.norm(F) < tol:
            break
        delta = np.linalg.solve(J, -F)
        U += delta
        F = get_F(v1, v2, v3, U, k, W)
        J = get_J_bif(v1, v2, v3, U, k)
    return U


def update_bif(v1, v2, v3, U):
    v1.u[-1] = U[0]
    v2.u[0] = U[1]
    v3.u[0] = U[2]

    v1.A[-1] = U[3] ** 4
    v2.A[0] = U[4] ** 4
    v3.A[0] = U[5] ** 4

    v1.Q[-1] = v1.u[-1] * v1.A[-1]
    v2.Q[0] = v2.u[0] * v2.A[0]
    v3.Q[0] = v3.u[0] * v3.A[0]


def join_vessels(v1, v2, v3):
    k = (
        np.sqrt(1.5 * v1.gamma[-1]),
        np.sqrt(1.5 * v2.gamma[0]),
        np.sqrt(1.5 * v3.gamma[0]),
    )
    U = get_U_bif(v1, v2, v3)
    W = np.array([
        U[0] + 4 * k[0] * U[3],
        U[1] - 4 * k[1] * U[4],
        U[2] - 4 * k[2] * U[5],
    ])
    F = get_F(v1, v2, v3, U, k, W)
    J = get_J_bif(v1, v2, v3, U, k)

    U = NR_bif(U, W, J, F, k, v1, v2, v3)
    update_bif(v1, v2, v3, U)



def join_vessels_bif(v1, v2, v3):
    k = (
        np.sqrt(1.5 * v1.gamma[-1]),
        np.sqrt(1.5 * v2.gamma[0]),
        np.sqrt(1.5 * v3.gamma[0]),
    )
    U = get_U_bif(v1, v2, v3)
    W = np.array([
        U[0] + 4 * k[0] * U[3],
        U[1] - 4 * k[1] * U[4],
        U[2] - 4 * k[2] * U[5],
    ])
    F = get_F(v1, v2, v3, U, k, W)
    J = get_J_bif(v1, v2, v3, U, k)

    U = NR_bif(U, W, J, F, k, v1, v2, v3)
    update_bif(v1, v2, v3, U)


