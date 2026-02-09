import numpy as np
#from numba import njit


class Blood:
    def __init__(self, config):
        self.mu = config["mu"]
        self.rho = config["rho"]
        self.Cf = 8 * np.pi * self.mu / self.rho


def wave_speed(A, gamma):
    return np.sqrt(1.5 * gamma * np.sqrt(A))


def pressure(A, A0, beta, Pext):
    return Pext + beta * (np.sqrt(A / A0) - 1.0)


def mesh(config):
    L = config["L"]
    M = max(config.get("M", 5), int(np.ceil(config["L"] * 1e3)))
    dx = L / M
    inv_dx = M / L
    half_dx = 0.5 * dx
    return L, M, dx, inv_dx, half_dx


def radii(config):
    if "R0" not in config and "Rp" not in config:
        raise ValueError(f"missing radius in {config['label']}")
    R0 = config.get("R0", 0.0)
    Rp = config.get("Rp", R0)
    Rd = config.get("Rd", Rp)
    return Rp, Rd


class Vessel:
    def __init__(self, config, blood: Blood, jump: int, tokeep: list[str]):
        self.label = config["label"]
        self.tosave = config.get("to_save", True)
        self.sn = config["sn"]
        self.tn = config["tn"]

        Rp, Rd = radii(config)
        self.Rp = Rp
        self.Rd = Rd
        L, M, dx, inv_dx, half_dx = mesh(config)
        self.L = L

        self.solved = False
        self.M = M
        self.x = np.linspace(0, L, M)
        self.dx = dx
        self.invDx = inv_dx
        self.halfDx = half_dx

        E = config["E"]
        sigma = 0.5
        Pext = config.get("Pext", 0.0)
        initial_pressure = config.get("initial_pressure", 0.0)
        initial_flow = config.get("initial_flow", 0.0)

        A0 = np.zeros(M)
        R0 = np.zeros(M)
        h0 = np.full(M, config.get("h0", 0.0))

        dA0dx = np.zeros(M)
        dTaudx = np.zeros(M)
        radius_slope = (Rd - Rp) / (M - 1)
        self.tapered = not np.isclose(Rd, Rp, atol=1e-4)

        ah, bh, ch, dh = 0.2802, -5.053e2, 0.1324, -11.14

        for i in range(M):
            R0[i] = radius_slope * i * dx + Rp
            if "h0" not in config:
                h0[i] = R0[i] * (ah * np.exp(bh * R0[i]) + ch * np.exp(dh * R0[i]))
            A0[i] = np.pi * R0[i] ** 2
            dA0dx[i] = 2 * np.pi * R0[i] * radius_slope

            if "h0" in config:
                dTaudx[i] = 0.0
            else:
                dTaudx[i] = (np.sqrt(np.pi) * E * radius_slope * 1.3 *
                             (h0[i] / R0[i] + R0[i] *
                              (ah * bh * np.exp(bh * R0[i]) + ch * dh * np.exp(dh * R0[i]))))

        beta = np.sqrt(np.pi / A0) * h0 * E / (1 - sigma ** 2)

        gamma = np.zeros(M + 2)
        gamma[1:M + 1] = beta / (3 * blood.rho * R0 * np.sqrt(np.pi))
        gamma[0] = gamma[1]
        gamma[-1] = gamma[-2]

        viscoelastic = config.get("visco-elastic", False)
        if viscoelastic:
            b0, b1 = 0.6, 0.00150
            Γ = b1 * 0.5 / R0 + b0
            Cv = Γ / (blood.rho * np.sqrt(A0))
        else:
            Cv = np.zeros(M)

        A = A0.copy()
        Q = np.full(M, initial_flow)
        u = Q / A
        P = pressure(A, A0, beta, Pext)

        self.U00A = A0[0]
        self.UM1A = A0[-1]
        self.U00Q = initial_flow
        self.UM1Q = initial_flow

        c = wave_speed(A[-1], gamma[-2])
        self.W1M0 = u[-1] - 4 * c
        self.W2M0 = u[-1] + 4 * c

        self.node2 = round(M * 0.25)
        self.node3 = round(M * 0.5)
        self.node4 = round(M * 0.75)

        self.Rt = config.get("Rt", 0.0)
        self.R1 = config.get("R1", 0.0)
        self.R2 = config.get("R2", self.R1 - blood.rho * wave_speed(A0[-1], gamma[-2]) / A0[-1]
                             if config.get("R2", 0.0) == 0.0 else config["R2"])

        self.total_peripheral_resistance = self.R1 + self.R2
        self.inlet_impedance_matching = config.get("inlet_impedance_matching", False)
        self.usewk3 = self.R2 != 0.0

        self.Cc = config.get("Cc", 0.0)
        self.Pc = 0.0
        self.Pout = config.get("Pout", 0.0)

        # MUSCL arrays
        sz = M + 2
        self.fluxA = np.zeros(sz)
        self.fluxQ = np.zeros(sz)
        self.vA = np.zeros(sz)
        self.vQ = np.zeros(sz)
        self.dUA = np.zeros(sz)
        self.dUQ = np.zeros(sz)
        self.slopesA = np.zeros(sz)
        self.slopesQ = np.zeros(sz)
        self.Al = np.zeros(sz)
        self.Ar = np.zeros(sz)
        self.Ql = np.zeros(sz)
        self.Qr = np.zeros(sz)
        self.Fl = np.zeros(sz)
        self.Fr = np.zeros(sz)

        self.gamma_profile = config.get("gamma_profile", 2)

        self.waveforms = {q: np.zeros((jump, 6)) for q in tokeep}

        # Assign everything else
        self.beta = beta
        self.Cv = Cv
        self.viscoelastic = viscoelastic
        self.gamma = gamma
        self.A0 = A0
        self.dA0dx = dA0dx
        self.dTaudx = dTaudx
        self.Pext = Pext
        self.A = A
        self.Q = Q
        self.u = u
        self.P = P

