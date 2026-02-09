import numpy as np
#from numba import njit
#from scipy.linalg import solve_banded
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from .boundary_conditions import wave_speed, inbc, outbc
from .anastomosis import solve_anastomosis
from .conjunctions import join_vessels_conj
from .bifurcations import join_vessels_bif

def calculate_dt(network):
    min_dt = 1.0
    for vessel in network.vessels.values():
        max_speed = 0.0
        for i in range(len(vessel.u)):
            speed = abs(vessel.u[i] + wave_speed(vessel.A[i], vessel.gamma[i + 1]))
            if np.isnan(speed):
                continue
            max_speed = max(max_speed, speed)
        dt = vessel.dx / max_speed if max_speed > 0 else 1.0
        if np.isnan(dt):
            continue
        min_dt = min(min_dt, dt)
    return min_dt * network.Ccfl


def skip_this(network, s, t):
    if network.vessels[(s, t)].solved:
        return True
    if s == 1:
        return False

    indeg = network.graph.in_degree(s)
    if indeg == 1:
        parent_src = next(iter(network.graph.predecessors(s)))
        return not network.vessels[(parent_src, s)].solved
    elif indeg == 2:
        parents = list(network.graph.predecessors(s))
        return not (network.vessels[(parents[0], s)].solved and network.vessels[(parents[1], s)].solved)

    return False


def solve(network, dt, current_time, NN=True):
    for vessel in network.vessels.values():
        vessel.solved = False

    while not all(v.solved for v in network.vessels.values()):
        for s, t in network.edges:
            #
            #"""
            v = network.vessels[(s, t)]
            v.solved = True
            outdeg = network.graph.out_degree(t)
            if outdeg == 0:
                outbc(v, dt, network.blood.rho)
            else:
                continue
            #"""
            #

            if skip_this(network, s, t):
                continue

            v = network.vessels[(s, t)]

            if s == 1:
                inbc(v, current_time, dt, network.heart)

            outdeg = network.graph.out_degree(t)
            if outdeg == 0:
                outbc(v, dt, network.blood.rho)
            elif outdeg == 1:
                indeg = network.graph.in_degree(t)
                d = next(iter(network.graph.successors(t)))
                if indeg == 1:
                    join_vessels_conj(v, network.vessels[(t, d)], network.blood.rho)
                elif indeg == 2:
                    ps = list(network.graph.predecessors(t))
                    solve_anastomosis(
                        network.vessels[(ps[0], t)],
                        network.vessels[(ps[1], t)],
                        network.vessels[(t, d)],
                    )
            elif outdeg == 2:
                ds = list(network.graph.successors(t))
                join_vessels_bif(v, network.vessels[(t, ds[0])], network.vessels[(t, ds[1])])
            if not NN:
                muscl2(v, dt, network.blood)
            else:
                pass
                # pinn(v, dt, network.blood)
            v.solved = True


def muscl2(v, dt, b):

    dxDt = v.dx / dt
    invDxDt = 1.0 / dxDt
    
    # Step 1
    # Boundary conditions
    v.vA[0] = v.U00A
    v.vQ[0] = v.U00Q
    
    # Interior points
    for i in range(len(v.A)): # do we need the last point? this is what i am confusing the last point isnt just caclulated by wk3 
        #rather it is refined using the wk3
        v.vA[i+1] = v.A[i]
        v.vQ[i+1] = v.Q[i]
    
    # Other boundary
    v.vA[-1] = v.UM1A
    v.vQ[-1] = v.UM1Q
    
    # Limiters
    limiter2(v, v.vA, v.slopesA)
    limiter2(v, v.vQ, v.slopesQ)
    
    # Reconstruct left and right states
    for i in range(len(v.Al)):
        v.Al[i] = v.vA[i] + v.slopesA[i]
        v.Ar[i] = v.vA[i] - v.slopesA[i]
        
        v.Ql[i] = v.vQ[i] + v.slopesQ[i]
        v.Qr[i] = v.vQ[i] - v.slopesQ[i]
        
        v.Fl[i] = v.Ql[i]**2 / v.Al[i] + v.gamma[i] * v.Al[i] * np.sqrt(v.Al[i])
        v.Fr[i] = v.Qr[i]**2 / v.Ar[i] + v.gamma[i] * v.Ar[i] * np.sqrt(v.Ar[i])
    
    # Compute fluxes
    for i in range(v.M + 1):
        v.fluxA[i] = 0.5 * (v.Qr[i+1] + v.Ql[i] - dxDt * (v.Ar[i+1] - v.Al[i]))
        v.fluxQ[i] = 0.5 * (v.Fr[i+1] + v.Fl[i] - dxDt * (v.Qr[i+1] - v.Ql[i]))
    
    # Step 2 - First update
    for i in range(1, v.M + 1):
        v.vA[i] += invDxDt * (v.fluxA[i-1] - v.fluxA[i])
        v.vQ[i] += invDxDt * (v.fluxQ[i-1] - v.fluxQ[i])
    
    # Boundary conditions again
    v.vA[0] = v.vA[1]
    v.vQ[0] = v.vQ[1]
    v.vA[-1] = v.vA[-2]
    v.vQ[-1] = v.vQ[-2]
    
    # Limiters again
    limiter2(v, v.vA, v.slopesA)
    limiter2(v, v.vQ, v.slopesQ)
    
    # Reconstruct again
    for i in range(len(v.Al)):
        v.Al[i] = v.vA[i] + v.slopesA[i]
        v.Ar[i] = v.vA[i] - v.slopesA[i]
        
        v.Ql[i] = v.vQ[i] + v.slopesQ[i]
        v.Qr[i] = v.vQ[i] - v.slopesQ[i]
        
        v.Fl[i] = v.Ql[i]**2 / v.Al[i] + v.gamma[i] * v.Al[i] * np.sqrt(v.Al[i])
        v.Fr[i] = v.Qr[i]**2 / v.Ar[i] + v.gamma[i] * v.Ar[i] * np.sqrt(v.Ar[i])
    
    # Compute fluxes again
    for i in range(v.M + 1):
        v.fluxA[i] = 0.5 * (v.Qr[i+1] + v.Ql[i] - dxDt * (v.Ar[i+1] - v.Al[i]))
        v.fluxQ[i] = 0.5 * (v.Fr[i+1] + v.Fl[i] - dxDt * (v.Qr[i+1] - v.Ql[i]))
    
    # Final update
    for i in range(len(v.A)):
        v.A[i] = 0.5 * (v.A[i] + v.vA[i+1] + invDxDt * (v.fluxA[i] - v.fluxA[i+1]))
        v.Q[i] = 0.5 * (v.Q[i] + v.vQ[i+1] + invDxDt * (v.fluxQ[i] - v.fluxQ[i+1]))
    
    # Source terms
    for i in range(len(v.Q)):
        # Viscosity
        v.Q[i] -= 2 * (v.gamma_profile + 2) * np.pi * b.mu * v.Q[i] / (v.A[i] * b.rho) * dt
        
        if v.tapered:
            # dP/dA0
            v.Q[i] += dt * 0.5 * v.beta[i] * np.sqrt(v.A[i]) * v.A[i] / (v.A0[i] * b.rho) * v.dA0dx[i]
            
            # dP/dh0
            v.Q[i] -= dt * (v.A[i] / b.rho) * (np.sqrt(v.A[i] / v.A0[i]) - 1.0) * v.dTaudx[i]
        
        # Update velocity
        v.u[i] = v.Q[i] / v.A[i]
    
    # Visco-elastic system
    if v.viscoelastic:
        a = v.Cv * dt / (v.dx**2)
        
        # Create tridiagonal matrix
        Tupper = -a[1:]  # Upper diagonal (shifted by +1)
        Tlower = -a[:-1]  # Lower diagonal (shifted by -1)
        Tdiagonal = 1.0 + 2.0 * a
        Tdiagonal[0] -= a[0]
        Tdiagonal[-1] -= a[-1]
        
        # Create sparse matrix and solve
        T = diags([Tlower, Tdiagonal, Tupper], [-1, 0, 1], format='csc')
        
        # Right-hand side
        d = (1.0 - 2.0 * a) * v.Q
        d[0] += a[1] * v.Q[1] + a[0] * v.Q[0]
        d[1:-1] += a[:-2] * v.Q[:-2] + a[2:] * v.Q[2:]
        d[-1] += a[-2] * v.Q[-2] + a[-1] * v.Q[-1]
        
        v.Q = spsolve(T, d)


def limiter2(v, U, slopes):
    for i in range(1, v.M + 2):
        v.dUA[i] = (U[i] - U[i - 1]) * v.invDx
        v.dUQ[i - 1] = (U[i] - U[i - 1]) * v.invDx
    superbee2(slopes, v.dUA, v.dUQ, v.halfDx)



def superbee2(s, a, b, h):
    for i in range(len(s)):
        ai = a[i]
        bi = b[i]
        t1 = max(min(ai, 2 * bi), min(2 * ai, bi))
        t2 = min(max(ai, 2 * bi), max(2 * ai, bi))
        if ai > 0:
            s[i] = t1 * h if bi > 0 else 0.0
        else:
            s[i] = t2 * h if bi < 0 else 0.0

#@njit
def limiter_numba(v_dUA, v_dUQ, U, invDx, M, slopes, halfDx):
    """Numba-compatible limiter function"""
    for i in range(1, M + 2):
        v_dUA[i] = (U[i] - U[i-1]) * invDx
        v_dUQ[i-1] = v_dUA[i]
    
    # Superbee limiter implementation
    for i in range(len(slopes)):
        dUL = v_dUA[i]
        dUR = v_dUQ[i]
        
        if dUL * dUR <= 0:
            slopes[i] = 0.0
        else:
            r = dUL / dUR if abs(dUR) > 1e-10 else 1.0
            phi = max(0.0, min(1.0, 2.0 * r), min(2.0, r))
            slopes[i] = phi * dUL * halfDx

# Main MUSCL function (partially njit-compatible)
def muscl(v, dt, blood):
    # Extract arrays to make them njit-friendly
    dxDt = v.dx / dt
    invDxDt = 1.0 / dxDt
    M = v.M
    
    # Call njit-compiled helper for the core computation
    muscl_core(
        v.vA, v.vQ, v.A, v.Q, v.U00A, v.U00Q, v.UM1A, v.UM1Q, v.dUA, v.dUQ, v.invDx, v.halfDx, v.gamma_profile, v.beta, v.A0, v.dTaudx, v.u, v.dA0dx,
        v.slopesA, v.slopesQ, v.Al, v.Ar, v.Ql, v.Qr, v.Fl, v.Fr,
        v.fluxA, v.fluxQ, v.gamma, dxDt, invDxDt, M, dt, v.tapered, blood.mu, blood.rho
    )
    
    # Handle viscoelastic part separately (can't be njit compiled)
    if v.viscoelastic:
        a = v.Cv * dt / (v.dx**2)
        
        # Create tridiagonal matrix
        Tupper = -a[1:]
        Tlower = -a[:-1]
        Tdiagonal = 1.0 + 2.0 * a
        Tdiagonal[0] -= a[0]
        Tdiagonal[-1] -= a[-1]
        
        T = diags([Tlower, Tdiagonal, Tupper], [-1, 0, 1], format='csc')
        
        # Right-hand side
        d = (1.0 - 2.0 * a) * v.Q
        d[0] += a[1] * v.Q[1] + a[0] * v.Q[0]
        d[1:-1] += a[:-2] * v.Q[:-2] + a[2:] * v.Q[2:]
        d[-1] += a[-2] * v.Q[-2] + a[-1] * v.Q[-1]
        
        v.Q[:] = spsolve(T, d)

#@njit
def muscl_core(vA, vQ, A, Q, U00A, U00Q, UM1A, UM1Q, dUA, dUQ, invDx, halfDx, gamma_profile, beta, A0, dTaudx, u, dA0dx,
               slopesA, slopesQ, Al, Ar, Ql, Qr, Fl, Fr,
               fluxA, fluxQ, gamma, dxDt, invDxDt, M, dt, tapered, mu, rho):
    """Numba-compiled core of MUSCL scheme"""
    # Step 1 - Boundary conditions
    vA[0] = U00A
    vQ[0] = U00Q
    
    # Interior points
    for i in range(len(A)):
        vA[i+1] = A[i]
        vQ[i+1] = Q[i]

    # other boundaries
    vA[-1] = UM1A
    vQ[-1] = UM1Q
    
    # Limiters (using njit-compatible version)
    limiter_numba(dUA, dUQ, vA, invDx, M, slopesA, halfDx)
    limiter_numba(dUA, dUQ, vQ, invDx, M, slopesQ, halfDx)
    
    # Reconstruct left and right states
    for i in range(len(Al)):
        Al[i] = vA[i] + slopesA[i]
        Ar[i] = vA[i] - slopesA[i]

        Ql[i] = vQ[i] + slopesQ[i]
        Qr[i] = vQ[i] - slopesQ[i]

        Fl[i] = Ql[i]**2 / Al[i] + gamma[i] * Al[i] * np.sqrt(Al[i])
        #print(f"gamma[{i}] = {gamma[i]}")
        Fr[i] = Qr[i]**2 / Ar[i] + gamma[i] * Ar[i] * np.sqrt(Ar[i])
    
    # Flux computation
    for i in range(M + 1):
        fluxA[i] = 0.5 * (Qr[i+1] + Ql[i] - dxDt * (Ar[i+1] - Al[i]))
        fluxQ[i] = 0.5 * (Fr[i+1] + Fl[i] - dxDt * (Qr[i+1] - Ql[i]))
    
    # Step 2 -  first Update
    for i in range(1, M + 1):
        vA[i] += invDxDt * (fluxA[i-1] - fluxA[i])
        vQ[i] += invDxDt * (fluxQ[i-1] - fluxQ[i])
    
    # Boundary conditions
    vA[0] = vA[1]
    vQ[0] = vQ[1]

    vA[-1] = vA[-2]
    vQ[-1] = vQ[-2]
    

    # Repeat limiter and reconstruction
    limiter_numba(dUA, dUQ, vA, invDx, M, slopesA, halfDx)
    limiter_numba(dUA, dUQ, vQ, invDx, M, slopesQ, halfDx)


    # reconstruct again
    for i in range(len(Al)):
        Al[i] = vA[i] + slopesA[i]
        Ar[i] = vA[i] - slopesA[i]

        Ql[i] = vQ[i] + slopesQ[i]
        Qr[i] = vQ[i] - slopesQ[i]

        Fl[i] = Ql[i]**2 / Al[i] + gamma[i] * Al[i] * np.sqrt(Al[i])
        Fr[i] = Qr[i]**2 / Ar[i] + gamma[i] * Ar[i] * np.sqrt(Ar[i])

    # compute fluxes again
    for i in range(M + 1):
        fluxA[i] = 0.5 * (Qr[i+1] + Ql[i] - dxDt * (Ar[i+1] - Al[i]))
        fluxQ[i] = 0.5 * (Fr[i+1] + Fl[i] - dxDt * (Qr[i+1] - Ql[i]))
    
    # Final update
    for i in range(len(A)):
        A[i] = 0.5 * (A[i] + vA[i+1] + invDxDt * (fluxA[i] - fluxA[i+1]))
        Q[i] = 0.5 * (Q[i] + vQ[i+1] + invDxDt * (fluxQ[i] - fluxQ[i+1]))


    # Source terms
    for i in range(len(Q)):
        # Viscosity
        Q[i] -= 2 * (gamma_profile + 2) * np.pi * mu * Q[i] / (A[i] * rho) * dt
        
        if tapered:
            # dP/dA0
            Q[i] += dt * 0.5 * beta[i] * np.sqrt(A[i]) * A[i] / (A0[i] * rho) * dA0dx[i]
            
            # dP/dh0
            Q[i] -= dt * (A[i] / rho) * (np.sqrt(A[i] / A0[i]) - 1.0) * dTaudx[i]
        
        # Update velocity
        u[i] = Q[i] / A[i]
