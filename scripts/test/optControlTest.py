#!/usr/bin/env python3
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


def build_duffing_integrator_fixed_tf(alpha: float, beta: float, tf: float):
    """
    Integrate canonical ODE over [0, tf] with fixed tf using CasADi integrator.
    Augment with running-cost accumulator J where Jdot = 0.5*u^2 and u = -lam2.
    State: [x1, x2, lam1, lam2, J]
    """
    x1, x2, lam1, lam2, J = ca.SX.sym("x1"), ca.SX.sym("x2"), ca.SX.sym("lam1"), ca.SX.sym("lam2"), ca.SX.sym("J")
    X = ca.vertcat(x1, x2, lam1, lam2, J)

    u = -lam2
    dx1 = x2
    dx2 = -alpha * x1 - beta * (x1**3) - lam2
    dlam1 = alpha * lam2 + 3.0 * beta * (x1**2) * lam2
    dlam2 = -lam1
    dJ = 0.5 * (u**2)

    f = ca.Function("f", [X], [ca.vertcat(dx1, dx2, dlam1, dlam2, dJ)])

    dae = {"x": X, "ode": f(X)}
    opts = {
        "tf": float(tf),
        "abstol": 1e-10,
        "reltol": 1e-8,
    }
    # cvodes is typically the most robust choice for smooth ODEs
    F = ca.integrator("F", "cvodes", dae, opts)
    return F


def solve_free_final_state_casadi(x0, xf, tf, alpha, beta, W=1.0, l0_guess=(1.0, 1.0)):
    """
    Decision variable: l0 = [lam1(0), lam2(0)].
    Integrate to tf, minimize:
        0.5*(x(tf)-xf)' W (x(tf)-xf) + integral_0^tf 0.5*u^2 dt
    subject to transversality:
        lam(tf) = W*(x(tf)-xf)
    (Your original code hard-coded W=I. This generalizes scalar W or 2x2 W.)
    """
    x0 = np.asarray(x0, dtype=float).reshape(2)
    xf = np.asarray(xf, dtype=float).reshape(2)

    if np.isscalar(W):
        Wm = float(W) * ca.SX.eye(2)
    else:
        Wm = ca.SX(np.asarray(W, dtype=float).reshape(2, 2))

    F = build_duffing_integrator_fixed_tf(alpha, beta, tf)

    l0 = ca.SX.sym("l0", 2)
    Xinit = ca.vertcat(x0[0], x0[1], l0[0], l0[1], 0.0)  # J(0)=0
    Xf = F(x0=Xinit)["xf"]

    x_tf = Xf[0:2]
    lam_tf = Xf[2:4]
    Jrun = Xf[4]

    e = x_tf - ca.SX(xf)
    Jterm = 0.5 * ca.mtimes([e.T, Wm, e])
    J = Jterm + Jrun

    # transversality
    g = lam_tf - ca.mtimes(Wm, e)

    nlp = {"x": l0, "f": J, "g": g}
    solver = ca.nlpsol(
        "solver",
        "ipopt",
        nlp,
        {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.tol": 1e-10,
            "ipopt.constr_viol_tol": 1e-10,
        },
    )

    sol = solver(
        x0=np.asarray(l0_guess, dtype=float).reshape(2),
        lbg=np.zeros(2),
        ubg=np.zeros(2),
    )
    l0_star = np.array(sol["x"]).reshape(2)

    # Dense rollout for plotting (simple fixed-step RK4 in casadi for speed/portability)
    T, traj = rollout_dense_numeric(x0, l0_star, tf, alpha, beta, N=2001)
    return l0_star, T, traj


def solve_min_time_fixed_final_state_casadi(x0, xf, alpha, beta, tf_guess=2*np.pi, l0_guess=(1.0, 1.0)):
    """
    Minimum-time, fixed final state, free final time tf.

    Decision variables: z = [lam1(0), lam2(0), tf]
    Use time scaling tau in [0,1]:
        dX/dtau = tf * f(X)
    Running cost:
        tf + integral_0^tf 0.5*u^2 dt
      = tf + integral_0^1 tf * 0.5*u(tau)^2 d tau

    Constraints:
        x(tf) = xf
        free-final-time condition: H(tf) + 1 = 0

    NOTE: This implements the CORRECT Hamiltonian term lam1*x2.
    If you insist on matching your original (likely wrong) c5, replace lam1_tf*x2_tf with lam1_tf*x1_tf.
    """
    x0 = np.asarray(x0, dtype=float).reshape(2)
    xf = np.asarray(xf, dtype=float).reshape(2)

    # symbolic state (no explicit time)
    x1, x2, lam1, lam2, J = ca.SX.sym("x1"), ca.SX.sym("x2"), ca.SX.sym("lam1"), ca.SX.sym("lam2"), ca.SX.sym("J")
    X = ca.vertcat(x1, x2, lam1, lam2, J)
    u = -lam2

    # physical-time dynamics
    dx1 = x2
    dx2 = -alpha * x1 - beta * (x1**3) - lam2
    dlam1 = alpha * lam2 + 3.0 * beta * (x1**2) * lam2
    dlam2 = -lam1
    dJ_phys = 0.5 * (u**2)
    f_phys = ca.vertcat(dx1, dx2, dlam1, dlam2, dJ_phys)

    tf_var = ca.SX.sym("tf")
    ode_tau = tf_var * f_phys  # time-scaled
    dae = {"x": X, "p": tf_var, "ode": ode_tau}

    # integrate over tau in [0,1]
    F = ca.integrator(
        "FminT",
        "cvodes",
        dae,
        {
            "tf": 1.0,
            "abstol": 1e-10,
            "reltol": 1e-8,
        },
    )

    lam0 = ca.SX.sym("lam0", 2)
    z = ca.vertcat(lam0, tf_var)

    Xinit = ca.vertcat(x0[0], x0[1], lam0[0], lam0[1], 0.0)
    Xf = F(x0=Xinit, p=tf_var)["xf"]

    x_tf = Xf[0:2]
    lam_tf = Xf[2:4]
    Jrun = Xf[4]

    # Objective: tf + running integral (already in Jrun because Jdot = 0.5 u^2 in physical time,
    # but we used tau dynamics, so Jrun currently equals integral_0^1 tf*0.5*u^2 d tau = integral_0^tf 0.5*u^2 dt.
    J = tf_var + Jrun

    # Fixed-final-state constraints
    g1 = x_tf - ca.SX(xf)

    # Free-final-time Hamiltonian condition:
    # H = 0.5*u^2 + lam1*x2 + lam2*(-alpha*x1 - beta*x1^3 + u) + 1, with u = -lam2
    # Your dynamics uses dx2 = ... - lam2, so u=-lam2 is consistent.
    x1_tf, x2_tf = x_tf[0], x_tf[1]
    lam1_tf, lam2_tf = lam_tf[0], lam_tf[1]

    # Correct Hamiltonian (lam1*x2 term):
    H_tf = 0.5 * (lam2_tf**2) + lam1_tf * x2_tf + lam2_tf * (-alpha * x1_tf - beta * (x1_tf**3) - lam2_tf)
    gH = H_tf + 1.0

    g = ca.vertcat(g1, gH)

    nlp = {"x": z, "f": J, "g": g}
    solver = ca.nlpsol(
        "solver_minT",
        "ipopt",
        nlp,
        {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.tol": 1e-10,
            "ipopt.constr_viol_tol": 1e-10,
        },
    )

    z0 = np.array([l0_guess[0], l0_guess[1], tf_guess], dtype=float)
    lbx = np.array([-ca.inf, -ca.inf, 1e-6], dtype=float)  # keep tf positive
    ubx = np.array([ ca.inf,  ca.inf, 1e6], dtype=float)

    sol = solver(
        x0=z0,
        lbx=lbx,
        ubx=ubx,
        lbg=np.zeros(3),
        ubg=np.zeros(3),
    )
    z_star = np.array(sol["x"]).reshape(3)
    lam0_star = z_star[0:2]
    tf_star = float(z_star[2])

    T, traj = rollout_dense_numeric(x0, lam0_star, tf_star, alpha, beta, N=2001)
    return lam0_star, tf_star, T, traj


def rollout_dense_numeric(x0, l0, tf, alpha, beta, N=2001):
    """
    Dense numeric rollout for plotting using RK4 on the canonical ODE (no CasADi integrator calls in a loop).
    State: [x1, x2, lam1, lam2]
    """
    x0 = np.asarray(x0, float).reshape(2)
    l0 = np.asarray(l0, float).reshape(2)

    T = np.linspace(0.0, float(tf), int(N))
    dt = T[1] - T[0]
    Y = np.zeros((4, int(N)), dtype=float)
    Y[:, 0] = np.array([x0[0], x0[1], l0[0], l0[1]], dtype=float)

    def f(y):
        x1, x2, lam1, lam2 = y
        dx1 = x2
        dx2 = -alpha * x1 - beta * (x1**3) - lam2
        dlam1 = alpha * lam2 + 3.0 * beta * (x1**2) * lam2
        dlam2 = -lam1
        return np.array([dx1, dx2, dlam1, dlam2], dtype=float)

    for k in range(N - 1):
        y = Y[:, k]
        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        Y[:, k + 1] = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return T, Y


if __name__ == "__main__":
    # ---------------------------
    # Problem A: free final state
    # ---------------------------
    x0 = np.array([10.0, 2.0])
    xf = np.array([0.0, 0.0])
    tf = 2.0 * np.pi /3
    alpha = 1.0
    beta = 0.1
    W = 1.0

    l0_star, T, Y = solve_free_final_state_casadi(
        x0=x0, xf=xf, tf=tf, alpha=alpha, beta=beta, W=W, l0_guess=(1.0, 1.0)
    )
    x1, x2, lam1, lam2 = Y
    u = -lam2

    print("Free-final-state:")
    print("  l0* =", l0_star)
    print("  terminal x =", np.array([x1[-1], x2[-1]]))
    print("  terminal lam =", np.array([lam1[-1], lam2[-1]]))

    plt.figure()
    plt.plot(T, x1, label="x1")
    plt.plot(T, x2, label="x2")
    plt.plot(T, u, label="u")
    plt.xlabel("t")
    plt.legend()
    plt.title("State and Control (free final state)")
    plt.grid()
    plt.tight_layout()

    # ---------------------------------------
    # Problem B: minimum time, fixed final
    # ---------------------------------------
    lam0_star, tf_star, T2, Y2 = solve_min_time_fixed_final_state_casadi(
        x0=x0, xf=xf, alpha=alpha, beta=beta, tf_guess=tf, l0_guess=(1.0, 1.0)
    )
    x1b, x2b, lam1b, lam2b = Y2
    ub = -lam2b

    print("\nMin-time, fixed-final-state (free tf):")
    print("  lam0* =", lam0_star)
    print("  tf* =", tf_star)
    print("  terminal x =", np.array([x1b[-1], x2b[-1]]))
    print("  terminal lam =", np.array([lam1b[-1], lam2b[-1]]))

    plt.figure()
    plt.plot(T2, x1b, label="x1")
    plt.plot(T2, x2b, label="x2")
    plt.plot(T2, ub, label="u")
    plt.xlabel("t")
    plt.legend()
    plt.title("State and Control (min time, fixed final state)")
    plt.tight_layout()
    plt.grid()

    plt.show()
