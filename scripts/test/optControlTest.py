#!/usr/bin/env python3
"""
Free-final-state optimal control for a (controlled) Duffing oscillator via single-shooting.

MATLAB -> Python mapping (one-to-one where possible):
---------------------------------------------------------------------------
MATLAB main script lines:
    x0 = [10;2];               --> x0 = np.array([10.0, 2.0])
    xf = [0;0];                --> xf = np.array([0.0, 0.0])
    tf = 2*pi;                 --> tf = 2*np.pi
    alpha = 1; beta = 0.1;     --> alpha = 1.0; beta = 0.1
    W = 1;                     --> W = 1.0  (interpreted as scalar times I_2)

Shooting decision variable:
    V0 = l0;                   --> l0_guess = np.ones(2)

Optimizer call:
    fmincon(@DuffingCostFrFS, ..., @(x) DuffingConstFrFS)
                               --> scipy.optimize.minimize with equality constraint

ODE integration:
    ode45(@DuffingDyn, [0 tf], [x0; l0]) 
                               --> solve_ivp(duffing_dyn, [0, tf], np.r_[x0, l0], ...)

Control:
    u = -X(:,4);               --> u = -lam2(t) = -X[3,:]  (0-based indexing)

Constraints (free final state transversality, W=I):
    ceq = [lam1(tf) - x1(tf) + xf1;
           lam2(tf) - x2(tf) + xf2]
                               --> same in Python

Cost:
    0.5*(x(tf)-xf)' W I (x(tf)-xf) + 0.5*trapz(T, u.^2)
                               --> same with numpy.trapz

Notes:
- This keeps the exact structure of your MATLAB, including using an equality constraint *and* minimizing the cost.
- For robustness, we sample a dense t-grid (t_eval) so that trapz is stable and plots match MATLAB aesthetics.
- No analytical gradients are supplied; SLSQP is generally sufficient for this 2D decision variable.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# ------------------------------------------
# Dynamics (state + costate)  (MATLAB: DuffingDyn)
# ------------------------------------------
def duffing_dyn(t: float, X: ArrayLike, alpha: float, beta: float) -> np.ndarray:
    """
    Canonical equations (Pontryagin) for controlled Duffing with quadratic control cost.

    State:   x1, x2
    Costate: lam1, lam2

    MATLAB:
      dx(1) = x(2)
      dx(2) = -alpha*x(1) - beta*x(1)^3 - x(4)
      dx(3) =  alpha*x(4) + 3*beta*x(1)^2*x(4)
      dx(4) = -x(3)

    Python mirrors the above exactly (0-based indices).
    Control law used in the Hamiltonian minimization is u = -lam2, which is why lam2 enters x2 dynamics as "-x(4)".
    """
    x1, x2, lam1, lam2 = X

    dx1   = x2
    # Dynamics has control u with u = -lam2, hence the "-lam2" term below:
    dx2   = -alpha * x1 - beta * (x1 ** 3) - lam2

    # Costate dynamics from Hx partials:
    dlam1 =  alpha * lam2 + 3.0 * beta * (x1 ** 2) * lam2
    dlam2 = -lam1

    return np.array([dx1, dx2, dlam1, dlam2])


# -------------------------------------------------------
# Rollout given l0   (MATLAB: ode45(..., [x0; l0]))
# -------------------------------------------------------
def rollout(l0: ArrayLike, t0: float, tf: float, x0: ArrayLike, alpha: float, beta: float, num: int = 2001):
    """
    Integrate the canonical ODE from t0 to tf with initial state [x0; l0].

    Returns (T, X) where:
      T: shape (N,), time grid
      X: shape (4,N), stacked [x1, x2, lam1, lam2] over time

    Mirrors MATLAB's [T,X] = ode45(...), except we construct a dense fixed grid via t_eval for stable trapz and plotting.
    """
    Xinit = np.r_[np.asarray(x0, dtype=float), np.asarray(l0, dtype=float)]
    t_eval = np.linspace(t0, tf, num)

    sol = solve_ivp(
        fun=lambda t, X: duffing_dyn(t, X, alpha, beta),
        t_span=(t0, tf),
        y0=Xinit,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        method="RK45",  # MATLAB uses ode45 ~ Dormand–Prince 4/5; RK45 is the direct analogue
    )
    if not sol.success:
        raise RuntimeError(f"IVP solve failed: {sol.message}")

    # sol.y has shape (4, N); transpose to (N, 4) when convenient
    return sol.t, sol.y


# --------------------------------------------------------
# Cost (MATLAB: DuffingCostFrFS)
# --------------------------------------------------------
def cost_free_final(l0: ArrayLike, x0: ArrayLike, xf: ArrayLike, tf: float, alpha: float, beta: float, W: float | np.ndarray) -> float:
    """
    MATLAB:
      f = 0.5*(X(end,1:2)'-xf)' * W*I * (X(end,1:2)'-xf) + 0.5*trapz(T, u.^2)

    Here, W may be scalar or 2x2. We interpret scalar W as W*I_2.
    u(t) = -lam2(t) = -X[3,:]
    """
    T, Y = rollout(l0, 0.0, tf, x0, alpha, beta)
    x_tf = Y[0:2, -1]              # [x1(tf), x2(tf)]
    lam2 = Y[3, :]                 # lam2(t)
    u    = -lam2                   # control

    e = x_tf - xf                  # terminal state error
    if np.isscalar(W):
        term = 0.5 * W * float(e @ e)
    else:
        W = np.asarray(W, dtype=float)
        term = 0.5 * float(e @ (W @ e))

    running = 0.5 * float(np.trapz(u * u, x=T))
    return term + running


# -----------------------------------------------------------------
# Equality constraints (MATLAB: DuffingConstFrFS)
# -----------------------------------------------------------------
def ceq_free_final(l0: ArrayLike, x0: ArrayLike, xf: ArrayLike, tf: float, alpha: float, beta: float) -> np.ndarray:
    """
    MATLAB:
      ceq = [ X(end,3) - X(end,1) + xf(1);
              X(end,4) - X(end,2) + xf(2) ];
      (i.e., lam1(tf) = x1(tf) - xf1,  lam2(tf) = x2(tf) - xf2) with W = I

    That matches the free-final-state transversality for terminal cost (1/2)*||x(tf)-xf||^2.
    """
    T, Y = rollout(l0, 0.0, tf, x0, alpha, beta)
    x1_tf, x2_tf = Y[0, -1], Y[1, -1]
    lam1_tf, lam2_tf = Y[2, -1], Y[3, -1]

    # Enforce exactly zero:
    c1 = lam1_tf - x1_tf + xf[0]
    c2 = lam2_tf - x2_tf + xf[1]
    return np.array([c1, c2], dtype=float)


# ---------------------------------------
# Solve the shooting problem (MATLAB FM)
# ---------------------------------------
def solve_free_final(x0, xf, tf, alpha, beta, W, l0_guess):
    """
    Replicates the MATLAB fmincon call:

    [V_FM_FrFS, ~] = fmincon(@DuffingCostFrFS, l0_guess, ... , @(x) DuffingConstFrFS(...))

    We use SLSQP with an equality constraint. No gradients are provided (small 2D problem).
    """
    def fun(l0):
        return cost_free_final(l0, x0, xf, tf, alpha, beta, W)

    # Equality constraint ceq(l0) = 0
    cons = {
        "type": "eq",
        "fun": lambda l0: ceq_free_final(l0, x0, xf, tf, alpha, beta),
    }

    from scipy.optimize import NonlinearConstraint
    nlc = NonlinearConstraint(lambda l0: ceq_free_final(l0, x0, xf, tf, alpha, beta),
                            lb=np.zeros(2), ub=np.zeros(2))
    res = minimize(fun, l0_guess, method="trust-constr", constraints=[nlc],
                options=dict(xtol=1e-9, verbose=0))

    if not res.success:
        # Mirror MATLAB's habit of exposing exit info; do not soften the message.
        raise RuntimeError(f"Optimization failed: {res.message} (status={res.status})")

    return res.x, res





def cost_min_time_fixed_final(l0: ArrayLike, x0: ArrayLike, xf: ArrayLike, tf: float, alpha: float, beta: float, W: float | np.ndarray) -> float:
    """
    MATLAB:
      f  = tf + 0.5*trapz(T,u.^2);

    u(t) = -lam2(t) = -X[3,:]
    """
    T, Y = rollout(l0, 0.0, tf, x0, alpha, beta)
    x_tf = Y[0:2, -1]              # [x1(tf), x2(tf)]
    lam2 = Y[3, :]                 # lam2(t)
    u    = -lam2                   # control

    term = tf

    running = 0.5 * float(np.trapz(u * u, x=T))
    return term + running

def ceq_min_time_fixed_final(l0: ArrayLike, x0: ArrayLike, xf: ArrayLike, tf: float, alpha: float, beta: float) -> np.ndarray:
    """
    MATLAB:
    ceq = [X(end,1) - xf(1); ...
        X(end,2) - xf(2); ...
        X(end,3) -  nu(1); ...
        X(end,4) - nu(2); ...
        0.5*X(end,4)^2 + X(end,3)*X(end,1) + X(end,4)*(-alpha*X(end,1) - beta*X(end,1)^3 - X(end,4)) + 1];

    """
    T, Y = rollout(l0, 0.0, tf, x0, alpha, beta)
    x1_tf, x2_tf = Y[0, -1], Y[1, -1]
    lam1_tf, lam2_tf = Y[2, -1], Y[3, -1]

    # Enforce exactly zero:
    c1 = x1_tf - xf[0]
    c2 = x2_tf - xf[1]
    c3 = lam1_tf - 0.0
    c4 = lam2_tf - 0.0
    c5 = 0.5 * lam2_tf**2 + lam1_tf * x1_tf + lam2_tf * (-alpha * x1_tf - beta * (x1_tf ** 3) - lam2_tf) + 1.0
    return np.array([c1, c2, c3, c4, c5], dtype=float)

def solve_min_time_fixed_final(x0, xf, tf, alpha, beta, W, l0_guess):
    """
    Replicates the MATLAB fmincon call:

    [V_FM_FrFS, ~] = fmincon(@DuffingCostMTFxFS, l0_guess, ... , @(x) DuffingConstMTFxFS(...))

    We use SLSQP with an equality constraint. No gradients are provided (small 2D problem).
    """
    def fun(l0):
        return cost_min_time_fixed_final(l0, x0, xf, tf, alpha, beta, W)

    # Equality constraint ceq(l0) = 0
    cons = {
        "type": "eq",
        "fun": lambda l0: ceq_min_time_fixed_final(l0, x0, xf, tf, alpha, beta),
    }

    from scipy.optimize import NonlinearConstraint
    nlc = NonlinearConstraint(lambda l0: ceq_min_time_fixed_final(l0, x0, xf, tf, alpha, beta),
                            lb=np.zeros(5), ub=np.zeros(5))
    res = minimize(fun, l0_guess, method="trust-constr", constraints=[nlc],
                options=dict(xtol=1e-9, verbose=0))

    if not res.success:
        # Mirror MATLAB's habit of exposing exit info; do not soften the message.
        raise RuntimeError(f"Optimization failed: {res.message} (status={res.status})")

    return res.x, res

# -------------
# Run + Plot
# -------------
if __name__ == "__main__":
    # ---------------------------
    # Problem setup (MATLAB block)
    # ---------------------------
    x0 = np.array([10.0,  2.0])  # MATLAB: x0 = [10;2];
    xf = np.array([ 0.0,  0.0])  # MATLAB: xf = [0;0];
    tf = 2.0 * np.pi             # MATLAB: tf = 2*pi;

    alpha = 1.0                  # MATLAB: alpha = 1;
    beta  = 0.1                  # MATLAB: beta  = 0.1;

    W = 1.0                      # MATLAB: W = 1;  (scalar multiplying I_2)

    l0 = np.array([1.0, 1.0])  # MATLAB: l0 = ones(2,1);
    nu0 = np.array([0.0, 0.0])  # MATLAB: nu0 = zeros(2,1);



    # Initial guess for costates l(0) = [lam1(0); lam2(0)] (MATLAB: l0 = ones(2,1))
    l0_guess = l0

    # Solve for optimal initial costate l0*
    l0_star, opt_res = solve_free_final(x0, xf, tf, alpha, beta, W, l0_guess)

    # Final rollout with optimal l0
    T, Y = rollout(l0_star, 0.0, tf, x0, alpha, beta)
    x1, x2, lam1, lam2 = Y
    u = -lam2  # MATLAB: u = -X(:,4)

    # Diagnostics comparable to MATLAB
    J = cost_free_final(l0_star, x0, xf, tf, alpha, beta, W)
    ceq_val = ceq_free_final(l0_star, x0, xf, tf, alpha, beta)

    print("Optimal l0* =", l0_star)
    print("Equality constraint residuals (should be ~0):", ceq_val)
    print("Optimal cost J =", J)

    # Plot analogous to MATLAB:
    # figure; plot(T,X(:,1),T,X(:,2),T,u); legend('x','x_d','u');
    plt.figure()
    plt.plot(T, x1, label="x1")
    plt.plot(T, x2, label="x2")
    plt.plot(T, u,  label="u")
    plt.xlabel("t")
    plt.legend()
    plt.title("State and Control (free final state)")
    plt.tight_layout()


    # ---------------------------
    # Fixed final state problem
    # ---------------------------

    # reset initial conditions
    x0 = np.array([10.0,  2.0])  # MATLAB: x0 = [10;2];
    xf = np.array([ 0.0,  0.0])  # MATLAB: xf = [0;0];
    tf = 2.0 * np.pi             # MATLAB: tf = 2*pi;

    alpha = 1.0                  # MATLAB: alpha = 1;
    beta  = 0.1                  # MATLAB: beta  = 0.1;

    W = 1.0                      # MATLAB: W = 1;  (scalar multiplying I_2)

    l0 = np.array([1.0, 1.0])  # MATLAB: l0 = ones(2,1);
    nu0 = np.array([0.0, 0.0])  # MATLAB: nu0 = zeros(2,1);



    l0_guess = [l0,nu0,tf]

    # Solve for optimal initial costate l0*
    l0_star, opt_res = solve_min_time_fixed_final(x0, xf, tf, alpha, beta, W, l0_guess)

    # Final rollout with optimal l0
    T, Y = rollout(l0_star, 0.0, tf, x0, alpha, beta)
    x1, x2, lam1, lam2 = Y
    u = -lam2  # MATLAB: u = -X(:,4)

    # Diagnostics comparable to MATLAB
    J = cost_min_time_fixed_final(l0_star, x0, xf, tf, alpha, beta, W)
    ceq_val = ceq_min_time_fixed_final(l0_star, x0, xf, tf, alpha, beta)

    print("Optimal l0* =", l0_star)
    print("Equality constraint residuals (should be ~0):", ceq_val)
    print("Optimal cost J =", J)

    # Plot analogous to MATLAB:
    # figure; plot(T,X(:,1),T,X(:,2),T,u); legend('x','x_d','u');
    plt.figure()
    plt.plot(T, x1, label="x1")
    plt.plot(T, x2, label="x2")
    plt.plot(T, u,  label="u")
    plt.xlabel("t")
    plt.legend()
    plt.title("State and Control (min time, fixed final state)")
    plt.tight_layout()
    plt.show()
