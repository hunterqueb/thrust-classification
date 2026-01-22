import casadi as ca

def make_time_scaled_flow(m=1.0, k=4.0):
    x = ca.MX.sym("x")
    v = ca.MX.sym("v")
    X = ca.vertcat(x, v)

    tf = ca.MX.sym("tf")

    # original dynamics
    f = ca.vertcat(v, -(k/m)*x)

    # time-scaled dynamics: dX/dtau = tf * f(X)
    ode_tau = tf * f

    dae = {"x": X, "p": tf, "ode": ode_tau}

    I = ca.integrator("I", "cvodes", dae,
    0.0, 1.0,
    {
        "abstol": 1e-12,
        "reltol": 1e-12,
        "max_num_steps": 200000,
    }
    )

    x0 = ca.MX.sym("x0")
    v0 = ca.MX.sym("v0")
    X0 = ca.vertcat(x0, v0)

    out = I(x0=X0, p=tf)
    Xf = out["xf"]

    F = ca.Function(
        "F",
        [x0, v0, tf],
        [Xf[0], Xf[1]],
        ["x0","v0","tf"],
        ["x_tf","v_tf"]
    )

    x_target = ca.MX.sym("x_target")
    x_tf, v_tf = F(x0, v0, tf)

    r = x_tf - x_target
    drdt = ca.jacobian(r, tf)

    R = ca.Function(
        "R",
        [x0, v0, tf, x_target],
        [r, drdt],
        ["x0","v0","tf","x_target"],
        ["r","drdt"]
    )

    return F, R

def newton_time_of_flight(x0, v0, x_target, tf_guess, R, max_iter=30, tol=1e-12):
    tf = float(tf_guess)

    for it in range(max_iter):
        r, drdt = R(x0, v0, tf, x_target)  # tuple outputs
        r = float(r)
        drdt = float(drdt)

        if abs(r) < tol:
            return tf, {"iters": it, "residual": r}

        if abs(drdt) < 1e-14:
            raise RuntimeError(f"dt sensitivity too small at iter {it}: dr/dt={drdt}")

        step = -r / drdt

        # damping / backtracking
        alpha = 1.0
        improved = False
        for _ in range(12):
            tf_try = tf + alpha * step
            if tf_try <= 0:
                alpha *= 0.5
                continue

            r_try, _ = R(x0, v0, tf_try, x_target)
            r_try = float(r_try)

            if abs(r_try) < abs(r):
                tf = tf_try
                improved = True
                break

            alpha *= 0.5

        if not improved:
            raise RuntimeError(f"Line search failed at iter {it}: r={r}, drdt={drdt}, step={step}")

    raise RuntimeError(f"Did not converge in {max_iter} iterations; final residual={r}")

if __name__ == "__main__":
    F, R = make_time_scaled_flow(m=1.0, k=4.0)

    x0 = 1.0
    v0 = 0.0
    x_target = 0.0
    tf_guess = 2.5 * 4

    tf_star, info = newton_time_of_flight(x0, v0, x_target, tf_guess, R)
    print("tf*:", tf_star, info)

    x_tf, v_tf = F(x0, v0, tf_star)
    print("final:", float(x_tf), float(v_tf))

    # Exact solution for comparison
    import numpy as np
    t = np.linspace(0, tf_star, 100)
    x_exact = np.cos(2 * t) * x0
    v_exact = -2 * np.sin(2 * t) * x0

    print("Exact solution at tf*:")
    print("x_exact:", x_exact[-1])
    print("v_exact:", v_exact[-1])

    # Plotting
    import matplotlib.pyplot as plt

    # plot the trajectory from the integrator
    t_num = np.linspace(0, tf_star, 100)
    x_num = []
    v_num = []
    for ti in t_num:
        xi, vi = F(x0, v0, ti)
        x_num.append(float(xi))
        v_num.append(float(vi))
    x_num = np.array(x_num)
    v_num = np.array(v_num)
    plt.figure()
    plt.plot(t_num, x_num, label="Numerical x(t)")
    plt.plot(t, x_exact, "--", label="Exact x(t)")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.legend()
    plt.figure()
    plt.plot(t_num, v_num, label="Numerical v(t)")
    plt.plot(t, v_exact, "--", label="Exact v(t)")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.legend()
    plt.show()