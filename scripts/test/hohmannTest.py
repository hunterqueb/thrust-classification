import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def twoBody(t, y, mu):
    r = y[0:2]   # (x, y)
    R = np.linalg.norm(r)
    vx, vy = y[2], y[3]
    
    # Gravity
    ax = -mu * r[0] / R**3
    ay = -mu * r[1] / R**3
    
    return [vx, vy, ax, ay]

# Example usage:
mu = 3.986004418e14  # Earth’s mu in m^3/s^2

# Initial state at t=0
# Suppose we start on a circular orbit of radius r1 with velocity v1
r1 = 7000e3
v1 = np.sqrt(mu / r1)
# Let's align it so we start at (r1,0) and velocity in the +y direction
y0 = np.array([r1, 0.0, 0.0, v1])

def run_hohmann_impulse(deltaV=1000.0):

    # 1) Integrate up to time of first burn. 
    #    For a true Hohmann, the first burn might be immediate, t_burn=0, 
    #    but let's pretend we want to wait 1000s (arbitrary).
    t_burn1 = 1000.0

    sol1 = solve_ivp(fun=lambda t, y: twoBody(t, y, mu),
                     t_span=(0, t_burn1),
                     y0=y0, max_step=10.0, rtol=1e-8, atol=1e-10)

    # 2) Apply the delta-v at t_burn1
    #    Extract final state at t_burn1
    y_burn1 = sol1.y[:, -1].copy()   # [x, y, vx, vy]
    
    # Suppose we computed delta-v from standard Hohmann equations (impulsive).
    # Here, just a dummy example: we add 100 m/s in the +y direction
    # get unit vector in direction of velocity
    v = np.array([y_burn1[2], y_burn1[3]])  # (vx, vy)
    v = v/np.linalg.norm(v) * deltaV  # Normalize to get unit vector and apply delta-v in that direction
    
    dvx, dvy = v[0], v[1]  # 100 m/s in the direction of velocity

    y_burn1[2] += dvx
    y_burn1[3] += dvy

    # 3) Continue integration from that new state
    t_burn2 = t_burn1 + 3600.0
    sol2 = solve_ivp(fun=lambda t, y: twoBody(t, y, mu),
                     t_span=(t_burn1, t_burn2),
                     y0=y_burn1, max_step=10.0, rtol=1e-8, atol=1e-10)

    # 4) Possibly apply second burn, etc...

    # Combine results if you like, or analyze in pieces
    # (sol1.t, sol1.y)  and  (sol2.t, sol2.y)

    return sol1, sol2

if __name__ == "__main__":
    sol1,sol2 = run_hohmann_impulse()  # Run the Hohmann transfer simulation
    
    
    originalCircOrbit = solve_ivp(fun=lambda t, y: twoBody(t, y, mu),
                     t_span=(0, 2*(np.pi*np.sqrt((7000e3)**3/mu))),
                     y0=y0, max_step=10.0, rtol=1e-8, atol=1e-10)


    # plot the entire trasnfer trajectory
    plt.figure(figsize=(10, 10))
    plt.title('Hohmann Transfer Orbit')
    plt.plot(originalCircOrbit.y[0], originalCircOrbit.y[1], 'g--', label='Original Circular Orbit')
    plt.plot(sol1.y[0], sol1.y[1], 'b-', label='Before Burn')
    plt.plot(sol2.y[0], sol2.y[1], 'r-', label='After Burn')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()  # Show the plot of the trajectory