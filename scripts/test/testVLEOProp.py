import numpy as np
import torch
import matplotlib.pyplot as plt

# use webagg for plotting
plt.switch_backend('WebAgg')


from qutils.integrators import ode87
from qutils.orbital import OE2ECI, ECI2OE

rng = np.random.default_rng() # Seed for reproducibility

mu = 3.986004418e14  # Earth’s mu in m^3/s^2
R = 6371e3 # radius of earth in m

DU = R
TU = np.sqrt(R**3/mu) # time unit in seconds


pam = [mu,1.08263e-3]

A_sat = 10 # m^2, cross section area of satellite

# Atmospheric model parameters
rho_0 = 1.29 # kg/m^3
c_d = 2.1 #shperical model
h_scale = 5000

numOrbits = 10

def twoBodyJ2Drag(t, y, mu,m_sat):
    # two body problem with J2 perturbation in 6 dimensions taken from astroforge library
    # https://github.com/mit-ll/AstroForge/blob/main/src/astroforge/force_models/_models.py
    # https://github.com/mit-ll/AstroForge/blob/main/src/astroforge/force_models/_forces.py

    # x, v = np.split(y, 2) # orginal line in Astroforge
    # faster than above
    x = y[:3]
    v = y[3:]

    J2 = 4.84165368e-4 * np.sqrt(5)

    M2 = J2 * np.diag(np.array([0.5, 0.5, -1.0]))
    r = np.sqrt(x @ x) # faster than np.linalg.norm(x) (original line in Astroforge)
    v_norm = np.sqrt(v @ v) # faster than np.linalg.norm(v)

    # compute monopole force
    F0 = -mu * x / r**3

    # compute the quadropole force in ITRS
    acc = (mu * R**2 / r**5) * (-5 * x * (x @ M2 @ x) / r**2 + 2 * M2 @ x) + F0

    # ydot = np.hstack((v, acc)) # orginal line in Astroforge
    # faster than above
    ydot = np.empty(6)
    ydot[:3] = v
    ydot[3:] = acc

    rho = rho_0 * np.exp(-(r-R) / h_scale)  # Atmospheric density model
    drag_factor = -0.5 * (rho / m_sat) * c_d * A_sat * v_norm

    a_drag = v * drag_factor
    ydot[3:] += a_drag

    # print(f"rho: {rho}, satellite mass: {m_sat}, a_drag: {a_drag}, force: {np.linalg.norm(ydot[3:]*m_sat)}")

    return ydot

def twoBodyJ2(t, y, mu,m_sat):
    # two body problem with J2 perturbation in 6 dimensions taken from astroforge library
    # https://github.com/mit-ll/AstroForge/blob/main/src/astroforge/force_models/_models.py
    # https://github.com/mit-ll/AstroForge/blob/main/src/astroforge/force_models/_forces.py

    # x, v = np.split(y, 2) # orginal line in Astroforge
    # faster than above
    x = y[:3]
    v = y[3:]

    J2 = 4.84165368e-4 * np.sqrt(5)

    M2 = J2 * np.diag(np.array([0.5, 0.5, -1.0]))
    r = np.sqrt(x @ x) # faster than np.linalg.norm(x) (original line in Astroforge)

    # compute monopole force
    F0 = -mu * x / r**3

    # compute the quadropole force in ITRS
    acc = (mu * R**2 / r**5) * (-5 * x * (x @ M2 @ x) / r**2 + 2 * M2 @ x) + F0

    # ydot = np.hstack((v, acc)) # orginal line in Astroforge
    # faster than above
    ydot = np.empty(6)
    ydot[:3] = v
    ydot[3:] = acc

    return ydot

# Random Conditions for dataset generation
m_sat = 100 * rng.random() + 100 # mass of satellite in kg
a = rng.uniform(R + 100e3,R + 200e3) # random semimajor axis in m
e = 0.01 * rng.random() # eccentricity
i = np.deg2rad(10 * rng.random()) # inclination

h = np.sqrt(mu*a*(1-e)) # specific angular momentum

OE = [a,e,i,0,0,0]
y0 = OE2ECI(OE,mu=mu)
print(y0)

tf = 2*np.pi*a**2*np.sqrt(1-e**2)/h # time of flight

teval = np.linspace(0, tf, 1000) # time to evaluate the solution

t,y = ode87(fun=lambda t, y: twoBodyJ2Drag(t, y, mu,m_sat),tspan=(0, tf),y0=y0, t_eval=teval, rtol=1e-8, atol=1e-10)
tNoDrag,yNoDrag = ode87(fun=lambda t, y: twoBodyJ2(t, y, mu,m_sat),tspan=(0, tf),y0=y0,t_eval=teval, rtol=1e-8, atol=1e-10)


# 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(y[:,0],y[:,1],y[:,2],label='with drag')
ax.plot(yNoDrag[:,0],yNoDrag[:,1],yNoDrag[:,2],label='no drag')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.legend()
ax.set_title('Orbit with and without drag')
plt.axis('equal')

plt.figure()
plt.plot(t,y[:,0]-yNoDrag[:,0],label='x diff with drag')
plt.plot(t,y[:,1]-yNoDrag[:,1],label='y diff with drag')
plt.plot(t,y[:,2]-yNoDrag[:,2],label='z diff with drag')
plt.xlabel('t [s]')
plt.ylabel('difference [m]')
plt.legend()
plt.grid()

plt.show()
