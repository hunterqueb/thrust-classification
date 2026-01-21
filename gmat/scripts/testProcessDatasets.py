import numpy as np
import matplotlib.pyplot as plt
from qutils.orbital import dim2NonDim6,orbitalEnergy

dt = 60

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--propMin', type=int, default=10, help='Number of minutes of propagation')
parser.add_argument('--systems', type=int, default=10000, help='Number of random systems')
parser.add_argument('--orbit', type=str, default='vleo', help='Type of orbit')
parser.add_argument('--norm', action='store_true', help='Normalize the states')
parser.add_argument('--randPlots', type=int, default=20, help='Number of random plots to generate')

args = parser.parse_args()
numMinProp = args.propMin
numRandSys = args.systems
orbitType = args.orbit
norm = args.norm
randPlots = args.randPlots

print(f"Processing datasets for {orbitType} with {numMinProp} minutes and {numRandSys} random systems.")

import yaml
with open("data.yaml", 'r') as f:
    dataConfig = yaml.safe_load(f)
dataLoc = dataConfig["classification"] + orbitType + "/" + str(numMinProp) + "min-" + str(numRandSys)

# get npz files in folder and load them into script

a = np.load(f"{dataLoc}/statesArrayChemical.npz")
statesArrayChemical = a['statesArrayChemical']

a = np.load(f"{dataLoc}/statesArrayElectric.npz")
statesArrayElectric = a['statesArrayElectric']

a = np.load(f"{dataLoc}/statesArrayImpBurn.npz")
statesArrayImpBurn = a['statesArrayImpBurn']

a = np.load(f"{dataLoc}/statesArrayNoThrust.npz")
statesArrayNoThrust = a['statesArrayNoThrust']

print(statesArrayChemical.shape)
print(statesArrayElectric.shape)
print(statesArrayImpBurn.shape)
print(statesArrayNoThrust.shape)

if norm:
    for i in range(statesArrayChemical.shape[0]):
        statesArrayChemical[i,:,:] = dim2NonDim6(statesArrayChemical[i,:,:])
        statesArrayElectric[i,:,:] = dim2NonDim6(statesArrayElectric[i,:,:])
        statesArrayImpBurn[i,:,:] = dim2NonDim6(statesArrayImpBurn[i,:,:])
        statesArrayNoThrust[i,:,:] = dim2NonDim6(statesArrayNoThrust[i,:,:])

energyChemical = np.zeros((statesArrayChemical.shape[0],statesArrayChemical.shape[1],1))
energyElectric= np.zeros((statesArrayChemical.shape[0],statesArrayChemical.shape[1],1))
energyImpBurn= np.zeros((statesArrayChemical.shape[0],statesArrayChemical.shape[1],1))
energyNoThrust= np.zeros((statesArrayChemical.shape[0],statesArrayChemical.shape[1],1))
for i in range(statesArrayChemical.shape[0]):
    energyChemical[i,:,0] = orbitalEnergy(statesArrayChemical[i,:,:])
    energyElectric[i,:,0] = orbitalEnergy(statesArrayElectric[i,:,:])
    energyImpBurn[i,:,0] = orbitalEnergy(statesArrayImpBurn[i,:,:])
    energyNoThrust[i,:,0] = orbitalEnergy(statesArrayNoThrust[i,:,:])

t = np.linspace(0,numMinProp*dt,len(statesArrayChemical[0,:,0]))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(statesArrayChemical[0,:,0],statesArrayChemical[0,:,1],statesArrayChemical[0,:,2],label='Chemical')
ax.plot(statesArrayElectric[0,:,0],statesArrayElectric[0,:,1],statesArrayElectric[0,:,2],label='Electric')
ax.plot(statesArrayImpBurn[0,:,0],statesArrayImpBurn[0,:,1],statesArrayImpBurn[0,:,2],label='Impulsive')
ax.plot(statesArrayNoThrust[0,:,0],statesArrayNoThrust[0,:,1],statesArrayNoThrust[0,:,2],label='No Thrust')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('3D Trajectory of a Single Earth Orbiter')
ax.legend(loc='lower left')
ax.axis('equal')

from matplotlib.lines import Line2D
colors = ['C0', 'C1', 'C2', 'C3']
lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
labels = ['Chemical Thrust', 'Electrical Thrust', 'Impulsive Thrust', 'No Thrust']
ax.legend(lines, labels)
ax.axis('equal')

plt.figure()
for j in range(randPlots):
    i = np.random.randint(0, len(statesArrayChemical))

    plt.plot(t, energyChemical[i,:,0], label='Chemical',color='C0')
    plt.plot(t, energyElectric[i,:,0], label='Electric',color='C1')
    plt.plot(t, energyImpBurn[i,:,0], label='Impulsive',color='C2')
    plt.plot(t, energyNoThrust[i,:,0], label='No Thrust',color='C3')
plt.legend(lines, labels)
plt.grid()
plt.xlabel('Time (s)')
plt.title("Energy of "+str(randPlots*4)+" Earth Orbiters")

plt.figure()
plt.plot(t, statesArrayChemical[0,:,0], label='Chemical X')
plt.plot(t, statesArrayElectric[0,:,0], label='Electric X')
plt.plot(t, statesArrayImpBurn[0,:,0], label='Impulsive X')
plt.plot(t, statesArrayNoThrust[0,:,0], label='No Thrust X')
plt.plot(t, statesArrayChemical[0,:,1], label='Chemical Y')
plt.plot(t, statesArrayElectric[0,:,1], label='Electric Y')
plt.plot(t, statesArrayImpBurn[0,:,1], label='Impulsive Y')
plt.plot(t, statesArrayNoThrust[0,:,1], label='No Thrust Y')
plt.plot(t, statesArrayChemical[0,:,2], label='Chemical Z')
plt.plot(t, statesArrayElectric[0,:,2], label='Electric Z')
plt.plot(t, statesArrayImpBurn[0,:,2], label='Impulsive Z')
plt.plot(t, statesArrayNoThrust[0,:,2], label='No Thrust Z')
plt.xlabel('Time (s)')
plt.ylabel('Position (km)')
plt.title('Position vs Time for Different Thruster Profiles')
plt.legend(loc='lower left')
plt.grid()

def plotDiffFromNoThrust(statesArray, label):
    plt.figure()
    plt.plot(t, statesArray[0,:,0]-statesArrayNoThrust[0,:,0], label=label+' X')
    plt.plot(t, statesArray[0,:,1]-statesArrayNoThrust[0,:,1], label=label+' Y')
    plt.plot(t, statesArray[0,:,2]-statesArrayNoThrust[0,:,2], label=label+' Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Position Difference from No Thrust (km)')
    plt.title(f'Position Difference from No Thrust vs Time for {label} Thruster Profile')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(t, statesArray[0,:,3]-statesArrayNoThrust[0,:,3], label=label+' VX')
    plt.plot(t, statesArray[0,:,4]-statesArrayNoThrust[0,:,4], label=label+' VY')
    plt.plot(t, statesArray[0,:,5]-statesArrayNoThrust[0,:,5], label=label+' VZ')
    plt.xlabel('Time (s)')
    plt.ylabel('Position Difference from No Thrust (km/s)')
    plt.title(f'Position Difference from No Thrust vs Time for {label} Thruster Profile')
    plt.legend()
    plt.grid()


# plotDiffFromNoThrust(statesArrayChemical, 'Chemical')
# plotDiffFromNoThrust(statesArrayElectric, 'Electric')
# plotDiffFromNoThrust(statesArrayImpBurn, 'Impulsive')

R_E = 6378.137  # km

def plot_earth_fast(ax, R=R_E, u_res=60, v_res=30):
    u = np.linspace(0, 2*np.pi, u_res)
    v = np.linspace(0, np.pi, v_res)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones_like(u), np.cos(v))
    surf = ax.plot_surface(
        x, y, z,
        linewidth=0, antialiased=False, shade=False,
        color="#A8922D", alpha=0.3, zorder=0
    )
    # make sure the sphere sits between back and front lines in the 3D sort
    try: surf.set_sort_zpos(0.0)
    except Exception: pass
    surf.set_rasterized(True)
    return surf

def _view_vec(ax):
    A = np.deg2rad(ax.azim); E = np.deg2rad(ax.elev)
    v = np.array([np.cos(E)*np.cos(A), np.cos(E)*np.sin(A), np.sin(E)])
    return v / np.linalg.norm(v)

def draw_orbits_translucent(ax, orbits, R=R_E, tag="__orbit__",
                            front_kw=None, back_kw=None):
    # remove previously drawn segments from earlier views
    for ln in list(ax.lines):
        if getattr(ln, "_tag", None) == tag:
            ln.remove()

    v = _view_vec(ax)
    for kind, (x, y, z) in orbits:
        p_dot_v = x*v[0] + y*v[1] + z*v[2]
        r2 = x*x + y*y + z*z
        perp2 = r2 - p_dot_v**2
        visible = (p_dot_v >= 0) | (perp2 >= R*R)

        # split into contiguous visible/invisible segments
        cuts = np.flatnonzero(np.diff(visible.astype(np.int8)) != 0) + 1
        for seg in np.split(np.arange(x.size), cuts):
            if seg.size < 2:
                continue
            is_front = bool(visible[seg[0]])
            kw = (front_kw or {}).get(kind, {}) if is_front else (back_kw or {}).get(kind, {})
            zord = 4 if is_front else 2
            ln, = ax.plot(x[seg], y[seg], z[seg], zorder=zord, **kw)
            ln._tag = tag
            # hard-order in 3D painter: back ≪ sphere ≪ front
            try: ln.set_sort_zpos(+1e9 if is_front else -1e9)
            except Exception: pass

# --- shuffle your datasets (unchanged) ---
indices = np.random.permutation(statesArrayChemical.shape[0]) 
statesArrayChemical = statesArrayChemical[indices] 
indices = np.random.permutation(statesArrayElectric.shape[0]) 
statesArrayElectric = statesArrayElectric[indices] 
indices = np.random.permutation(statesArrayImpBurn.shape[0])
statesArrayImpBurn = statesArrayImpBurn[indices] 
indices = np.random.permutation(statesArrayNoThrust.shape[0])
statesArrayNoThrust = statesArrayNoThrust[indices]

# --- figure ---
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = plot_earth_fast(ax, R=R_E, u_res=48, v_res=24)
ax.set_proj_type('ortho')  # cheaper + avoids perspective ambiguity

# collect random trajectories once
orbits = []
for _ in range(randPlots):
    iC = np.random.randint(statesArrayChemical.shape[0])
    iE = np.random.randint(statesArrayElectric.shape[0])
    iI = np.random.randint(statesArrayImpBurn.shape[0])
    iN = np.random.randint(statesArrayNoThrust.shape[0])

    orbits.append(("Chemical", (statesArrayChemical[iC,:,0],
                                statesArrayChemical[iC,:,1],
                                statesArrayChemical[iC,:,2])))
    orbits.append(("Electric", (statesArrayElectric[iE,:,0],
                                statesArrayElectric[iE,:,1],
                                statesArrayElectric[iE,:,2])))
    orbits.append(("Impulsive", (statesArrayImpBurn[iI,:,0],
                                 statesArrayImpBurn[iI,:,1],
                                 statesArrayImpBurn[iI,:,2])))
    orbits.append(("NoThrust", (statesArrayNoThrust[iN,:,0],
                                statesArrayNoThrust[iN,:,1],
                                statesArrayNoThrust[iN,:,2])))

# styles
front_kw = {
    "Chemical": {"color": "C0", "lw": 1.8},
    "Electric": {"color": "C1", "lw": 1.8},
    "Impulsive": {"color": "C2", "lw": 1.8},
    "NoThrust": {"color": "C3", "lw": 1.8},
}
# slightly faded for the hidden half (still visible through Earth)
back_kw = {
    "Chemical": {"color": "C0", "lw": 1.2, "alpha": 0.5},
    "Electric": {"color": "C1", "lw": 1.2, "alpha": 0.5},
    "Impulsive": {"color": "C2", "lw": 1.2, "alpha": 0.5},
    "NoThrust": {"color": "C3", "lw": 1.2, "alpha": 0.5},
}

# initial draw (order: back lines, sphere, front lines) achieved via sort_zpos
draw_orbits_translucent(ax, orbits, R=R_E, front_kw=front_kw, back_kw=back_kw)

# axes, limits, labels
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)')
ax.set_title(f'Trajectories of {randPlots*4} Earth Orbiters')
ax.set_box_aspect((1, 1, 1))
max_r = max([np.sqrt(np.nanmax(x*x + y*y + z*z)) for _, (x, y, z) in orbits] + [R_E])
lim = 1.05 * max_r
ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
plt.tight_layout()

# legend (manual)
lines = [Line2D([0],[0], color=c, linewidth=3, linestyle='-')
         for c in ['C0','C1','C2','C3']]
labels = ['Chemical Thrust','Electrical Thrust','Impulsive Thrust','No Thrust']
ax.legend(lines, labels, loc='upper left')

# redraw orbits when camera changes
def _redraw(_evt=None):
    draw_orbits_translucent(ax, orbits, R=R_E, front_kw=front_kw, back_kw=back_kw)
    fig.canvas.draw_idle()
fig.canvas.mpl_connect('button_release_event', _redraw)
fig.canvas.mpl_connect('key_release_event', _redraw)

plt.show()