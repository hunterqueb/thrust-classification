import numpy as np
import matplotlib.pyplot as plt

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

a = np.load(f"{dataLoc}/OEArrayChemical.npz")
statesArrayChemical = a['OEArrayChemical']

a = np.load(f"{dataLoc}/OEArrayElectric.npz")
statesArrayElectric = a['OEArrayElectric']

a = np.load(f"{dataLoc}/OEArrayImpBurn.npz")
statesArrayImpBurn = a['OEArrayImpBurn']

a = np.load(f"{dataLoc}/OEArrayNoThrust.npz")
statesArrayNoThrust = a['OEArrayNoThrust']


if norm:
    R = 6378.1363 # km
    statesArrayChemical[:,:,0] = statesArrayChemical[:,:,0] / R
    statesArrayElectric[:,:,0] = statesArrayElectric[:,:,0] / R
    statesArrayImpBurn[:,:,0] = statesArrayImpBurn[:,:,0] / R
    statesArrayNoThrust[:,:,0] = statesArrayNoThrust[:,:,0] / R

indices = np.random.permutation(statesArrayChemical.shape[0]) 
statesArrayChemical = statesArrayChemical[indices] 
indices = np.random.permutation(statesArrayElectric.shape[0]) 
statesArrayElectric = statesArrayElectric[indices] 
indices = np.random.permutation(statesArrayImpBurn.shape[0])
statesArrayImpBurn = statesArrayImpBurn[indices] 
indices = np.random.permutation(statesArrayNoThrust.shape[0])
statesArrayNoThrust = statesArrayNoThrust[indices]

t = np.linspace(0,numMinProp*dt,len(statesArrayChemical[0,:,0]))

plt.figure()
plt.plot(t, statesArrayChemical[0,:,0], label='Chemical')
plt.plot(t, statesArrayElectric[0,:,0], label='Electric')
plt.plot(t, statesArrayImpBurn[0,:,0], label='Impulsive')
plt.plot(t, statesArrayNoThrust[0,:,0], label='No Thrust')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.title("Semi-Major Axis")


plt.figure()

plt.plot(t, statesArrayChemical[0,:,1], label='Chemical')
plt.plot(t, statesArrayElectric[0,:,1], label='Electric')
plt.plot(t, statesArrayImpBurn[0,:,1], label='Impulsive')
plt.plot(t, statesArrayNoThrust[0,:,1], label='No Thrust')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.title("Eccentricity")

plt.figure()

plt.plot(t, statesArrayChemical[0,:,2], label='Chemical')
plt.plot(t, statesArrayElectric[0,:,2], label='Electric')
plt.plot(t, statesArrayImpBurn[0,:,2], label='Impulsive')
plt.plot(t, statesArrayNoThrust[0,:,2], label='No Thrust')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.title("Inclination")

plt.figure()
plt.plot(t, statesArrayChemical[0,:,3], label='Chemical')
plt.plot(t, statesArrayElectric[0,:,3], label='Electric')
plt.plot(t, statesArrayImpBurn[0,:,3], label='Impulsive')
plt.plot(t, statesArrayNoThrust[0,:,3], label='No Thrust')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.title("RAAN")

plt.figure()

plt.plot(t, statesArrayChemical[0,:,4], label='Chemical')
plt.plot(t, statesArrayElectric[0,:,4], label='Electric')
plt.plot(t, statesArrayImpBurn[0,:,4], label='Impulsive')
plt.plot(t, statesArrayNoThrust[0,:,4], label='No Thrust')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.title("Arg of Perigee")

plt.figure()

plt.plot(t, statesArrayChemical[0,:,5], label='Chemical')
plt.plot(t, statesArrayElectric[0,:,5], label='Electric')
plt.plot(t, statesArrayImpBurn[0,:,5], label='Impulsive')
plt.plot(t, statesArrayNoThrust[0,:,5], label='No Thrust')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.title("Mean Anomaly")

plt.figure()

plt.plot(t, statesArrayChemical[0,:,6], label='Chemical')
plt.plot(t, statesArrayElectric[0,:,6], label='Electric')
plt.plot(t, statesArrayImpBurn[0,:,6], label='Impulsive')
plt.plot(t, statesArrayNoThrust[0,:,6], label='No Thrust')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.title("Period")

# def plotDiffFromNoThrust(statesArray, label):
#     plt.figure()
#     plt.plot(t, statesArray[0,:,0]-statesArrayNoThrust[0,:,0], label=label+' X')
#     plt.plot(t, statesArray[0,:,1]-statesArrayNoThrust[0,:,1], label=label+' Y')
#     plt.plot(t, statesArray[0,:,2]-statesArrayNoThrust[0,:,2], label=label+' Z')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Position Difference from No Thrust (km)')
#     plt.title(f'Position Difference from No Thrust vs Time for {label} Thruster Profile')
#     plt.legend()
#     plt.grid()

#     plt.figure()
#     plt.plot(t, statesArray[0,:,3]-statesArrayNoThrust[0,:,3], label=label+' VX')
#     plt.plot(t, statesArray[0,:,4]-statesArrayNoThrust[0,:,4], label=label+' VY')
#     plt.plot(t, statesArray[0,:,5]-statesArrayNoThrust[0,:,5], label=label+' VZ')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Position Difference from No Thrust (km/s)')
#     plt.title(f'Position Difference from No Thrust vs Time for {label} Thruster Profile')
#     plt.legend()
#     plt.grid()


# plotDiffFromNoThrust(statesArrayChemical, 'Chemical')
# plotDiffFromNoThrust(statesArrayElectric, 'Electric')
# plotDiffFromNoThrust(statesArrayImpBurn, 'Impulsive')


from matplotlib.lines import Line2D
colors = ['C0', 'C1', 'C2', 'C3']
lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
labels = ['Chemical Thrust', 'Electrical Thrust', 'Impulsive Thrust', 'No Thrust']


plt.figure()
for j in range(randPlots):
    # take a random index
    i = np.random.randint(0, len(statesArrayChemical))
    plt.plot(t, statesArrayChemical[i,:,0], label='Chemical',color='C0')
    plt.plot(t, statesArrayElectric[i,:,0], label='Electric',color='C1')
    plt.plot(t, statesArrayImpBurn[i,:,0], label='Impulsive',color='C2')
    plt.plot(t, statesArrayNoThrust[i,:,0], label='No Thrust',color='C3')
    plt.grid()
    plt.legend(lines, labels)
    plt.xlabel('Time (s)')
    plt.title("Semi-Major Axis of "+str(randPlots*4)+" Earth Orbiters")

plt.figure()
for j in range(randPlots):
    i = np.random.randint(0, len(statesArrayChemical))

    plt.plot(t, statesArrayChemical[i,:,1], label='Chemical',color='C0')
    plt.plot(t, statesArrayElectric[i,:,1], label='Electric',color='C1')
    plt.plot(t, statesArrayImpBurn[i,:,1], label='Impulsive',color='C2')
    plt.plot(t, statesArrayNoThrust[i,:,1], label='No Thrust',color='C3')
    plt.grid()
    plt.legend(lines, labels)
    plt.xlabel('Time (s)')
    plt.title("Eccentricity of "+str(randPlots*4)+" Earth Orbiters")

plt.figure()
for j in range(randPlots):
    i = np.random.randint(0, len(statesArrayChemical))

    plt.plot(t, statesArrayChemical[i,:,2], label='Chemical',color='C0')
    plt.plot(t, statesArrayElectric[i,:,2], label='Electric',color='C1')
    plt.plot(t, statesArrayImpBurn[i,:,2], label='Impulsive',color='C2')
    plt.plot(t, statesArrayNoThrust[i,:,2], label='No Thrust',color='C3')
    plt.grid()
    plt.legend(lines, labels)
    plt.xlabel('Time (s)')
    plt.title("Inclination of "+str(randPlots*4)+" Earth Orbiters")

plt.figure()
for j in range(randPlots):
    i = np.random.randint(0, len(statesArrayChemical))

    plt.plot(t, statesArrayChemical[i,:,3], label='Chemical',color='C0')
    plt.plot(t, statesArrayElectric[i,:,3], label='Electric',color='C1')
    plt.plot(t, statesArrayImpBurn[i,:,3], label='Impulsive',color='C2')
    plt.plot(t, statesArrayNoThrust[i,:,3], label='No Thrust',color='C3')
    plt.grid()
    plt.legend(lines, labels)
    plt.xlabel('Time (s)')
    plt.title("RAAN of " +str(randPlots*4)+" Earth Orbiters")

plt.figure()
for j in range(randPlots):
    i = np.random.randint(0, len(statesArrayChemical))

    plt.plot(t, statesArrayChemical[i,:,4], label='Chemical',color='C0')
    plt.plot(t, statesArrayElectric[i,:,4], label='Electric',color='C1')
    plt.plot(t, statesArrayImpBurn[i,:,4], label='Impulsive',color='C2')
    plt.plot(t, statesArrayNoThrust[i,:,4], label='No Thrust',color='C3')
    plt.grid()
    plt.legend(lines, labels)
    plt.xlabel('Time (s)')
    plt.title("Arg of Perigee of "+str(randPlots*4)+" Earth Orbiters")

plt.figure()
for j in range(randPlots):
    i = np.random.randint(0, len(statesArrayChemical))

    plt.plot(t, statesArrayChemical[i,:,5], label='Chemical',color='C0')
    plt.plot(t, statesArrayElectric[i,:,5], label='Electric',color='C1')
    plt.plot(t, statesArrayImpBurn[i,:,5], label='Impulsive',color='C2')
    plt.plot(t, statesArrayNoThrust[i,:,5], label='No Thrust',color='C3')
    plt.grid()
    plt.legend(lines, labels)
    plt.xlabel('Time (s)')
    plt.title("Mean Anomaly of "+str(randPlots*4)+" Earth Orbiters")

plt.figure()
for j in range(randPlots):
    i = np.random.randint(0, len(statesArrayChemical))

    plt.plot(t, statesArrayChemical[i,:,6], label='Chemical',color='C0')
    plt.plot(t, statesArrayElectric[i,:,6], label='Electric',color='C1')
    plt.plot(t, statesArrayImpBurn[i,:,6], label='Impulsive',color='C2')
    plt.plot(t, statesArrayNoThrust[i,:,6], label='No Thrust',color='C3')
    plt.grid()
    plt.legend(lines, labels)
    plt.xlabel('Time (s)')
    plt.title("Period of "+str(randPlots*4)+" Earth Orbiters")



plt.show()