from load_gmat import *
import numpy as np  
from matplotlib import pyplot as plt
import math

# -----------configuration preliminaries----------------------------

mu = 398600

# Spacecraft
earthorb = gmat.Construct("Spacecraft", "EarthOrbiter") # create a spacecraft object named EarthOrbiter
earthorb.SetField("DateFormat", "UTCGregorian")
earthorb.SetField("Epoch", "20 Jul 2020 12:00:00.000") # set the epoch of the spacecraft

# Set the coordinate system and display state type
earthorb.SetField("CoordinateSystem", "EarthMJ2000Eq")
earthorb.SetField("DisplayStateType", "Keplerian")


# Initial Condition of the orbiting spacecraft -- two examples provided: OE and Cartesian coordinates
## Orbital Elements
# earthorb.SetField("SMA", 7000) # km
# earthorb.SetField("ECC", 0.05)
# earthorb.SetField("INC", 10) # deg
# earthorb.SetField("RAAN", 0) # deg
# earthorb.SetField("AOP", 0) # deg
# earthorb.SetField("TA", 0) # deg

## Cartesian Coordinates
earthorb.SetField("X",6650.0) # km
earthorb.SetField("Y",0.0) # km
earthorb.SetField("Z",0.0) # km
earthorb.SetField("VX",0.0) # km/s
earthorb.SetField("VY",7.812754425429622) # km/s
earthorb.SetField("VZ",1.3775993988527033) # km/s

# Spacecraft ballistic properties for the SRP and Drag models
earthorb.SetField("SRPArea", 2.5)
earthorb.SetField("Cr", 1.75)
earthorb.SetField("DragArea", 1.8)
earthorb.SetField("Cd", 2.1)
earthorb.SetField("DryMass", 80)

# Force model settings
fm = gmat.Construct("ForceModel", "FM")
fm.SetField("CentralBody", "Earth")

# A Full High-Fidelity 360x360 gravity field (incredibly slow)
earthgrav = gmat.Construct("GravityField")
earthgrav.SetField("BodyName","Earth")
earthgrav.SetField("PotentialFile", 'EGM96.cof')
earthgrav.SetField("Degree",360)
earthgrav.SetField("Order",360)

# A faster 8x8 degree and order
# earthgrav.SetField("PotentialFile", 'JGM2.cof')
# earthgrav.SetField("Degree",8)
# earthgrav.SetField("Order",8)


# Drag using Jacchia-Roberts
jrdrag = gmat.Construct("DragForce", "JRDrag")
jrdrag.SetField("AtmosphereModel","JacchiaRoberts")

# Build and set the atmosphere for the model
atmos = gmat.Construct("JacchiaRoberts", "Atmos")
jrdrag.SetReference(atmos)

# Construct Solar Radiation Pressure model
srp = gmat.Construct("SolarRadiationPressure", "SRP")

# Add forces into the ODEModel container
fm.AddForce(earthgrav)
fm.AddForce(jrdrag)
fm.AddForce(srp)

# Initialize propagator object
pdprop = gmat.Construct("Propagator","PDProp")

# Create and assign a numerical integrator for use in the propagation
gator = gmat.Construct("RungeKutta89", "Gator")
pdprop.SetReference(gator)

# Assign the force model contructed above
pdprop.SetReference(fm)

# Set some of the fields for the integration
pdprop.SetField("InitialStepSize", 60.0)
pdprop.SetField("Accuracy", 1.0e-12)
pdprop.SetField("MinStep", 0.0)

# construct the chemical thruster and tank
tank = gmat.Construct("ChemicalTank", "Fuel") # create a chemical tank with the name "Fuel"
tank.SetField("FuelMass", 200.0)
thruster = gmat.Construct("ChemicalThruster", "Thruster") # create a chemical thruster with the name "Thruster"
thruster.SetField("Tank", "Fuel") # set the tank for the "Thruster" to use the "Fuel" object
thruster.SetField("DecrementMass", True)
thruster.SetField("C1",100) # sets the first thrust coefficent. by default, chemical thrusters are set to a constant force output of 10 N and a 300 Ns impulse governed by a complex polynomial. See https://documentation.help/gmat/Thruster.html for specifics
earthorb.SetField("Tanks", "Fuel") # set possible tanks for the "Thruster" to use the "Fuel" object
earthorb.SetField("Thrusters", "Thruster") # set possible thrusters to use the "Thruster" object

# Perform initializations
gmat.Initialize()

# Setup the spacecraft that is propagated
pdprop.AddPropObject(earthorb)
pdprop.PrepareInternals()
# Refresh the 'gator reference
gator = pdprop.GetPropagator()

# -------------Propagation of the spacecraft------------------------
problemDim = 6
numStepsBB = 10
numStepsAB = 30

stateArrayIC = np.zeros((1,problemDim))
stateArrayBB = np.zeros((numStepsBB,problemDim))
stateArrayAB = np.zeros((numStepsAB,problemDim))
stateArrayNB = np.zeros((numStepsAB,problemDim))

# Build the burn model used for the burn
burn = gmat.Construct("FiniteBurn", "TheBurn")
burn.SetField("Thrusters", "Thruster")
burn.SetSolarSystem(gmat.GetSolarSystem())
burn.SetSpacecraftToManeuver(earthorb)

# construct the burn force model
def setThrust(s, b):
    bf = gmat.FiniteThrust("Thrust")
    bf.SetRefObjectName(gmat.SPACECRAFT, s.GetName())
    bf.SetReference(b)
    gmat.ConfigManager.Instance().AddPhysicalModel(bf)
    return bf
burnForce = setThrust(earthorb, burn)
gmat.Initialize()

pdprop.AddPropObject(earthorb)
pdprop.PrepareInternals()
# Access the thruster cloned onto the spacecraft
theThruster = earthorb.GetRefObject(gmat.THRUSTER, "Thruster")

elapsed = 0.0
dt = 60.0
gator = pdprop.GetPropagator()
stateArrayIC[0,:] = gator.GetState()

# Step for 10 minutes
print("\n Propagate 10 minutes without a burn\n-------------------------------------------------\n")
print("Firing: ", theThruster.GetField("IsFiring"), "\n")
state = gator.GetState()
r = math.sqrt(state[0]**2 + state[1]**2 + state[2]**2)
vsq = state[3]**2 + state[4]**2 + state[5]**2
sma = r * mu / (2.0 * mu - vsq * r)
print(elapsed, sma, state, earthorb.GetField("TotalMass"))
for i in range(numStepsBB):
    gator.Step(dt)
    elapsed = elapsed + dt
    state = gator.GetState()
    stateArrayBB[i,:] = state
    r = math.sqrt(state[0]**2 + state[1]**2 + state[2]**2)
    vsq = state[3]**2 + state[4]**2 + state[5]**2
    sma = r * mu / (2.0 * mu - vsq * r)
    print(elapsed, sma, state, earthorb.GetField("TotalMass"))
    gator.UpdateSpaceObject()
elapsedBB = elapsed
print("\n Propagate 30 minutes with a burn\n-------------------------------------------------\n")
# -----------------------------
# Finite Burn Specific Settings
# -----------------------------
# Turn on the thruster
theThruster.SetField("IsFiring", True)
earthorb.IsManeuvering(True)
burn.SetSpacecraftToManeuver(earthorb)
# Add the thrust to the force model
pdprop.AddForce(burnForce)
psm = pdprop.GetPropStateManager()
psm.SetProperty("MassFlow")
# -----------------------------

pdprop.PrepareInternals()
gator = pdprop.GetPropagator()
print("Firing: ", theThruster.GetField("IsFiring"), "\n")
# Now propagate through the burn
for i in range(numStepsAB):
    gator.Step(dt)
    elapsed = elapsed + dt
    state = gator.GetState()
    stateArrayAB[i,:] = state[0:6]
    r = math.sqrt(state[0]**2 + state[1]**2 + state[2]**2)
    vsq = state[3]**2 + state[4]**2 + state[5]**2
    sma = r * mu / (2.0 * mu - vsq * r)
    print(elapsed, sma, state, earthorb.GetField("TotalMass"))
    gator.UpdateSpaceObject()

# Turn off the thruster and remove the burn force
fm = pdprop.GetODEModel()
fm.DeleteForce(burnForce)
theThruster.SetField("IsFiring", False)
earthorb.IsManeuvering(False)
pdprop.PrepareInternals()
gator = pdprop.GetPropagator()

# reintialize spacecraft to final state before the burn and reprop without the thruster
earthorb.SetField("X",stateArrayBB[-1,0]) # km
earthorb.SetField("Y",stateArrayBB[-1,1]) # km
earthorb.SetField("Z",stateArrayBB[-1,2]) # km
earthorb.SetField("VX",stateArrayBB[-1,3]) # km/s
earthorb.SetField("VY",stateArrayBB[-1,4]) # km/s
earthorb.SetField("VZ",stateArrayBB[-1,5]) # km/s

elapsed = elapsedBB
pdprop.PrepareInternals()
gator = pdprop.GetPropagator()
print("Firing: ", theThruster.GetField("IsFiring"), "\n")
# Now propagate through the burn
for i in range(numStepsAB):
    gator.Step(dt)
    elapsed = elapsed + dt
    state = gator.GetState()
    stateArrayNB[i,:] = state[0:6]
    r = math.sqrt(state[0]**2 + state[1]**2 + state[2]**2)
    vsq = state[3]**2 + state[4]**2 + state[5]**2
    sma = r * mu / (2.0 * mu - vsq * r)
    print(elapsed, sma, state, earthorb.GetField("TotalMass"))
    gator.UpdateSpaceObject()


# Plot
stateArrayBurn = np.concatenate((stateArrayIC,stateArrayBB,stateArrayAB))
stateArrayNoBurn = np.concatenate((stateArrayIC,stateArrayBB,stateArrayNB))
t = np.linspace(0,elapsed,int(elapsed/dt)+1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(stateArrayBurn[:,0],stateArrayBurn[:,1],stateArrayBurn[:,2],label='Manuever')
ax.plot(stateArrayBurn[:,0],stateArrayBurn[:,1],stateArrayBurn[:,2],label='No Manuever')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('3D Trajectory of Earth Orbiter')
ax.legend()
ax.axis('equal')

plt.figure()
plt.plot(t,stateArrayBurn[:,0],label='Burn: x')
plt.plot(t,stateArrayNoBurn[:,0],label='No Burn: x')
plt.plot(t,stateArrayBurn[:,1],label='Burn: y')
plt.plot(t,stateArrayNoBurn[:,1],label='No Burn: y')
plt.plot(t,stateArrayBurn[:,2],label='Burn: z')
plt.plot(t,stateArrayNoBurn[:,2],label='No Burn: z')
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Distance (km)")
plt.show()