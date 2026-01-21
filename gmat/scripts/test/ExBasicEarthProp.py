from load_gmat import *
import numpy as np  
from matplotlib import pyplot as plt


# -----------configuration preliminaries----------------------------

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

# Perform initializations
#  top level initializations
gmat.Initialize()
# Setup the spacecraft that is propagated
pdprop.AddPropObject(earthorb)
pdprop.PrepareInternals()
# Refresh the 'gator reference
gator = pdprop.GetPropagator()

# -------------Propagation of the spacecraft------------------------

problemDim = 6
numSteps = 144

stateArray = np.zeros((numSteps,problemDim))

# Take a 600 second steps for 1 day
time = 0.0
step = 600.0
print(time, " sec, state = ", gator.GetState())

stateArray[0,:] = gator.GetState()

# Propagate for 1 day (via 144 10-minute steps)
for x in range(numSteps-1):
    gator.Step(step)
    time = time + step
    # print(time, " sec, state = ", gator.GetState())
    stateArray[x+1,:] = gator.GetState()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(stateArray[:,0],stateArray[:,1],stateArray[:,2])
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('3D Trajectory of Earth Orbiter')

plt.show()

# -------------Propagation of the spacecraft------------------------

problemDim = 6
numSteps = 144

stateArray = np.zeros((numSteps,problemDim))

# Take a 600 second steps for 1 day
time = 0.0
step = 600.0
print(time, " sec, state = ", gator.GetState())

stateArray[0,:] = gator.GetState()

# Propagate for 1 day (via 144 10-minute steps)
for x in range(numSteps-1):
    gator.Step(step)
    time = time + step
    # print(time, " sec, state = ", gator.GetState())
    stateArray[x+1,:] = gator.GetState()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(stateArray[:,0],stateArray[:,1],stateArray[:,2])
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('3D Trajectory of Earth Orbiter')

plt.show()