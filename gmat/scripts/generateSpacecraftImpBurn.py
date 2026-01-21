from load_gmat import *
import numpy as np  
from matplotlib import pyplot as plt

gmat.EchoLogFile()

problemDim = 6
numRandSys = 1
mu = 398600  # Earth’s mu in km^3/s^2
R = 6371 # radius of earth in km
numMinProp = 60 * 24 # take a step 60 times in an hour and for 24 hours
numMinProp = 80 # take a step 60 times in an hour and for 24 hours
dt = 60.0 # step every 60 secs
elapsed = 0.0

# -----------configuration preliminaries----------------------------

# Spacecraft
earthorb = gmat.Construct("Spacecraft", "EarthOrbiter") # create a spacecraft object named EarthOrbiter
earthorb.SetField("DateFormat", "UTCGregorian")
earthorb.SetField("Epoch", "20 Jul 2020 12:00:00.000") # set the epoch of the spacecraft

# Set the coordinate system and display state type
earthorb.SetField("CoordinateSystem", "EarthMJ2000Eq")
earthorb.SetField("DisplayStateType", "Keplerian")

# Spacecraft ballistic properties for the SRP and Drag models
earthorb.SetField("SRPArea", 2.5)
earthorb.SetField("Cr", 1.75)
earthorb.SetField("DragArea", 1.8)
earthorb.SetField("Cd", 2.1)
earthorb.SetField("DryMass", 80)

# Force model settings
fm = gmat.Construct("ForceModel", "FM")
fm.SetField("CentralBody", "Earth")

earthgrav = gmat.Construct("GravityField")
earthgrav.SetField("BodyName","Earth")
earthgrav.SetField("PotentialFile", 'JGM2.cof')
earthgrav.SetField("Degree",8)
earthgrav.SetField("Order",8)


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

rng = np.random.default_rng()
rng.random()



statesArrayImpBurn = np.zeros((numRandSys,numMinProp,problemDim))


tank = gmat.Construct("ChemicalTank", "impFuel") # create a chemical tank with the name "Fuel"
earthorb.SetField("Tanks", "impFuel") # set the tank for the spacecraft to use the "Fuel" object

gmat.Initialize()

burn = gmat.Construct("ImpulsiveBurn", "EarthOrbiterBurn")
# burn.SetField("Tank", "impFuel") # set the tank for the "Burn" to use the "Fuel" object
burn.SetField("CoordinateSystem", "Local")
burn.SetField("Element1", 10)
burn.SetField("Element2", 0)
burn.SetField("Element3", 0)
burn.SetSolarSystem(gmat.GetSolarSystem())
burn.SetSpacecraftToManeuver(earthorb)

gmat.Initialize()

# burn_maneuver = gmat.Construct("Maneuver", "BurnManeuver")
# burn_maneuver.SetField("Burn", "EarthOrbiterBurn")
# burn_maneuver.SetField("Spacecraft", "EarthOrbiter")


# gmat.Initialize()


# burn.Help()
# burn_maneuver.Help()

for i in range(numRandSys):
    earthorb.SetField("SMA", 9000) # km
    earthorb.SetField("ECC", 0.05)
    earthorb.SetField("INC", 10) # deg
    earthorb.SetField("RAAN", 0) # deg
    earthorb.SetField("AOP", 0) # deg
    earthorb.SetField("TA", 0) # deg

    # Perform initializations
    gmat.Initialize()

    pdprop.AddPropObject(earthorb)

    gmat.Initialize()

    pdprop.PrepareInternals()
    gator = pdprop.GetPropagator()

    for j in range(numMinProp):

        if j == 40:
            propS = gmat.Command("Propagate", "PDProp(EarthOrbiter) {EarthOrbiter.ElapsedSecs = 60.0}") 
            maneuver = gmat.Command("Maneuver", "EarthOrbiterBurn(EarthOrbiter)")  
            gmat.Initialize()
            
            status = gmat.Execute()
            print(status)
            # after the burn, we can check the state of the spacecraft
            print("State after burn:")
            # earthorb.UpdateSpaceObject()
            print(earthorb.GetState().GetState())
            statesArrayImpBurn[i,j,:] = earthorb.GetState().GetState()

            gmat.Initialize()
            pdprop.PrepareInternals()
            gator = pdprop.GetPropagator()
            # gator.UpdateSpaceObject()


        else:
            gator.Step(dt)
            elapsed = elapsed + dt
            state = gator.GetState()
            statesArrayImpBurn[i,j,:] = state[0:6]
            gator.UpdateSpaceObject()

    t = np.linspace(0,numMinProp*dt,len(statesArrayImpBurn[0,:,0]))


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(statesArrayImpBurn[0,:,0],statesArrayImpBurn[0,:,1],statesArrayImpBurn[0,:,2],label='Impulsive')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('3D Trajectory of Earth Orbiter')
ax.legend()
ax.axis('equal')

plt.figure()
plt.plot(t, statesArrayImpBurn[0,:,3], label='X Velocity')
plt.plot(t, statesArrayImpBurn[0,:,4], label='Y Velocity')
plt.plot(t, statesArrayImpBurn[0,:,5], label='Z Velocity')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Velocity (km/s)')
plt.show()
