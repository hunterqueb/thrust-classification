from load_gmat import *
import numpy as np  
from matplotlib import pyplot as plt
import argparse

problemDim = 6
mu = 398600  # Earth’s mu in km^3/s^2
R = 6378 # radius of earth in km
dt = 60.0 # step every 60 secs
elapsed = 0.0


parser = argparse.ArgumentParser(description='GMAT Dataset Generation for Different Thruster Types')
parser.add_argument('--deltaV', type=float, default=0.05, help='Delta V for impulsive burn in km/s')
parser.add_argument('--numRandSys', type=int, default=1, help='Number of random systems to generate')
parser.add_argument('--numMinProp', type=int, default=10, help='Number of minutes to propagate')
parser.add_argument('--chemThrust',type=float, default=10, help='Chemical thrust coefficent (N)')
parser.add_argument('--elecThrust',type=float, default=0.1, help='Electric thrust coefficent (N)')
parser.add_argument('--no-plot', dest="plotOn", action='store_false', help='Plot the results')
parser.set_defaults(plotOn=True)
parser.add_argument('--log', dest='echoLogFile', action='store_true', help='Log the GMAT output to console')
parser.set_defaults(echoLogFile=False)
parser.add_argument('--no-save', dest='saveOn', action='store_false', help='Save the results to file')
parser.set_defaults(saveOn=True)
parser.add_argument('--slurm', dest='slurm', action='store_true', help='Run in slurm mode')
parser.set_defaults(slurm=False)
parser.add_argument('--propType', type=str, default='chem', help='Type of propagation to perform: chem, elec, imp, none')
parser.add_argument("--lowerAlt", type=float, default=35786, help="Lower altitude bound in km")
parser.add_argument("--upperAlt", type=float, default=35800, help="Upper altitude bound in km")
parser.add_argument("--folder", type=str, default="", help="Folder to save the data in, if applicable")

args = parser.parse_args()

deltaV = args.deltaV
numRandSys = args.numRandSys
numMinProp = args.numMinProp
chemThrust = args.chemThrust
elecThrust = args.elecThrust
plotOn = args.plotOn
echoLogFile = args.echoLogFile
saveOn = args.saveOn
slurm = args.slurm
propType = args.propType
lowerAlt = args.lowerAlt
upperAlt = args.upperAlt
folder = args.folder
if len(folder) > 0 and folder[-1] == '/':
    folder = folder[:-1]  # Remove trailing slash if present

if echoLogFile:
    gmat.EchoLogFile()


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

# gravity field
earthgrav = gmat.Construct("GravityField")
earthgrav.SetField("BodyName","Earth")
earthgrav.SetField("PotentialFile", 'JGM2.cof')
earthgrav.SetField("Degree",70)
earthgrav.SetField("Order",70)


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

rng = np.random.default_rng(123)
rng.random()

SMA = np.zeros(numRandSys)
ECC = np.zeros(numRandSys)
INC = np.zeros(numRandSys)
RAAN = np.zeros(numRandSys)
AOP = np.zeros(numRandSys)
TA = np.zeros(numRandSys)

SRPAreas = np.zeros(numRandSys)
Crs = np.zeros(numRandSys)
DragAreas = np.zeros(numRandSys)
Cds = np.zeros(numRandSys)
DryMasses = np.zeros(numRandSys)

for i in range(numRandSys):
    # SMA[i] = rng.uniform(26335, 26500)  # km, typical HEO altitudes
    # ECC[i] = 0.05 * rng.random() + 0.7
    ECC[i] = 0.01 * rng.random()
    
    r_p_min = R + lowerAlt
    r_p_max = R + upperAlt
    a_min = r_p_min/(1 - ECC[i])
    a_max = r_p_max/(1 - ECC[i])

    SMA[i] = rng.uniform(a_min, a_max)
    INC[i] = 180 * rng.random() # deg
    RAAN[i] = 180 * rng.random() # deg
    AOP[i] = 180 * rng.random() # deg
    nu = 180
    TA[i] = rng.uniform(nu-180, nu+180) # deg

    SRPAreas[i] = rng.uniform(1.0, 5.0)
    Crs[i] = rng.uniform(1.0, 2.0)
    DragAreas[i] = rng.uniform(1.0, 3.0)
    Cds[i] = rng.uniform(1.0, 3.0)
    DryMasses[i] = rng.uniform(50.0, 150.0)

    print(f"System {i+1}/{numRandSys}: a={SMA[i]:.2f} km, e={ECC[i]:.4f}, i={INC[i]:.2f} deg, RAAN={RAAN[i]:.2f} deg, AOP={AOP[i]:.2f} deg, TA={TA[i]:.2f} deg")
    print(f"               SRPArea={SRPAreas[i]:.2f} m^2, Cr={Crs[i]:.2f}, DragArea={DragAreas[i]:.2f} m^2, Cd={Cds[i]:.2f}, DryMass={DryMasses[i]:.2f} kg")
# construct the burn force model
def setThrust(s, b):
    bf = gmat.FiniteThrust("Thrust")
    bf.SetRefObjectName(gmat.SPACECRAFT, s.GetName())
    bf.SetReference(b)
    gmat.ConfigManager.Instance().AddPhysicalModel(bf)
    return bf


if propType == 'chem':
    statesArrayChemical = np.zeros((numRandSys,numMinProp,problemDim))

    tank = gmat.Construct("ChemicalTank", "Fuel") # create a chemical tank with the name "Fuel"
    thruster = gmat.Construct("ChemicalThruster", "Thruster") # create a chemical thruster with the name "Thruster"
    thruster.SetField("C1", chemThrust) # set the thrust coefficient for the "Thruster" to use the chemThrust value
    thruster.SetField("DecrementMass", False)
    thruster.SetField("Tank", "Fuel") # set the tank for the "Thruster" to use the "Fuel" object
    earthorb.SetField("Tanks", "Fuel") # set possible tanks for the "Thruster" to use the "Fuel" object
    earthorb.SetField("Thrusters", "Thruster") # set possible thrusters to use the "Thruster" object

    gmat.Initialize()



    burn = gmat.Construct("FiniteBurn", "TheBurn")
    burn.SetField("Thrusters", "Thruster")
    burn.SetSolarSystem(gmat.GetSolarSystem())
    burn.SetSpacecraftToManeuver(earthorb)

    burnForce = setThrust(earthorb, burn)

    gmat.Initialize()

    print("Generating Chemical Thruster Data...")
    for i in range(numRandSys):
        earthorb.SetField("SRPArea", SRPAreas[i])
        earthorb.SetField("Cr", Crs[i])
        earthorb.SetField("DragArea", DragAreas[i])
        earthorb.SetField("Cd", Cds[i])
        earthorb.SetField("DryMass", DryMasses[i])

        earthorb.SetField("SMA", SMA[i]) # km
        earthorb.SetField("ECC", ECC[i])
        earthorb.SetField("INC", INC[i]) # deg
        earthorb.SetField("RAAN", RAAN[i]) # deg
        earthorb.SetField("AOP", AOP[i]) # deg
        earthorb.SetField("TA", TA[i]) # deg

        # Perform initializations
        gmat.Initialize()

        # Refresh the 'gator reference
        gator = pdprop.GetPropagator()

        gmat.Initialize()
        
        pdprop.AddPropObject(earthorb)
        pdprop.PrepareInternals()

        theThruster = earthorb.GetRefObject(gmat.THRUSTER, "Thruster")

        # -----------------------------
        # Finite Burn Specific Settings
        # -----------------------------
        # Turn on the thruster
        theThruster.SetField("IsFiring", True)
        earthorb.IsManeuvering(True)
        burn.SetSpacecraftToManeuver(earthorb)
        # # Add the thrust to the force model
        pdprop.AddForce(burnForce)
        psm = pdprop.GetPropStateManager()
        psm.SetProperty("MassFlow")
        # -----------------------------
        pdprop.PrepareInternals()
        gator = pdprop.GetPropagator()

        for j in range(numMinProp):
            gator.Step(dt)
            elapsed = elapsed + dt
            state = gator.GetState()
            statesArrayChemical[i,j,:] = state[0:6]
            gator.UpdateSpaceObject()

        fm = pdprop.GetODEModel()
        fm.DeleteForce(burnForce)
        # theThruster.SetField("IsFiring", False)
        earthorb.IsManeuvering(False)
        pdprop.PrepareInternals()
        gator = pdprop.GetPropagator()

        del(theThruster)
        del(gator)
        del(fm)
        del(psm)


    print("Chemical Thruster Data Generation Complete.")

    if saveOn:
        if slurm:
            saveDest = '~/pendulumRNN/gmat/data/classification/' + folder + "/"
        else:
            saveDest = 'gmat/data/classification/' + folder + "/"

        np.savez(saveDest+'statesArrayChemical.npz', statesArrayChemical=statesArrayChemical)

    OEArrayChemThrust = np.zeros((numRandSys,numMinProp,7))

    from qutils.orbital import ECI2OE
    for i in range(numRandSys):
        for j in range(numMinProp):
            OEArrayChemThrust[i,j,:] = ECI2OE(statesArrayChemical[i,j,0:3],statesArrayChemical[i,j,3:6])


    t = np.linspace(0,numMinProp*dt,len(statesArrayChemical[0,:,0]))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(statesArrayChemical[0,:,0],statesArrayChemical[0,:,1],statesArrayChemical[0,:,2],label='No Thrust')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('3D Trajectory of Earth Orbiter')
    ax.legend()
    ax.axis('equal')

    plt.figure()
    plt.plot(t, statesArrayChemical[0,:,0], label='No Thrust X')
    plt.plot(t, statesArrayChemical[0,:,1], label='No Thrust Y')
    plt.plot(t, statesArrayChemical[0,:,2], label='No Thrust Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (km)')
    plt.title('Position vs Time for Different Thruster Profiles')
    plt.legend()
    plt.grid()


    plt.figure()
    plt.plot(t, OEArrayChemThrust[0,:,0], label='No Thrust',color='C1')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("a")

    plt.figure()
    plt.plot(t, OEArrayChemThrust[0,:,1], label='No Thrust',color='C1')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("e")

    plt.figure()
    plt.plot(t, OEArrayChemThrust[0,:,2], label='No Thrust',color='C1')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("i")

    plt.figure()
    plt.plot(t, OEArrayChemThrust[0,:,3], label='No Thrust',color='C1')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("OMEGA")

    plt.figure()
    plt.plot(t, OEArrayChemThrust[0,:,4], label='No Thrust',color='C1')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("omega")

    plt.figure()
    plt.plot(t, OEArrayChemThrust[0,:,5], label='No Thrust',color='C1')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("nu")

    plt.figure()
    plt.plot(t, OEArrayChemThrust[0,:,6], label='No Thrust',color='C1')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("Period")


elif propType == 'elec':
    statesArrayElectric = np.zeros((numRandSys,numMinProp,problemDim))


    ETank = gmat.Construct("ElectricTank", "EFuel") # create an electric tank with the name "Fuel"
    EThruster = gmat.Construct("ElectricThruster", "EThruster") # create an electric thruster with the name "Thruster"
    powerSystem = gmat.Construct("SolarPowerSystem", "EPS") # create a power system with the name "EPS"
    # EThruster.SetField("ThrustModel", "ThrustMassPolynomial") # set the thrust coefficient for the "EThruster" to use the elecThrust value
    EThruster.SetField("ThrustModel", "ConstantThrustAndIsp") # set the thrust coefficient for the "EThruster" to use the elecThrust value
    EThruster.SetField("DecrementMass", True)
    EThruster.SetField("Isp", 2800) # set the thrust coefficient for the "EThruster" to use the elecThrust value
    EThruster.SetField("ConstantThrust", 0.08) # scale the thrust 
    EThruster.SetField("Tank", "EFuel") # set the tank for the "EThruster" to use the "EFuel" object
    EThruster.SetField("DutyCycle", 1)
        
    earthorb.SetField("Tanks", "EFuel") # set possible tanks for the "EThruster" to use the "EFuel" object
    earthorb.SetField("Thrusters", "EThruster") # set possible thrusters to use the "EThruster" object
    earthorb.SetField("PowerSystem", "EPS") # set the power system of the spacecraft to use the "EPS" object
    gmat.Initialize()

    # construct the burn force model (the burn force model is the same for both chemical and electric thrusters)
    burn = gmat.Construct("FiniteBurn", "TheEBurn")
    burn.SetField("Thrusters", "EThruster")
    burn.SetSolarSystem(gmat.GetSolarSystem())
    burn.SetSpacecraftToManeuver(earthorb)

    burnForce = setThrust(earthorb, burn)


    gmat.Initialize()

    print("Generating Electric Thruster Data...")
    for i in range(numRandSys):
        earthorb.SetField("SRPArea", SRPAreas[i])
        earthorb.SetField("Cr", Crs[i])
        earthorb.SetField("DragArea", DragAreas[i])
        earthorb.SetField("Cd", Cds[i])
        earthorb.SetField("DryMass", DryMasses[i])

        earthorb.SetField("SMA", SMA[i]) # km
        earthorb.SetField("ECC", ECC[i])
        earthorb.SetField("INC", INC[i]) # deg
        earthorb.SetField("RAAN", RAAN[i]) # deg
        earthorb.SetField("AOP", AOP[i]) # deg
        earthorb.SetField("TA", TA[i]) # deg

        # Perform initializations
        gmat.Initialize()

        # Refresh the 'gator reference
        gator = pdprop.GetPropagator()

        gmat.Initialize()
        
        pdprop.AddPropObject(earthorb)
        pdprop.PrepareInternals()

        theThruster = earthorb.GetRefObject(gmat.THRUSTER, "EThruster")

        # -----------------------------
        # Finite Burn Specific Settings
        # -----------------------------
        # Turn on the thruster
        theThruster.SetField("IsFiring", True) 
        earthorb.IsManeuvering(True)
        burn.SetSpacecraftToManeuver(earthorb)
        # # Add the thrust to the force model

        pdprop.AddForce(burnForce)
        psm = pdprop.GetPropStateManager()
        psm.SetProperty("MassFlow")
        # -----------------------------
        pdprop.PrepareInternals()
        gator = pdprop.GetPropagator()

        for j in range(numMinProp):
            gator.Step(dt)
            elapsed = elapsed + dt
            state = gator.GetState()
            statesArrayElectric[i,j,:] = state[0:6]
            gator.UpdateSpaceObject()

        fm = pdprop.GetODEModel()
        fm.DeleteForce(burnForce)
        # theThruster.SetField("IsFiring", False)
        earthorb.IsManeuvering(False)

        pdprop.PrepareInternals()
        # gator = pdprop.GetPropagator()
        # del(theThruster)
        # del(gator)
        # del(fm)
        # del(psm)
    print("Electric Thruster Data Generation Complete.")

    if saveOn:
        if slurm:
            saveDest = '~/pendulumRNN/gmat/data/classification/' + folder + "/"
        else:
            saveDest = 'gmat/data/classification/' + folder + "/"

        np.savez(saveDest+'statesArrayElectric.npz', statesArrayElectric=statesArrayElectric)
    
    OEArrayElecThrust = np.zeros((numRandSys,numMinProp,7))

    from qutils.orbital import ECI2OE
    for i in range(numRandSys):
        for j in range(numMinProp):
            OEArrayElecThrust[i,j,:] = ECI2OE(statesArrayElectric[i,j,0:3],statesArrayElectric[i,j,3:6])


    t = np.linspace(0,numMinProp*dt,len(statesArrayElectric[0,:,0]))

    testNum = 0

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(statesArrayElectric[testNum,:,0],statesArrayElectric[testNum,:,1],statesArrayElectric[testNum,:,2],label='No Thrust')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('3D Trajectory of Earth Orbiter')
    ax.legend()
    ax.axis('equal')

    plt.figure()
    plt.plot(t, statesArrayElectric[testNum,:,0], label='Elec Thrust X')
    plt.plot(t, statesArrayElectric[testNum,:,1], label='Elec Thrust Y')
    plt.plot(t, statesArrayElectric[testNum,:,2], label='Elec Thrust Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (km)')
    plt.title('Position vs Time for Different Thruster Profiles')
    plt.legend()
    plt.grid()


    plt.figure()
    plt.plot(t, OEArrayElecThrust[testNum,:,0], label='Elec Thrust',color='C2')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("a")

    plt.figure()
    plt.plot(t, OEArrayElecThrust[testNum,:,1], label='Elec Thrust',color='C2')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("e")

    plt.figure()
    plt.plot(t, OEArrayElecThrust[testNum,:,2], label='Elec Thrust',color='C2')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("i")

    plt.figure()
    plt.plot(t, OEArrayElecThrust[testNum,:,3], label='Elec Thrust',color='C2')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("OMEGA")

    plt.figure()
    plt.plot(t, OEArrayElecThrust[testNum,:,4], label='Elec Thrust',color='C2')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("omega")

    plt.figure()
    plt.plot(t, OEArrayElecThrust[testNum,:,5], label='Elec Thrust',color='C2')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("nu")

    plt.figure()
    plt.plot(t, OEArrayElecThrust[testNum,:,6], label='Elec Thrust',color='C2')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("Period")


elif propType == 'imp':
    statesArrayImpBurn = np.zeros((numRandSys,numMinProp,problemDim))

    print("Generating Impulsive Burn Data...")
    for i in range(numRandSys):
        earthorb.SetField("SMA", SMA[i]) # km
        earthorb.SetField("ECC", ECC[i])
        earthorb.SetField("INC", INC[i]) # deg
        earthorb.SetField("RAAN", RAAN[i]) # deg
        earthorb.SetField("AOP", AOP[i]) # deg
        earthorb.SetField("TA", TA[i]) # deg

        gmat.Initialize()

        # Refresh the 'gator reference
        gator = pdprop.GetPropagator()

        gmat.Initialize()
        
        pdprop.AddPropObject(earthorb)
        pdprop.PrepareInternals()

        randImpTime = np.random.randint(1, numMinProp-1) # randomly select a time to apply the impulsive burn
        for j in range(numMinProp):
            if j == randImpTime:  
                craftVel = state[3:6]
                craftVel = craftVel/np.linalg.norm(craftVel) * deltaV + craftVel

                earthorb.SetField("VX", craftVel[0])
                earthorb.SetField("VY", craftVel[1])
                earthorb.SetField("VZ", craftVel[2])

                pdprop.PrepareInternals()

            try:
                gator.Step(dt)
            except Exception as e:
                print(f"Error during propagation at i={i}, j={j}: {e}")
                print("Craft State at Error:", state)
                break
            elapsed = elapsed + dt
            state = gator.GetState()
            statesArrayImpBurn[i,j,:] = state[0:6]
            gator.UpdateSpaceObject()

        del(gator)

    print("Impulsive Burn Data Generation Complete.")

    if saveOn:
        if slurm:
            saveDest = '~/pendulumRNN/gmat/data/classification/'
        else:
            saveDest = 'gmat/data/classification/'

        np.savez(saveDest+'statesArrayImpBurn.npz', statesArrayImpBurn=statesArrayImpBurn)

    OEArrayImpBurn = np.zeros((numRandSys,numMinProp,7))

    from qutils.orbital import ECI2OE
    for i in range(numRandSys):
        for j in range(numMinProp):
            OEArrayImpBurn[i,j,:] = ECI2OE(statesArrayImpBurn[i,j,0:3],statesArrayImpBurn[i,j,3:6])


    t = np.linspace(0,numMinProp*dt,len(statesArrayImpBurn[0,:,0]))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(statesArrayImpBurn[0,:,0],statesArrayImpBurn[0,:,1],statesArrayImpBurn[0,:,2],label='No Thrust')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('3D Trajectory of Earth Orbiter')
    ax.legend()
    ax.axis('equal')

    plt.figure()
    plt.plot(t, statesArrayImpBurn[0,:,0], label='No Thrust X')
    plt.plot(t, statesArrayImpBurn[0,:,1], label='No Thrust Y')
    plt.plot(t, statesArrayImpBurn[0,:,2], label='No Thrust Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (km)')
    plt.title('Position vs Time for Different Thruster Profiles')
    plt.legend()
    plt.grid()


    plt.figure()
    plt.plot(t, OEArrayImpBurn[0,:,0], label='No Thrust',color='C3')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("a")

    plt.figure()
    plt.plot(t, OEArrayImpBurn[0,:,1], label='No Thrust',color='C3')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("e")

    plt.figure()
    plt.plot(t, OEArrayImpBurn[0,:,2], label='No Thrust',color='C3')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("i")

    plt.figure()
    plt.plot(t, OEArrayImpBurn[0,:,3], label='No Thrust',color='C3')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("OMEGA")

    plt.figure()
    plt.plot(t, OEArrayImpBurn[0,:,4], label='No Thrust',color='C3')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("omega")

    plt.figure()
    plt.plot(t, OEArrayImpBurn[0,:,5], label='No Thrust',color='C3')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("nu")

    plt.figure()
    plt.plot(t, OEArrayImpBurn[0,:,6], label='No Thrust',color='C3')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("Period")

if propType == 'none' or True:
    statesArrayNoThrust = np.zeros((numRandSys,numMinProp,problemDim))

    print("Generating No Thrust Data...")
    for i in range(numRandSys):
        earthorb.SetField("SMA", SMA[i]) # km
        earthorb.SetField("ECC", ECC[i])
        earthorb.SetField("INC", INC[i]) # deg
        earthorb.SetField("RAAN", RAAN[i]) # deg
        earthorb.SetField("AOP", AOP[i]) # deg
        earthorb.SetField("TA", TA[i]) # deg

        gmat.Initialize()

        # Refresh the 'gator reference
        gator = pdprop.GetPropagator()

        gmat.Initialize()
        
        pdprop.AddPropObject(earthorb)
        pdprop.PrepareInternals()

        for j in range(numMinProp):
            gator.Step(dt)
            elapsed = elapsed + dt
            state = gator.GetState()
            statesArrayNoThrust[i,j,:] = state[0:6]
            gator.UpdateSpaceObject()
        
        del(gator)

    print("No Thrust Data Generation Complete.")

        
    if saveOn:
        if slurm:
            saveDest = '~/pendulumRNN/gmat/data/classification/'
        else:
            saveDest = 'gmat/data/classification/'

        np.savez(saveDest+'statesArrayNoThrust.npz', statesArrayNoThrust=statesArrayNoThrust)



    OEArrayNoThrust = np.zeros((numRandSys,numMinProp,7))

    from qutils.orbital import ECI2OE
    for i in range(numRandSys):
        for j in range(numMinProp):
            OEArrayNoThrust[i,j,:] = ECI2OE(statesArrayNoThrust[i,j,0:3],statesArrayNoThrust[i,j,3:6])


    t = np.linspace(0,numMinProp*dt,len(statesArrayNoThrust[0,:,0]))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(statesArrayNoThrust[0,:,0],statesArrayNoThrust[0,:,1],statesArrayNoThrust[0,:,2],label='No Thrust')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('3D Trajectory of Earth Orbiter')
    ax.legend()
    ax.axis('equal')

    plt.figure()
    plt.plot(t, statesArrayNoThrust[0,:,0], label='No Thrust X')
    plt.plot(t, statesArrayNoThrust[0,:,1], label='No Thrust Y')
    plt.plot(t, statesArrayNoThrust[0,:,2], label='No Thrust Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (km)')
    plt.title('Position vs Time for Different Thruster Profiles')
    plt.legend()
    plt.grid()


    plt.figure()
    plt.plot(t, OEArrayNoThrust[0,:,0], label='No Thrust',color='C3')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("a")

    plt.figure()
    plt.plot(t, OEArrayNoThrust[0,:,1], label='No Thrust',color='C3')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("e")

    plt.figure()
    plt.plot(t, OEArrayNoThrust[0,:,2], label='No Thrust',color='C3')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("i")

    plt.figure()
    plt.plot(t, OEArrayNoThrust[0,:,3], label='No Thrust',color='C3')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("OMEGA")

    plt.figure()
    plt.plot(t, OEArrayNoThrust[0,:,4], label='No Thrust',color='C3')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("omega")

    plt.figure()
    plt.plot(t, OEArrayNoThrust[0,:,5], label='No Thrust',color='C3')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("nu")

    plt.figure()
    plt.plot(t, OEArrayNoThrust[0,:,6], label='No Thrust',color='C3')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.title("Period")




if plotOn:
    plt.show()
