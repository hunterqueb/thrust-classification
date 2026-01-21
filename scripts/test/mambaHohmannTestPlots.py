import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('WebAgg')

# This script generates a plot for the Mamba Hohmann test results.
# plot the randomly generated data bounds

# plot the circular bounds betweeen 200e3 m and 1000e3 m
# the bounds are 200e3 m and 1000e3 m
R = 6371e3  # radius of the Earth in meters
r_min = R + 200e3  # minimum radius in meters
r_max = R + 1000e3  # maximum radius in meters
r_min = r_min / 1000  # convert to kilometers
r_max = r_max / 1000  # convert to kilometers
R = R / 1000  # convert to kilometers

# create a figure and plot the two circles
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
# ax.set_xlim(-1200, 1200)
# ax.set_ylim(-1200, 1200)
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_title('Random Initial Circular Orbit Bounds')
ax.plot(np.cos(np.linspace(0, 2 * np.pi, 100)) * r_min, np.sin(np.linspace(0, 2 * np.pi, 100)) * r_min, 'r--', label='Min Radius')
ax.plot(np.cos(np.linspace(0, 2 * np.pi, 100)) * r_max, np.sin(np.linspace(0, 2 * np.pi, 100)) * r_max, 'b--', label='Max Radius')
# plot a circle at the origin with radius 6731 km
ax.plot(np.cos(np.linspace(0, 2 * np.pi, 100)) * R, np.sin(np.linspace(0, 2 * np.pi, 100)) * R, 'g', label='Earth Radius')
plt.legend()
plt.grid()

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
# ax.set_xlim(-1200, 1200)
# ax.set_ylim(-1200, 1200)
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_title('Random Initial Circular Orbit Bounds')
ax.plot(np.cos(np.linspace(0, 2 * np.pi, 100)) * r_min, np.sin(np.linspace(0, 2 * np.pi, 100)) * r_min, 'r--', label='Min Radius')
ax.plot(np.cos(np.linspace(0, 2 * np.pi, 100)) * r_max, np.sin(np.linspace(0, 2 * np.pi, 100)) * r_max, 'b--', label='Max Radius')
# plot a circle at the origin with radius 6731 km
plt.legend()
plt.grid()
plt.show()