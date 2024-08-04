import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# Define the parameters for Kapitza's pendulum
L = 1.0        # Length of the pendulum (m)
m = 1.0        # Mass of the pendulum (kg)
g = 9.81       # Gravitational acceleration (m/s^2)
A = 0.5        # Amplitude of the pivot's oscillation (m)
omega = 20.0   # Frequency of the pivot's oscillation (rad/s)
time_step = 0.01  # Time step for simulation (s)
total_time = 10.0  # Total time for simulation (s)

# Define the differential equations for the pendulum's motion
def equations(y, t):
    theta, omega_pendulum = y
    dtheta_dt = omega_pendulum
    domega_dt = (-g / L) * np.sin(theta) - (A * omega**2 / L) * np.cos(omega * t) * np.sin(theta)
    return np.array([dtheta_dt, domega_dt])

# Implement the Runge-Kutta 4th-order method
def runge_kutta(y, t, dt):
    k1 = dt * equations(y, t)
    k2 = dt * equations(y + 0.5 * k1, t + 0.5 * dt)
    k3 = dt * equations(y + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * equations(y + k3, t + dt)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Initialize variables
t_values = np.arange(0, total_time, time_step)
theta_values = []
omega_values = []
height_values = A * np.cos(omega * t_values)  # Height of the pendulum axle

# Initial conditions
theta_0 = np.pi / 4  # Initial angle (rad)
omega_0 = 0.0        # Initial angular velocity (rad/s)
y = np.array([theta_0, omega_0])

# Simulation loop
for t in t_values:
    theta_values.append(y[0])
    omega_values.append(y[1])
    y = runge_kutta(y, t, time_step)

# Calculate the absolute angle relative to the x-axis
absolute_angle_values = np.arctan2(np.sin(theta_values), np.cos(theta_values) - (height_values / L))

# Convert the angle to Cartesian coordinates for plotting
x_values = L * np.sin(absolute_angle_values)
y_values = height_values - L * np.cos(absolute_angle_values)

# Set up the figure and axis
fig, ax = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1]})
ax[0].set_xlim(-L-0.1, L+0.1)
ax[0].set_ylim(-L-1.1, L+0.1)
ax[0].set_aspect('equal')
ax[0].grid()
ax[0].set_title("Kapitza's Pendulum")

# Plot for angle over time (absolute angle)
ax[1].set_xlim(0, total_time)
ax[1].set_ylim(np.min(absolute_angle_values)-0.1, np.max(absolute_angle_values)+0.1)
ax[1].set_title("Absolute Angle (Theta) Over Time")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Absolute Angle (rad)")
angle_line, = ax[1].plot([], [], 'b-')

# Plot for height over time
ax[2].set_xlim(0, total_time)
ax[2].set_ylim(np.min(height_values)-0.1, np.max(height_values)+0.1)
ax[2].set_title("Height of Pendulum Axle Over Time")
ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("Height (m)")
height_line, = ax[2].plot([], [], 'r-')

# Create the pendulum rod and mass
line, = ax[0].plot([], [], 'o-', lw=2)
pivot, = ax[0].plot(0, 0, 'ko', markersize=5)  # Pivot point

# Create text to display the current time, angle, and height
time_template = 'Time = {:.2f} s'
theta_template = 'Theta = {:.2f} rad'
height_template = 'Height = {:.2f} m'
time_text = ax[0].text(0.05, 0.9, '', transform=ax[0].transAxes)
theta_text = ax[0].text(0.05, 0.85, '', transform=ax[0].transAxes)
height_text = ax[0].text(0.05, 0.80, '', transform=ax[0].transAxes)

# Initialize the animation
def init():
    line.set_data([], [])
    angle_line.set_data([], [])
    height_line.set_data([], [])
    time_text.set_text('')
    theta_text.set_text('')
    height_text.set_text('')
    return line, angle_line, height_line, time_text, theta_text, height_text

# Animation function
def animate(i):
    # Update the pendulum line
    line.set_data([0, x_values[i]], [height_values[i], y_values[i]])
    # Update the live graphs
    angle_line.set_data(t_values[:i], absolute_angle_values[:i])
    height_line.set_data(t_values[:i], height_values[:i])
    # Update the text displays
    time_text.set_text(time_template.format(i * time_step))
    theta_text.set_text(theta_template.format(absolute_angle_values[i]))
    height_text.set_text(height_template.format(height_values[i]))
    return line, angle_line, height_line, time_text, theta_text, height_text

# Create the animation
ani = FuncAnimation(
    fig, animate, init_func=init, frames=len(t_values), interval=20, blit=True)

# Display the animation and plots
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
plt.show()
