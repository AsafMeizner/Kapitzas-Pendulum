import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Settings: Choose the starting angles for the pendulum
theta_0 = np.pi / 4  # Initial angle in radians (e.g., np.pi/4 for 45 degrees)
phi_0 = np.pi / 6    # Initial angle in radians (e.g., np.pi/6 for 30 degrees)

# Define the parameters for Kapitza's pendulum
L = 1.0        # Length of the pendulum (m)
m = 1.0        # Mass of the pendulum (kg)
g = 9.81       # Gravitational acceleration (m/s^2)
Ax = 0.2       # Amplitude of the pivot's oscillation in the x-axis (m)
Ay = 0.5       # Amplitude of the pivot's oscillation in the y-axis (m)
Az = 0.3       # Amplitude of the pivot's oscillation in the z-axis (m)
omegax = 15.0  # Frequency of the pivot's oscillation in the x-axis (rad/s)
omegay = 20.0  # Frequency of the pivot's oscillation in the y-axis (rad/s)
omegaz = 10.0  # Frequency of the pivot's oscillation in the z-axis (rad/s)
time_step = 0.01  # Time step for simulation (s)
total_time = 10.0  # Total time for simulation (s)

# Define the differential equations for the pendulum's motion
def equations(y, t):
    theta, phi, omega_theta, omega_phi = y
    
    # Derivatives for theta and phi (angles)
    dtheta_dt = omega_theta
    dphi_dt = omega_phi
    
    # Acceleration due to gravity and pivot motion
    domega_theta_dt = (
        - (g / L) * np.sin(theta)
        - (Ax * omegax**2 / L) * np.sin(omegax * t) * np.cos(theta) * np.cos(phi)
        - (Ay * omegay**2 / L) * np.cos(omegay * t) * np.sin(theta)
    )
    
    domega_phi_dt = (
        - (Az * omegaz**2 / L) * np.cos(omegaz * t) * np.sin(phi)
        + (Ax * omegax**2 / L) * np.sin(omegax * t) * np.sin(phi) * np.cos(theta)
    )
    
    return np.array([dtheta_dt, dphi_dt, domega_theta_dt, domega_phi_dt])

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
phi_values = []
omega_theta_values = []
omega_phi_values = []

# Initial conditions
omega_theta_0 = 0.0  # Initial angular velocity (rad/s)
omega_phi_0 = 0.0    # Initial angular velocity (rad/s)
y = np.array([theta_0, phi_0, omega_theta_0, omega_phi_0])

# Simulation loop
for t in t_values:
    theta_values.append(y[0])
    phi_values.append(y[1])
    omega_theta_values.append(y[2])
    omega_phi_values.append(y[3])
    y = runge_kutta(y, t, time_step)

# Calculate the pivot and pendulum end positions
pivot_x_values = Ax * np.sin(omegax * t_values)
pivot_y_values = Ay * np.cos(omegay * t_values)
pivot_z_values = Az * np.sin(omegaz * t_values)

pendulum_x_values = pivot_x_values + L * np.sin(theta_values) * np.cos(phi_values)
pendulum_y_values = pivot_y_values + L * np.sin(theta_values) * np.sin(phi_values)
pendulum_z_values = pivot_z_values - L * np.cos(theta_values)

# Calculate the relative angles
relative_theta_values = np.arctan2(pendulum_y_values - pivot_y_values, pendulum_x_values - pivot_x_values)
relative_phi_values = np.arctan2(pendulum_z_values - pivot_z_values, np.sqrt((pendulum_x_values - pivot_x_values)**2 + (pendulum_y_values - pivot_y_values)**2))

# Set up the figure and axis for 3D plotting
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.set_xlim(-L-0.5, L+0.5)
ax.set_ylim(-L-0.5, L+0.5)
ax.set_zlim(-L-0.5, L+0.5)
ax.set_title("3D Kapitza's Pendulum")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid()

# Plot for angles over time (relative angles)
ax2 = fig.add_subplot(222)
ax2.set_xlim(0, total_time)
ax2.set_ylim(np.min(relative_theta_values)-0.1, np.max(relative_theta_values)+0.1)
ax2.set_title("Theta (Relative to X-Y Axis) Over Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Relative Theta (rad)")
theta_line, = ax2.plot([], [], 'b-')

ax3 = fig.add_subplot(224)
ax3.set_xlim(0, total_time)
ax3.set_ylim(np.min(relative_phi_values)-0.1, np.max(relative_phi_values)+0.1)
ax3.set_title("Phi (Relative to X-Z Plane) Over Time")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Relative Phi (rad)")
phi_line, = ax3.plot([], [], 'r-')

# Create the pendulum rod and mass
line, = ax.plot([], [], [], 'o-', lw=2)
pivot, = ax.plot([], [], [], 'ko', markersize=5)  # Pivot point

# Create text to display the current time, theta, and phi
time_template = 'Time = {:.2f} s'
theta_template = 'Relative Theta = {:.2f} rad'
phi_template = 'Relative Phi = {:.2f} rad'
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
theta_text = ax.text2D(0.05, 0.90, '', transform=ax.transAxes)
phi_text = ax.text2D(0.05, 0.85, '', transform=ax.transAxes)

# Initialize the animation
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    theta_line.set_data([], [])
    phi_line.set_data([], [])
    time_text.set_text('')
    theta_text.set_text('')
    phi_text.set_text('')
    pivot.set_data([], [])
    pivot.set_3d_properties([])
    return line, theta_line, phi_line, time_text, theta_text, phi_text, pivot

# Animation function
def animate(i):
    # Update the pendulum line
    line.set_data([pivot_x_values[i], pendulum_x_values[i]],
                  [pivot_y_values[i], pendulum_y_values[i]])
    line.set_3d_properties([pivot_z_values[i], pendulum_z_values[i]])
    
    # Update the pivot point
    pivot.set_data(pivot_x_values[i], pivot_y_values[i])
    pivot.set_3d_properties(pivot_z_values[i])
    
    # Update the live graphs
    theta_line.set_data(t_values[:i], relative_theta_values[:i])
    phi_line.set_data(t_values[:i], relative_phi_values[:i])
    
    # Update the text displays
    time_text.set_text(time_template.format(i * time_step))
    theta_text.set_text(theta_template.format(relative_theta_values[i]))
    phi_text.set_text(phi_template.format(relative_phi_values[i]))
    return line, theta_line, phi_line, time_text, theta_text, phi_text, pivot

# Create the animation
ani = FuncAnimation(
    fig, animate, init_func=init, frames=len(t_values), interval=20, blit=True)

# Display the animation and plots
plt.tight_layout()
plt.show()
