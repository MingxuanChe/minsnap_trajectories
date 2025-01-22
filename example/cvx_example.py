import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
N = 7  # Order of polynomial (degree = N-1)
M = 10  # Number of segments
dt = 1.0  # Time duration per segment
t = np.linspace(0, M * dt, M + 1)  # Segment times
a_max = 0.7  # Maximum allowed acceleration

# Define variables: coefficients of the polynomial for each segment
coeff = cp.Variable((M, N))

# Objective function: minimize snap (4th derivative squared)
snap_cost = 0
for m in range(M):
    for i in range(4, N):
        snap_cost += cp.sum_squares(coeff[m, i] * i * (i - 1) * (i - 2) * (i - 3))

# Constraints
constraints = []

# Add boundary conditions (e.g., start and end positions/velocities)
constraints += [
    coeff[0, 0] == 0,  # Start position
    coeff[0, 1] == 0,  # Start velocity
    coeff[-1, 0] == 10,  # End position
    coeff[-1, 1] == 0,  # End velocity
]

# Add continuity constraints between segments (position, velocity, acceleration)
for m in range(M - 1):
    # Position continuity
    constraints += [cp.sum([coeff[m, i] * dt**i for i in range(N)]) ==
                    coeff[m + 1, 0]]
    # Velocity continuity
    constraints += [cp.sum([i * coeff[m, i] * dt**(i - 1) for i in range(1, N)]) ==
                    coeff[m + 1, 1]]
    # Acceleration continuity
    constraints += [cp.sum([i * (i - 1) * coeff[m, i] * dt**(i - 2) for i in range(2, N)]) ==
                    coeff[m + 1, 2]]

# Add acceleration constraints
for m in range(M):
    for t_sample in np.linspace(0, dt, 10):  # Sample 10 points per segment
        acc = cp.sum([i * (i - 1) * coeff[m, i] * t_sample**(i - 2) for i in range(2, N)])
        constraints += [cp.abs(acc) <= a_max]

# Solve the problem
problem = cp.Problem(cp.Minimize(snap_cost), constraints)
problem.solve()

# Check solution status
if problem.status != cp.OPTIMAL:
    print("No feasible solution found!")
    exit()

# Extract coefficients
coeff_values = coeff.value

# Generate trajectory and visualize
time = np.linspace(0, M * dt, 1000)  # Dense time array for plotting
pos = []
vel = []
acc = []

# Evaluate polynomials and their derivatives
for t_sample in time:
    seg = int(t_sample // dt)  # Determine segment
    seg_time = t_sample % dt
    if seg >= M:  # Handle edge case for the last time point
        seg = M - 1
        seg_time = dt

    # Compute position, velocity, and acceleration
    p = sum(coeff_values[seg, i] * seg_time**i for i in range(N))
    v = sum(i * coeff_values[seg, i] * seg_time**(i - 1) for i in range(1, N))
    a = sum(i * (i - 1) * coeff_values[seg, i] * seg_time**(i - 2) for i in range(2, N))

    pos.append(p)
    vel.append(v)
    acc.append(a)

# Plot results
plt.figure(figsize=(12, 8))

# Position plot
plt.subplot(3, 1, 1)
plt.plot(time, pos, label="Position", color='b')
plt.title("Minimal Snap Trajectory")
plt.ylabel("Position (m)")
plt.grid()

# Velocity plot
plt.subplot(3, 1, 2)
plt.plot(time, vel, label="Velocity", color='g')
plt.ylabel("Velocity (m/s)")
plt.grid()

# Acceleration plot
plt.subplot(3, 1, 3)
plt.plot(time, acc, label="Acceleration", color='r')
plt.axhline(y=a_max, color='k', linestyle='--', label="a_max")
plt.axhline(y=-a_max, color='k', linestyle='--')
plt.ylabel("Acceleration (m/s²)")
plt.xlabel("Time (s)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
