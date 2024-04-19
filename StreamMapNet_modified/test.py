import numpy as np

# Constants
total_distance = np.sqrt(5**2 + 5**2)  # Euclidean distance
speed_first_half = 5  # m/s
speed_second_half = 10  # m/s
dt = 0.1  # seconds

# Time to travel each half
time_first_half = (total_distance / 2) / speed_first_half
time_second_half = (total_distance / 2) / speed_second_half

# Steps for each half
steps_first_half = int(time_first_half / dt)
steps_second_half = int(time_second_half / dt)

# Total steps should be 20; adjust if necessary
total_steps = steps_first_half + steps_second_half
if total_steps < 20:
    steps_second_half += 20 - total_steps
elif total_steps > 20:
    steps_second_half -= total_steps - 20

# Generate position vectors
positions = np.zeros((20, 2))
for i in range(1, steps_first_half + 1):
    positions[i] = positions[i - 1] + (5 / steps_first_half) * np.array([1, 1])

for i in range(steps_first_half, 20):
    positions[i] = positions[i - 1] + (5 / steps_second_half) * np.array([1, 1])

# Example numpy array with shape (20, 2)
# positions = np.random.rand(20, 2)  # Replace this with your actual positions array

# Time interval
dt = 0.1

# Calculate differences in position
differences = np.diff(positions, axis=0)

# Calculate velocity (change in position over time)
velocity = differences / dt

# If you need the magnitude of velocity (speed)
speed = np.linalg.norm(velocity, axis=1)
# print(speed)
breakpoint()
