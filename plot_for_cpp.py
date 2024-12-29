# import matplotlib.pyplot as plt 
# import numpy as np 
# import pandas as pd
# true_state = r"C:\Users\panna\Documents\myCpp\true_state.csv"
# kf_estimate = r"C:\Users\panna\Documents\myCpp\kf_estimate.csv" 
# sensor_state = r"C:\Users\panna\Documents\myCpp\sensor_state.csv"

# true_state_data = pd.read_csv(true_state)
# kf_estimate_data = pd.read_csv(kf_estimate)
# sensor_state_data = pd.read_csv(sensor_state)

# # Plot the trajectory
# plt.plot(sensor_state_data["x"], sensor_state_data["y"], label="Sensor Reading", color="black")
# plt.plot(kf_estimate_data["x"], kf_estimate_data["y"], label = "KF Estimate", color="red")
# plt.plot(true_state_data['x'], true_state_data['y'], label='True State', color='green' )
# plt.scatter([0, 0, 10, 10, 5, 0], [0, 10, 10, 0, -5, 0], color="red", label="Waypoints")  # Waypoints
# plt.title("Go to Goal with Kalman Filter")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()
# plt.grid()
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# File paths
true_state = r"C:\Users\panna\Documents\myCpp\true_state.csv"
kf_estimate = r"C:\Users\panna\Documents\myCpp\kf_estimate.csv"
sensor_state = r"C:\Users\panna\Documents\myCpp\sensor_state.csv"

# Load data
true_state_data = pd.read_csv(true_state)
kf_estimate_data = pd.read_csv(kf_estimate)
sensor_state_data = pd.read_csv(sensor_state)

# Extract data
true_x = true_state_data['x'].values
true_y = true_state_data['y'].values
kf_x = kf_estimate_data['x'].values
kf_y = kf_estimate_data['y'].values
sensor_x = sensor_state_data['x'].values
sensor_y = sensor_state_data['y'].values

# Initialize the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
true_line, = ax.plot([], [], label='True State', color='green')
kf_line, = ax.plot([], [], label='KF Estimate', color='red')
sensor_line, = ax.plot([], [], label='Sensor Reading', color='black')
waypoints = ax.scatter([0, 0, 10, 10, 5, 0], [0, 10, 10, 0, -5, 0], color="red", label="Waypoints")

ax.set_title("Go to Goal with EKF for a Differential Drive Mobile Robot")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
ax.grid()

# Real-time update
for i in range(len(true_x)):
    # Update data for each line
    true_line.set_data(true_x[:i+1], true_y[:i+1])
    kf_line.set_data(kf_x[:i+1], kf_y[:i+1])
    sensor_line.set_data(sensor_x[:i+1], sensor_y[:i+1])
    
    # Adjust axis limits if needed
    ax.set_xlim(min(true_x) - 1, max(true_x) + 1)
    ax.set_ylim(min(true_y) - 1, max(true_y) + 1)
    
    # Redraw the plot
    plt.pause(0.1)  # Pause for animation effect

plt.ioff()  # Turn off interactive mode
plt.show()
