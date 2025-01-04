import pandas as pd
import matplotlib.pyplot as plt
import glob

# File paths
true_state_file = "true_state_pf.csv"
particle_directory = "output_particles/"

# Load true state data
true_state_data = pd.read_csv(true_state_file, header=None, 
                              names=["timestep", "true_x", "true_y", "true_theta"])

# Get all particle files
particle_files = glob.glob(particle_directory + "particles_timestep_*.csv")
particle_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Initialize lists to accumulate true state path
true_state_path_x = []
true_state_path_y = []

# Loop through particle files
for file in particle_files:
    # Extract the timestep from the filename
    timestep = int(file.split('_')[-1].split('.')[0])
    print(timestep)

    # Load particle data
    particle_data = pd.read_csv(file, header=None, 
                                names=["particle_x", "particle_y", "particle_theta", "particle_weight"])
    
    # Replace NaN weights with a default value
    particle_data["particle_weight"] = particle_data["particle_weight"].fillna(1 / 100)
    
    # Get the true state for the current timestep
    true_state = true_state_data[true_state_data["timestep"] == timestep]
    if true_state.empty:
        print(f"No true state found for timestep {timestep}")
        continue

    true_x = true_state["true_x"].iloc[0]
    true_y = true_state["true_y"].iloc[0]

    # Accumulate the true state path
    true_state_path_x.append(true_x)
    true_state_path_y.append(true_y)

    # Plot true state path
    plt.plot(true_state_path_x, true_state_path_y, color='red', label="True State Path")

    # Plot particles
    plt.scatter(particle_data["particle_x"], particle_data["particle_y"], 
                s=particle_data["particle_weight"] * 100, c='blue', alpha=0.8, label="Particles")

    # Highlight current true state
    plt.scatter(true_x, true_y, c='green', marker='o', s=100, label="Current True State")

    # Add labels and legend
    plt.title(f"Timestep: {timestep}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()
    
    # Pause for animation effect
    plt.pause(0.1)
    plt.clf()

plt.show()
