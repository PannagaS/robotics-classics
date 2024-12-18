import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from scipy.stats import multivariate_normal
from matplotlib.animation import FuncAnimation, FFMpegWriter

class Robot:
    def __init__(self, x0, y0, theta0, v, w, delta_t):
        """
        state = [x y theta]'
        control = [v, w]'

        """
        self.mu_state = np.array([x0, y0, theta0]).reshape(3,1)
        self.sigma_state = np.eye(3)


        self.control = np.array([v, w]).reshape(2,1)
        self.delta_t = delta_t

        # Set uncertainty in measurement 
        self.Q = np.eye(3)*10

        # Set uncertainty in actuation
        self.R = np.eye(3)

        
        # C = [1, 1, 1]
        # self.C = np.array([0.5, 0, 0]).reshape(1,3)
        self.C = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
    
    def update_process_noise(self):
        """
        Update process noise covariance (R) to align with the robot's heading.
        """
        theta = self.mu_state[2, 0]  # Current orientation

        # Base covariance: higher noise in forward (x) direction
        R_base = np.array([
            [0.1, 0.0, 0.0],  # Forward direction
            [0.0, 0.01, 0.0], # Lateral direction
            [0.0, 0.0, 0.01]  # Small noise in theta
        ])

        # Rotation matrix to align with heading (theta)
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

        # Rotate the process noise covariance
        self.R = rotation @ R_base @ rotation.T

  
    def action_model(self):
        """
        update states: 
        mu_state(k+1) = A(k+1)*mu_state(k) + B(k+1)*control(k)
        sigma_state(k+1) = A(k+1)*sigma_state(k)*A(k+1)' + R
        """
        A_kp1, B_kp1 = self.dynamic_model()
        # embed()
        self.update_process_noise() #update process noise to simulate more realistic behavior
        self.mu_state = A_kp1@self.mu_state + B_kp1@self.control 
        self.sigma_state = A_kp1@self.sigma_state@A_kp1.transpose() + self.R

        return self.mu_state, self.sigma_state

    
    def sensor_model(self, K):
        """
        mu_state(k) = mu_state(k) + K*(z(k) - C(k)*mu_state(k))
        sigma_state(k) = sigma_state(k) - K*C(k)*sigma_state(k)

        """
        z_kp1 = self.C@self.mu_state + np.random.normal(0, 0.1, size = (3,1))

        self.mu_state = self.mu_state + K@(z_kp1 - self.C@self.mu_state)
        self.sigma_state = self.sigma_state - K@self.C@self.sigma_state 
        
        return self.mu_state, self.sigma_state


    
    def dynamic_model(self):
        """
        x(k+1) = x(k) + v(k+1)cos(theta(k))*delta_t
        y(k+1) = y(k) + v(k+1)sin(theta(k))*delta_t
        theta(k+1) = theta(k) + w(k+1)*delta_t

        state(k+1) = A*state(k) + B*control(k)

        A = [1, 0, 0; 0, 1, 0; 0, 0, 1]
        B = [cos(theta(k))*delta_t, 0;
             sin(theta(k))*delta_t, 0;
                0           , delta_t]

        returns A(k+1) and B(k+1)
        """
        A = np.eye(3)
        theta_k = self.mu_state[2][0]

        B = np.array([[np.cos(theta_k)*self.delta_t, 0],
                      [np.sin(theta_k)*self.delta_t, 0],
                      [0 , self.delta_t]])
        
        return A, B
    
        
    def kalman_filter(self):

        # prediction step 
        # call action model
        self.mu_state, self.sigma_state = self.action_model()

        # updation step
        # calculate kalman gain
       
        K_t = self.sigma_state @ self.C.transpose() * np.linalg.inv(self.C @ self.sigma_state @ self.C.transpose() + self.Q)

        # call sensor model
        self.mu_state, self.sigma_state = self.sensor_model(K_t)

        return self.mu_state, self.sigma_state
    

from matplotlib.patches import Ellipse

def plot_covariance_ellipse(mu, sigma, ax, n_std=1, color='green'):
    """
    Plot a covariance ellipse representing the uncertainty of the state estimate.
    
    Parameters:
    - mu: State mean (x, y, theta)
    - sigma: Covariance matrix (3x3)
    - ax: Matplotlib Axes to plot on
    - n_std: Number of standard deviations for the ellipse size
    - color: Color of the ellipse
    """
    # Extract the covariance matrix for x and y
    sigma_xy = sigma[:2, :2]

    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(sigma_xy)

    # Calculate ellipse parameters
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvals)

    # Create and add ellipse
    ellipse = Ellipse(xy=(mu[0, 0], mu[1, 0]), width=width, height=height,
                      angle=angle, edgecolor=color, facecolor='none', lw=1.5)
    ax.add_patch(ellipse)

def plot_covariance_gradient(mu, sigma, ax, grid_size=0.1, cmap='Blues'):
    """
    Visualize the covariance spread as a color gradient using a 2D Gaussian PDF.

    Parameters:
    - mu: Mean (x, y, theta) as a 2x1 array
    - sigma: Covariance matrix (3x3), we use only the top-left 2x2 part
    - ax: Matplotlib Axes to plot on
    - grid_size: Resolution of the grid
    - cmap: Colormap for the gradient
    """
    # Extract the mean and covariance for x, y
    mu_x, mu_y = mu[0, 0], mu[1, 0]
    sigma_xy = sigma[:2, :2]

    # Generate a grid around the mean
    x_range = np.linspace(mu_x - 3, mu_x + 3, int(6 / grid_size))
    y_range = np.linspace(mu_y - 3, mu_y + 3, int(6 / grid_size))
    X, Y = np.meshgrid(x_range, y_range)

    # Flatten the grid and compute the Gaussian PDF
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mean=[mu_x, mu_y], cov=sigma_xy)
    Z = rv.pdf(pos)

    # Plot the gradient
    ax.contourf(X, Y, Z, levels=30, cmap=cmap, alpha=0.6)

def main():
    # Initial robot state
    x0, y0, theta0 = 0, 0, 0  # Starting at origin facing right (0 radians)
    delta_t = 0.5  # Time step
    v_initial = 0  # Initial linear velocity
    w_initial = 0  # Initial angular velocity

    # Create Robot instance
    robot = Robot(x0, y0, theta0, v_initial, w_initial, delta_t)

    # Define square trajectory waypoints: [x, y, theta]
    waypoints = np.array([
        [0, 0, 0],        
        [0, 10, np.pi/2],  
        [10, 10, np.pi],  
        [10,0, -np.pi/2], 
        [5, -5, 0], 
        [0, 0, 0]          
    ])

    true_state = np.array([x0, y0, theta0]).reshape(3,1)
    # Parameters
    distance_threshold = 0.5 # Stop when within this distance of a waypoint
    max_iterations =  100  # To prevent infinite loops
    trajectory = []  # Store estimated trajectory for visualization
    Kp = 0.02
    vmin = 0.2
    vmax = 1
    
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sensor_readings = []       # Simulated sensor data
    ground_truth = []          # True robot trajectory
    for waypoint in waypoints:
        for _ in range(max_iterations):
            
            # Calculate distance and angle to waypoint
            delta_x = waypoint[0] - robot.mu_state[0, 0]
            delta_y = waypoint[1] - robot.mu_state[1, 0]
            distance_to_goal = np.sqrt(delta_x**2 + delta_y**2)
            desired_theta = np.arctan2(delta_y, delta_x)    

            v = distance_to_goal * Kp
            v = np.clip(v, vmin, vmax)

            angle_diff = (desired_theta - robot.mu_state[2, 0] + np.pi) % (2 * np.pi) - np.pi  # Wrap angle to [-pi, pi]
            
            # Break if robot is close enough to the waypoint
            if distance_to_goal <= distance_threshold:
                break
            
             
            w = np.clip(angle_diff / delta_t, -20, 20)  # Dynamic angular velocity

            # #################################################################################
            # # Calculate error to waypoint
            # delta_x = waypoint[0] - true_state[0, 0]
            # delta_y = waypoint[1] - true_state[1, 0]
            # distance_to_goal = np.sqrt(delta_x**2 + delta_y**2)
            # if distance_to_goal <= distance_threshold:
            #     break

            # # True motion (ground truth)
            # desired_theta = np.arctan2(delta_y, delta_x)
            # v = np.clip(distance_to_goal * Kp, vmin, vmax)
            # w = (desired_theta - true_state[2, 0]) / delta_t
            # true_state[0, 0] += v * np.cos(true_state[2, 0]) * delta_t
            # true_state[1, 0] += v * np.sin(true_state[2, 0]) * delta_t
            # true_state[2, 0] += w * delta_t
            

            ####################### GET THE NOISY STATE (MEASUREMENT) FROM THE ROBOT ########
            sensor_noise = np.random.multivariate_normal([0, 0, 0], np.diag([0.5, 0.5, 0.05]))
            sensor_reading = true_state.flatten() + sensor_noise
            sensor_readings.append(sensor_reading)
    
            # Update control inputs in robot
            robot.control = np.array([v, w]).reshape(2, 1)

            # Perform Kalman filter step
            mu_state, sigma_state = robot.kalman_filter()

            # Store estimated state
            trajectory.append(mu_state.flatten())
            ground_truth.append(true_state.flatten())
            sensor_readings.append(sensor_reading)
             
            # Plot the trajectory and covariance in real time
            ax.clear()
            ax.set_xlim(-5, 15)
            ax.set_ylim(-15, 15)
            ax.grid()
            ax.plot([wp[0] for wp in waypoints], [wp[1] for wp in waypoints], 'rx', markersize=10, label="Waypoints")
            trajectory_array = np.array(trajectory)
            # sensor_readings_np = np.array(sensor_readings)
            # ground_truth_np = np.array(ground_truth)
            # ax.plot(ground_truth_np[:, 0], ground_truth_np[:, 1], 'black', label="Ground Truth Trajectory")
            # ax.plot(sensor_readings_np[:, 0], sensor_readings_np[:, 1], 'r.', label="Sensor Readings")
            ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'b-', label="Estimated Trajectory")
            

            # Plot covariance ellipse
            plot_covariance_ellipse(mu_state, sigma_state, ax)
            # plot_covariance_gradient(mu_state, sigma_state, ax)


            ax.legend()
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.set_title("Real-Time Robot Trajectory with Covariance Ellipses")
            plt.pause(0.01)

    plt.ioff()  # Turn off interactive mode
    plt.show()

    
    # trajectory = np.array(trajectory)
  
    # # Plot the trajectory and waypoints
    # plt.figure(figsize=(8, 8))
    # plt.plot(trajectory[:, 0], trajectory[:, 1], label="Estimated Trajectory", marker="o")
    # plt.plot(waypoints[:, 0], waypoints[:, 1], "rx", label="Waypoints")
    # plt.xlabel("X Position")
    # plt.ylabel("Y Position")
    # plt.title("Simulated Robot Trajectory")
    # plt.legend()
    # plt.grid()
    # plt.show()


if __name__ == "__main__":
    main()