import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
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
        self.Q =1

        # Set uncertainty in actuation
        self.R = np.eye(3)*1

        
        # C = [1, 1, 1]
        self.C = np.array([1, 1, 1]).reshape(1,3)

  
    def action_model(self):
        """
        update states: 
        mu_state(k+1) = A(k+1)*mu_state(k) + B(k+1)*control(k)
        sigma_state(k+1) = A(k+1)*sigma_state(k)*A(k+1)' + R
        """
        A_kp1, B_kp1 = self.dynamic_model()
        # embed()
        self.mu_state = A_kp1@self.mu_state + B_kp1@self.control 
        self.sigma_state = A_kp1@self.sigma_state@A_kp1.transpose() + self.R

        return self.mu_state, self.sigma_state

    
    def sensor_model(self, K):
        """
        mu_state(k) = mu_state(k) + K*(z(k) - C(k)*mu_state(k))
        sigma_state(k) = sigma_state(k) - K*C(k)*sigma_state(k)

        """
        z_kp1 = self.C@self.mu_state + np.random.rand(1)

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
       
        K_t = self.sigma_state @ self.C.transpose() / (self.C @ self.sigma_state @ self.C.transpose() + self.Q)

        # call sensor model
        self.mu_state, self.sigma_state = self.sensor_model(K_t)

        return self.mu_state, self.sigma_state
    



def main():

    state = np.array([0, 0, 0])
    control_input = np.array([0,0])
    delta_t = 0.5

    # Robot : x, y, theta, v, w, delta_t
    robot = Robot(state[0], state[1], state[2], control_input[0], control_input[1], delta_t)
    mu_state, sigma_state = 0, 0
    
    dim_state = len(state)

    # kalman_mu_estimates = np.zeros((iterations, dim_state))
    # kalman_cov_estimates = np.zeros((dim_state * iterations, dim_state))

    # square trajectory
    waypoints = np.array([[0, 0,np.pi/2],[0, 30, np.pi/2], [0, 90, np.pi/2], [0, 100, np.pi/2], [30,100,0], [60,100,0], [80,100,0], [100,100,-np.pi/2], [100, 90,-np.pi/2], [100, 70,-np.pi/2], [100, 40,-np.pi/2], [100, 20,-np.pi/2], [100, 0, -np.pi], [80, 0, -np.pi], [60, 0, -np.pi], [40,0, -np.pi], [10,0, -np.pi], [0,0, -np.pi]])
    # waypoints = np.array([[0,0], [0, 10], [10,10], [10, 0], [0,0]])
    kalman_mu_estimates = []
    distance_threshold =  1e-2

    for waypoint in waypoints:
        
        while True:
            
            mu_state, sigma_state = robot.kalman_filter()
            
            # v = 1; w = del(theta)/delta_t
           
            delta_x = waypoint[0] - robot.mu_state[0, 0]
            delta_y = waypoint[1] - robot.mu_state[1, 0]
            print("robot at x: {}, y: {}".format(robot.mu_state[0,0], robot.mu_state[1,0]))
            if np.linalg.norm([delta_x, delta_y]) < distance_threshold:
                break
        
            v = min(0.8, np.linalg.norm([delta_x, delta_y]))

            desired_theta = waypoint[2]
            delta_theta = (desired_theta - robot.mu_state[2,0]+ np.pi) % (2 * np.pi) - np.pi # wrap delta_theta to [-pi, pi]
            # w = delta_theta / delta_t 
            w = np.clip(delta_theta / delta_t, -2, 2)
            control_input = np.array([v, w]).reshape(2,1)
            robot.control = control_input

            # embed()
            # kalman_mu_estimates[i, :] = mu_state.reshape(1,dim_state)
            kalman_mu_estimates.append(mu_state.reshape(-1))
            # kalman_cov_estimates[3*i :3*(i+1), :] = sigma_state

        # embed()
    kalman_mu_estimates = np.array(kalman_mu_estimates)
    plt.figure()
    plt.plot(kalman_mu_estimates[:, 0], kalman_mu_estimates[:, 1], label="Estimated Trajectory", marker="o")
    plt.plot(waypoints[:, 0], waypoints[:, 1], "rx", label="Waypoints")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("Kalman Filter Estimated Trajectory")
    plt.legend()
    plt.grid()
    plt.show()
    
    
if __name__ == "__main__":
    main()