#include<iostream>
#include <string>
#include <sstream>
#include <list>
#include <C:\\toolbox\\eigen-3.4.0\\Eigen/Dense>
// #include "Eigen/Dense"
#include <utility>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>

using namespace std; 
using namespace Eigen;
/*
Robot class has Linear KF implementation
*/
class Robot{
private:
    float x, y, theta; 
    float v, w;
    float delta_t; 
    Vector3f mu_state = Vector3f::Zero();
    Vector2f control = Vector2f::Zero();
    Matrix3f sigma_state = Matrix3f::Zero();

    Matrix3f Q = Matrix3f::Identity(3,3); // sensor noise 
    Matrix3f R = Matrix3f::Identity(3,3); // Sensor noise

    Matrix3f C ; 
    Matrix3f H ;

public: 
       

    Robot(float x, float y, float theta, float v, float w, float delta_t){
        /*
        state = [x, y, theta]'
        control = [v, w]'
        */
        this->x = x; 
        this->y = y;
        this->theta = theta;
        this->v = v; 
        this->w = w;

        
        this->mu_state <<  this->x, this->y, this->theta; //set pose of the robot
        this->control<< this->v, this->w; //set control 

        
        this->sigma_state = Matrix3f::Identity();
        
        this->delta_t = delta_t; 
        
        
      
        this->C << 1, 0, 0,
             0, 1, 0,
             0, 0, 0.1; 
        
        this->C = Matrix3f::Identity();
    }

    pair<Vector3f, Matrix3f> ActionModel(){
        /*
        update states: 
        mu_state(k+1) = A(k+1)*mu_state(k) + B(k+1)*control(k)
        sigma_state(k+1) = A(k+1)*sigma_state(k)*A(k+1)' + R
        */
        pair<Matrix3f, Matrix<float, 3, 2>> result = DynamicModel();
        Matrix3f A; 
        Matrix<float, 3,2> B;
        A = result.first; 
        B = result.second; 

        this->mu_state = A*this->mu_state + B*this->control; 
        this->sigma_state = A*this->sigma_state*A.transpose() + this->R; 


       return {this->mu_state, this->sigma_state};

    }


    pair<Vector3f, Matrix3f> NonlinearActionModel(){
        /*
        
        update states:
        g(u(k), mu(k-1)) 3x1 matrix
        g = [ [x + vcos(theta)*delta_t]
              [y + vsin(theta)*delta_t]
              [ theta + w*delta_t    ]]
        
        */

        Vector3f g; 
        g <<this->mu_state[0] + this->control[0]*cos(this->mu_state[2])*this->delta_t, 
            this->mu_state[1] + this->control[0]*sin(this->mu_state[2])*this->delta_t, 
            this->mu_state[2] + this->control[1]*this->delta_t; 

        // Jacobian of g 
        Matrix3f G;
        G << 1, 0, -this->control[0] * sin(this->mu_state[2]) * this->delta_t,
            0, 1,  this->control[0] * cos(this->mu_state[2]) * this->delta_t,
            0, 0,  1;
        
        this->sigma_state = G * this->sigma_state * G.transpose() + this->R; 
        this->mu_state = g; 



        return {this->mu_state, this->sigma_state};
    }

    pair<Matrix3f, Matrix<float, 3, 2>> DynamicModel(){
        /*
        x(k+1) = x(k) + v(k+1)cos(theta(k))*delta_t
        y(k+1) = y(k) + v(k+1)sin(theta(k))*delta_t
        theta(k+1) = theta(k) + w(k+1)*delta_t

        state(k+1) = A*state(k) + B*control(k)

        A = [1, 0, 0; 0, 1, 0; 0, 0, 1]
        B = [cos(theta(k))*delta_t, 0;
             sin(theta(k))*delta_t, 0;
                0           , delta_t]

        returns A(k+1) and B(k+1)
        
        */
        Matrix3f A = Matrix3f::Identity();
        Matrix<float, 3, 2> B ; 
        B << cos(this->theta)*this->delta_t, 0, 
             sin(this->theta)*this->delta_t, 0, 
             0,               this->delta_t; 
        return {A,B};

    }

    
    pair<Vector3f, Matrix3f> SensorModel(Matrix3f K, Vector3f Z){
        /*
        mu_state(k) = mu_state(k) + K*(z(k) - C(k)*mu_state(k))
        sigma_state(k) = sigma_state(k) - K*C(k)*sigma_state(k)
        */
        // Vector3f z;
        // Vector3f noise = Vector3f::Random();
        // z = this->C * this->mu_state + noise;

        this->mu_state = this->mu_state + K*(Z - this->C*this->mu_state);
        this->sigma_state = this->sigma_state - K*this->C*this->sigma_state;

        return {this->mu_state, this->sigma_state};
    }

    pair<Vector3f, Matrix3f> NonlinearSensorModel(Matrix3f K, Vector3f Z){
        
        // odometry reading = h_mu_bar 
        Vector3f h_mu_bar = this->mu_state; 
        this->mu_state = this->mu_state + K*(Z - h_mu_bar); 
        this->sigma_state = this->sigma_state - K*this->H*this->sigma_state; 

        return {this->mu_state, this->sigma_state}; 
    }

    pair<Vector3f, Matrix3f> KalmanFilter(Vector3f Z){

        // Prediction step 
        // Call action model 
        pair<Vector3f, Matrix3f> Posterior = this->ActionModel();
        this->mu_state = Posterior.first; 
        this->sigma_state = Posterior.second;
        
        // Updation step 
        // Calculate Kalman gain
        Matrix3f term_to_inverse = this->C * this->sigma_state * this->C.transpose() + this->Q;
        Matrix3f K = this->sigma_state * this->C.transpose() * term_to_inverse.inverse();
        
        // Call sensor model 
        pair<Vector3f, Matrix3f> CorrectedPosterior = this->SensorModel(K, Z);
        this->mu_state = CorrectedPosterior.first;
        this->sigma_state = CorrectedPosterior.second; 


        return {this->mu_state, this->sigma_state};
    }

    pair<Vector3f, Matrix3f> ExtendedKalmanFilter(Vector3f Z){

        // Call Non-linear Action Model
        pair<Vector3f, Matrix3f> Posterior = this->NonlinearActionModel();
        this->mu_state = Posterior.first; 
        this->sigma_state = Posterior.second;

        // Updation Step 
        // Calculate Kalman Gain 
        Matrix3f term_to_inverse; 
        term_to_inverse = this->H * this->sigma_state * this->H.transpose() + this->Q;
        Matrix3f K = this->sigma_state * this->H.transpose() * term_to_inverse.inverse();

        // Call Non-linear Sensor Model 
        pair<Vector3f, Matrix3f> CorrectedPosterior = this->NonlinearSensorModel(K, Z);
        this->mu_state = CorrectedPosterior.first;
        this->sigma_state = CorrectedPosterior.second;


        return {this->mu_state, this->sigma_state};
    }



    /*
    Getters and Setters for Robot Class
    */
    Vector3f getRobotState(){
        return mu_state;
    }

    Vector2f getRobotControl(){
        return control;
    }

    Matrix3f getRobotCov(){
        return sigma_state; 
    }

    void setRobotState(Vector3f mu_state){
        this->mu_state = mu_state;
    }

    void setRobotCov(Matrix3f sigma_state){
        this->sigma_state = sigma_state;
    }

    void setRobotControl(Vector2f control){
        this->control = control;
    }
};


Vector3f generateMultivariateNormal(Vector3f mean, Matrix3f covariance) {
    // Create a standard normal distribution
    std::random_device rd;
    std::mt19937 gen(rd()); // Random number generator
    std::normal_distribution<float> standardNormal(0.0, 1.0);

    // Sample from standard normal
    Vector3f standardSample;
    for (int i = 0; i < 3; ++i) {
        standardSample[i] = standardNormal(gen);
    }

    // Perform Cholesky decomposition of the covariance matrix
    Matrix3f L = covariance.llt().matrixL();

    // Transform the standard normal sample
    Vector3f multivariateSample = mean + L * standardSample;

    return multivariateSample;
}

int main(){
    // save trajectory 
    std::ofstream kf_estimate_csv("kf_estimate.csv");
    std::ofstream true_state_csv("true_state.csv");
    std::ofstream sensor_state_csv("sensor_state.csv");
    kf_estimate_csv<< "x,y\n";
    true_state_csv << "x,y\n";
    sensor_state_csv <<"x,y\n";

    // Define initial state
    float x = 0.0, y = 0.0, theta = 1.57; 
    float delta_t = 0.5; 
    float v_init = 0.0, w_init = 0.0; 

    // Create Robot Instance 
    Robot robot = Robot(x, y, theta, v_init, w_init, delta_t);
    
    list<Vector3f> waypoints;
    waypoints.push_back(Vector3f(0,0,0)); 
    waypoints.push_back(Vector3f(0, 10, M_PI/2));
    waypoints.push_back(Vector3f(10,10, M_PI));
    waypoints.push_back(Vector3f(10, 0, -M_PI/2));
    waypoints.push_back(Vector3f(5,-5,0));
    waypoints.push_back(Vector3f(0,0,0));
    Vector3f true_state(x, y, theta); 
    float v_min = 0.5, v_max = 5; 
    float Kp = 0.02; 
    int max_iters = 100; 
    float dist_threshold = 0.5; 
    float v = 0.0; 
    float w = 0.0;

    // sensor noise mean and covariance 
    Vector3f mean(0,0,0); 
    Matrix3f cov = Matrix3f::Identity() * 0.005;
    Vector3f sensor_reading(0,0,0);
    
    // Robot State 
    Vector3f robot_state = robot.getRobotState();

    for(Vector3f waypoint: waypoints){
        for(int i=0; i<max_iters; i++){ 
            
            // True state of the robot without sensor errors 
            true_state[0] = true_state[0] + v * cos(true_state[2])*delta_t; 
            true_state[1] = true_state[1] + v * sin(true_state[2])*delta_t; 
            true_state[2] = true_state[2] + w * delta_t; 

            // Save true state without any noise
            true_state_csv << true_state[0] << ',' << true_state[1] << "\n";

            

            // Get noisy state (measurement from robot) 
            Vector3f sensor_noise = generateMultivariateNormal(mean, cov);
            sensor_reading = true_state + sensor_noise; 
            // Save true state with sensor noise for plotting 
            sensor_state_csv << sensor_reading[0] << ',' << sensor_reading[1] << "\n";

            // calculate distance and angle to waypoint 
            //Vector3f robot_state = robot.getRobotState();
            float delta_x = waypoint[0] - true_state[0];
            float delta_y = waypoint[1] - true_state[1];
            float distance_to_goal = sqrt(delta_x*delta_x + delta_y*delta_y);
            float desired_theta = atan2f(delta_y, delta_x);
            v = distance_to_goal * Kp; 
            
            // guard v
            if (v<=v_min)
                v = v_min; 
            else if (v>=v_max)
                v = v_max;
            
            float angle_diff = fmod(desired_theta - true_state[2] + M_PI, 2 * M_PI) - M_PI;
            
            if (distance_to_goal <= dist_threshold){
                break;
            }
            w = angle_diff / delta_t;


            // Kalman Filter Step 
            Vector2f control(v,w);
            robot.setRobotControl(control);

            // pair<Vector3f, Matrix3f> kf_result = robot.KalmanFilter(sensor_reading);
            
            pair<Vector3f, Matrix3f> kf_result = robot.ExtendedKalmanFilter(sensor_reading); 


            robot.setRobotState(kf_result.first);
            robot.setRobotCov(kf_result.second);

            kf_estimate_csv << kf_result.first[0] << "," << kf_result.first[1] << "\n";



        }
    }

    cout<<"saved file";


} 