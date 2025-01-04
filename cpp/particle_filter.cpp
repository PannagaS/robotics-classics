/*
Bootstrap or Sequential Importance Resampling particle filter algorithm
*/

#include<iostream>
#include <string>
#include <sstream>
#include <list>
#include <C:\\toolbox\\eigen-3.4.0\\Eigen\\Dense>
#include <utility>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>

#include <list>

using namespace std; 
using namespace Eigen;


// Simulating sensor noise 
Vector3f generateMultivariateNormal(const Vector3f& mean, const Matrix3f& covariance) {
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


// void ExportParticles(const vector<pair<Vector3f, float>>& particles, const string& filename) {
//     std::ofstream file(filename);
//     if (file.is_open()) {
//         for (const auto& particle : particles) {
//             file << particle.first[0] << "," << particle.first[1] << "," << particle.second << "\n";
//         }
//         file.close();
//     } else {
//         std::cerr << "Unable to open file for writing\n";
//     }
// }

void ExportTrueState(const Vector3f& true_state, 
                     const string& filename, 
                     int timestep) {
    // Open file in append mode
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        // Write true state for the current timestep
        file << timestep << "," 
             << true_state[0] << "," 
             << true_state[1] << "," 
             << true_state[2] << "\n";
        file.close();
    } else {
        std::cerr << "Unable to open file for writing: " << filename << "\n";
    }
}


void ExportParticles(const vector<pair<Vector3f, float>>& particles, 
                     const string& directory, 
                     int timestep) {
    // Generate a unique filename for each timestep
    std::ostringstream filename;
    filename << directory << "/particles_timestep_" << timestep << ".csv";

    // Open file for writing
    std::ofstream file(filename.str());
    if (file.is_open()) {
        // Write particle data
        for (const auto& particle : particles) {
            file << particle.first[0] << "," 
                 << particle.first[1] << "," 
                 << particle.first[2] << "," 
                 << particle.second << "\n";
        }
        file.close();
    } else {
        std::cerr << "Unable to open file for writing: " << filename.str() << "\n";
    }
}


class Robot{

private: 

    float x, y, theta, v, w, delta_t; 
    Vector3f state = Vector3f::Zero(); 
    Vector2f control = Vector2f::Zero(); 
    int N = 100; //default number of particles = 100; 
      
    Matrix3f C = Matrix3f :: Identity();
    vector<pair<Vector3f, float>> particles; 

    // Process Noise Covraince
    Matrix3f R = Matrix3f::Identity() * 0.05;
    Vector3f mean = Vector3f::Zero();
    
    // Sensor Noise Covariance
    Matrix3f Q; 
    


public: 
    // Constructor 
    // friend Vector3f generateMultivariateNormal(Vector3f& mean, Matrix3f& covariance);
    Robot(float delta_t)
    {   
        // this->state << x, y, theta; 
        // this->control << v, w; 
        // this->N = N; // number of particles
        this->delta_t = delta_t;
        this->C << 1, 0, 0,
                0, 1, 0,
                0, 0,0.1;
        this->Q << 0.1, 0, 0,
                   0, 0.1, 0,
                   0, 0,  0.1;
    }   
        
    auto InitializeParticles(int N, Vector3f& init_state){
        /*
        Create a vector of pair<Vector3f,float> for {state, weight} of size N (number of particles). 
        Initialize all weights to be = 1/N.
        */
        Matrix3f scatter_covariance = Matrix3f::Identity() * 0.01;
        
        for(int i=0; i<N; i++){
            // random position
            Vector3f particleState = init_state +  generateMultivariateNormal(Vector3f::Zero(), scatter_covariance);
            float weight = 1.0/N; 
            this->particles.push_back({particleState, weight});
            // cout<<"Particle State init :"<<particleState.transpose()<<endl;
        }
    }



    Vector3f ParticleFilter(const Vector3f& Z){
        
        // pair<Vector3f, float> most_probable_particle;

        float totalWeight = 0.0;
        for(int i =0; i< this->particles.size(); i++){
            // Call Action Model
            Vector3f state = ActionModel(this->particles[i].first); 

            // Call Sensor Model 
            float weight = SensorModel(this->particles[i].first, Z, this->Q); 

            
            this->particles[i] = {state, weight};
            totalWeight += weight;

        }

        // Normalize weights
        for(auto& particle: this->particles){
            particle.second = particle.second / totalWeight; 
        }

        // Check for resampling
        float N_eff = 0.0; 
        for(auto& particle: this->particles){
            N_eff = N_eff + (1 / pow(particle.second,2)); 
        }
        
        // if N_eff < N/3, then resample
        if (N_eff < this->particles.size()/3){

            // Resample using Systematic Resampling Method
            int n = this->particles.size();
            vector<float> cumulative_values; 
            cumulative_values.push_back(this->particles[0].second);

            for (int i = 1; i < n; i++) {
                cumulative_values.push_back(cumulative_values[i - 1] + this->particles[i].second);
            }

            // start at a random point
            std::random_device rd;  // Seed for the random number generator
            std::mt19937 gen(rd()); // Mersenne Twister generator
            std::uniform_real_distribution<> dis(0.0, 1.0 / N);

            // Generate random starting point
            double startingPoint = dis(gen);

            vector<int> resampled_index; 
            int s; 
            for(int j=0; j<n; j++){
                double currentPoint = startingPoint + (1.0/n)*j; 
                s = 0;
                while(currentPoint > cumulative_values[s]){
                    s = s + 1; 
                }
                resampled_index.push_back(s);
            }


            // Create the new resampled particle set
            vector<pair<Vector3f, float>> resampled_particles;
            for (int index : resampled_index) {
                resampled_particles.push_back(this->particles[index]);
            }

            // Replace old particles with resampled ones
            this->particles = resampled_particles;
        }

        

        // choose the particle with with the highest weight
        auto most_probable_particle = *max_element(this->particles.begin(), this->particles.end(), [](const auto& lhs, const auto&rhs){return lhs.second<rhs.second;});
       
        // return most_probable_particle's state ONLY
        // most_probable_particle = {state, weight}
        
        //save the particles
        // ExportParticles(this->particles, "particle_filter_run.csv", );

        return most_probable_particle.first; 
    }   
    
    Vector3f ActionModel(Vector3f state){
        /*
        update states: 
        x(k+1) = A x(k) + B u(k)
        */
        pair<Matrix3f, Matrix<float, 3, 2>> result = DynamicModel(state);
        Matrix3f A; 
        Matrix<float, 3,2> B;
        A = result.first; 
        B = result.second; 

        // R is process noise covariance
        Vector3f process_noise = generateMultivariateNormal(mean, R); 
        state = A*state + B*this->control + process_noise; 
        // cout<<"Dynamics says "<<this->state.transpose()<<endl;
        // this->state[2] = fmod(this->state[2] + M_PI, 2 * M_PI) - M_PI;
        // state[2] = fmod(state[2] + M_PI, 2 * M_PI) - M_PI;
        return state; 
          
    }

    pair<Matrix3f, Matrix<float, 3, 2>> DynamicModel(Vector3f& state){
        /*
        Define your A and B
        */
        Matrix3f A = Matrix3f::Identity();
        Matrix<float, 3, 2> B ; 
        B << cos(state[2]) * this->delta_t, 0, 
             sin(state[2]) * this->delta_t, 0, 
             0, this->delta_t;
        return {A,B};

    }

    float SensorModel(const Vector3f& state, const Vector3f& Z, const Matrix3f& noise_cov) {
        /*
        y = Cx 
        error = z - y
        */
        Vector3f y = this->C * state; // Predicted observation
        Vector3f error = y - Z;
        
        // cout<<(error.transpose()*noise_cov.inverse()*error*(-0.5))<<endl;
        // float numerator = exp(-0.5 * error.transpose() * noise_cov.inverse() * error);
        float exponent = -0.5 * error.transpose() * noise_cov.inverse() * error;
        // exponent = std::max(exponent, -30.0f);
        float numerator = exp(exponent);
        float denominator = sqrt(pow(2 * M_PI, 3) * noise_cov.determinant());
        
        // cout<<"Num/Den " << numerator/denominator<<endl;
        return numerator / denominator;
        
    }
 



    void ShowParticles(){
        cout<<this->particles.size()<<endl; 
        for(int i = 0; i < particles.size(); i++){
            std::cout << "Particle " << i << ": State = (" 
                  << this->particles[i].first(0) << ", " 
                  << this->particles[i].first(1) << ", " 
                  << this->particles[i].first(2) << "), Weight = " 
                  << this->particles[i].second << std::endl;
        }
    } 


    // Getters and Setters 
    void setRobotControl(Vector2f control){
        this->control = control; 
    }

    void setRobotState(Vector3f state){
        this->state = state; 
    }

    Vector3f getRobotState(){
        return this->state;
    }

    Vector2f getRobotControl(){
        return this->control;
    }

    auto getRobotParticles(){
        return this->particles;
    }
};



   

int main(){
     
    int N = 100; // Define total particles
    // delta(t) = 0.5
    float delta_t = 0.5;
    Robot robot = Robot(delta_t); 

    /////////////////////////////////////////////////////////////////////////////
    //////////////////////       INITIALIZE PARTICLES       /////////////////////
    /////////////////////////////////////////////////////////////////////////////  
    list<Vector3f> waypoints;
    waypoints.push_back(Vector3f(0,0,1.57)); 
    waypoints.push_back(Vector3f(0, 10, 0));
    waypoints.push_back(Vector3f(10,10, M_PI));
    waypoints.push_back(Vector3f(10, 0, -M_PI/2));
    // waypoints.push_back(Vector3f(0.0,0.0,0.0));
    waypoints.push_back(Vector3f(5,-5,0));
    waypoints.push_back(Vector3f(0,0,0));
    
    robot.setRobotState(waypoints.front());
    robot.InitializeParticles(N, waypoints.front());

    // True state
    Vector3f true_state;
    true_state = waypoints.front();
    
    // Control variables
    Vector2f control = Vector2f::Zero();

    // Sensor reading
    Vector3f sensor_reading = Vector3f::Zero();
    Vector3f mean; 
    mean<< 0.0, 0.0, 0.0; 
    Matrix3f cov = Matrix3f::Identity()*0.005;
    
    // Robot State 
    Vector3f state;

    // Tuning gain (Kp) 
    float Kp = 0.5; 
    float distance_threshold = 0.5;
    float v_min = -0.2, v_max = 0.8, w_min=-3.14, w_max=3.14;
    int max_iters = 100000;

    int timestep = 0;
    string particle_directory = "output_particles"; // Directory for particle files
    string true_state_file = "true_state_pf.csv";      // File for true state values



    for(auto& waypoint:waypoints){

        for(int i=0; i<max_iters; i++){
            
            // Calculate true state of the robot (No sensor noise in measurement)
            true_state[0] = true_state[0] + control[0] * cos(true_state[2])*delta_t; 
            true_state[1] = true_state[1] + control[0] * sin(true_state[2])*delta_t; 
            true_state[2] = true_state[2] + control[1] * delta_t; 

            // Get noisy state (measurement from robot) 
            Vector3f sensor_noise = generateMultivariateNormal(mean, cov);
            sensor_reading = true_state + sensor_noise; 

            float delta_x = waypoint[0] - true_state[0];
            float delta_y = waypoint[1] - true_state[1];
            float distance_to_goal = sqrt(delta_x*delta_x + delta_y*delta_y);
            float desired_theta = atan2f(delta_y, delta_x);
            control[0] = distance_to_goal * Kp; 
            
            // guard v
            if (control[0]<=v_min)
                control[0] = v_min; 
            else if (control[0]>=v_max)
                control[0] = v_max;
            
            float angle_diff = fmod(desired_theta - true_state[2] + M_PI, 2 * M_PI) - M_PI;
            
            if (distance_to_goal <= distance_threshold){
                break;
            }
            control[1] = angle_diff / delta_t;

            if (control[1]<=w_min) control[1] = w_min;
            else if(control[1]>=w_max) control[1] = w_max;

            robot.setRobotControl(control); // write setters
           
            ///////////////////////////////////////////////////////////////////////////
            ///////////////////////  Apply Particle Filter   //////////////////////////
            ///////////////////////////////////////////////////////////////////////////
            state = robot.ParticleFilter(sensor_reading);
             // Save particles to a timestep-specific file
            
            ExportParticles(robot.getRobotParticles(), particle_directory, timestep);

            // Save true state to the separate file
            ExportTrueState(true_state, true_state_file, timestep);
            // cout<<state.transpose()<<endl;
            timestep++;
            // cout<<"True State "<<true_state.transpose()<<endl;


        }
    }

    


    return 0;
}