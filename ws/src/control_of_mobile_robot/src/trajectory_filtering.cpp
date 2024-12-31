#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "/home/ws/eigen-3.4.0/Eigen/Dense"
#include <utility>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <list>

#include "robot_interfaces/msg/sensor_msg.hpp"
#include "robot_interfaces/msg/control_msg.hpp"

// for RViz 
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using namespace std::chrono_literals;
using namespace std;
using namespace Eigen; 


/*
Robot Class
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

 


class ProcessNode : public rclcpp::Node
{   
    public:
        Vector3f sensor_reading = Vector3f::Zero();  
        Vector2f control_input = Vector2f::Zero();
        float delta_t = 0.5;  // make sure to pass this or have this fixed somewhere
        Robot robot;

     


        ProcessNode(): Node("process_node_subscriber"), robot(sensor_reading[0], sensor_reading[1], sensor_reading[2], control_input[0], control_input[1], 0.5)
        {   
            // RViz path reset (for better visualization)
            estimate_path_.header.frame_id = "map";
            estimate_path_.poses.clear();

            // Create a subscriber that subscribes to "/sensor_reading" topic and estimates robot state
            subscription_ = this->create_subscription<robot_interfaces::msg::SensorMsg>(
                "/sensor_reading", 10, std::bind(&ProcessNode::filter_data_callback, this, std::placeholders::_1));

            control_subscription_ = this->create_subscription<robot_interfaces::msg::ControlMsg>(
                "/control_input", 10, std::bind(&ProcessNode::control_data_callback, this, std::placeholders::_1));
            
            // Create a publisher that publishes to "/kalman_estimate" topic
            publisher_ = this->create_publisher<robot_interfaces::msg::SensorMsg>("/kalman_estimate", 10);



             // Publishers for RViz visualization
            true_pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/true_pose", 10);
            estimate_pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/estimate_pose", 10);
            estimate_path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("/estimate_path", 10);

            
            
        }

    private:
        void filter_data_callback(const robot_interfaces::msg::SensorMsg & data){
            
            // Do the Kalman Filtering thing here
            sensor_reading << data.x , data.y , data.theta; 
            

            RCLCPP_INFO(this->get_logger(), "Received data: x=%f, y=%f, theta=%f",
                    sensor_reading[0], sensor_reading[1], sensor_reading[2]);

            compute_kalman_estimate();

        }
        
        void control_data_callback(const robot_interfaces::msg::ControlMsg & data){

            control_input << data.v, data.w; 
            
            RCLCPP_INFO(this->get_logger(), "Received data: v=%f, w=%f",
                    control_input[0], control_input[1]);

            compute_kalman_estimate();
        }

        
        void compute_kalman_estimate(){
            robot.setRobotControl(control_input); 
            pair<Vector3f, Matrix3f> kf_result = robot.KalmanFilter(sensor_reading); 
            robot.setRobotState(kf_result.first);
            robot.setRobotCov(kf_result.second);

            publish_kalman_estimate(kf_result.first);
            
            // Publish true pose
            auto true_pose = geometry_msgs::msg::PoseStamped();
            true_pose.header.stamp = this->now();
            true_pose.header.frame_id = "map";
            true_pose.pose.position.x = sensor_reading[0];
            true_pose.pose.position.y = sensor_reading[1];
            true_pose.pose.position.z = 0.0;
            true_pose.pose.orientation = tf2::toMsg(tf2::Quaternion(0, 0, sin(sensor_reading[2] / 2), cos(sensor_reading[2] / 2)));
            true_pose_publisher_->publish(true_pose);

            // Publish Kalman estimate pose
            auto estimate_pose = geometry_msgs::msg::PoseStamped();
            estimate_pose.header.stamp = this->now();
            estimate_pose.header.frame_id = "map";
            estimate_pose.pose.position.x = kf_result.first[0];
            estimate_pose.pose.position.y = kf_result.first[1];
            estimate_pose.pose.position.z = 0.0;
            estimate_pose.pose.orientation = tf2::toMsg(tf2::Quaternion(0, 0, sin(kf_result.first[2] / 2), cos(kf_result.first[2] / 2)));
            estimate_pose_publisher_->publish(estimate_pose);

            // Add estimate pose to the path and publish
            estimate_path_.header.stamp = this->now();
            estimate_path_.poses.push_back(estimate_pose);
            estimate_path_publisher_->publish(estimate_path_);


        }

        void publish_kalman_estimate(const Vector3f &state)
        {
            

            auto message = robot_interfaces::msg::SensorMsg(); 
            message.x = state[0];
            message.y = state[1];
            message.theta = state[2];
            publisher_->publish(message);
            RCLCPP_INFO(this->get_logger(), "Publishing: x=%f, y=%f, theta=%f", message.x, message.y, message.theta);
        
        }


        rclcpp::Subscription<robot_interfaces::msg::SensorMsg>::SharedPtr subscription_;
        rclcpp::Subscription<robot_interfaces::msg::ControlMsg>::SharedPtr control_subscription_;
        rclcpp::Publisher<robot_interfaces::msg::SensorMsg>::SharedPtr publisher_;

        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr true_pose_publisher_;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr estimate_pose_publisher_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr estimate_path_publisher_;
        nav_msgs::msg::Path estimate_path_;


};  

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ProcessNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
