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

using namespace std::chrono_literals;
using namespace std;
using namespace Eigen; 


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

class SensorNode : public rclcpp::Node
{
public:
    SensorNode() : Node("sensor_node_publisher")
    {   
        publisher_ = this->create_publisher<robot_interfaces::msg::SensorMsg>("/sensor_reading", 10);
        control_publisher_ = this->create_publisher<robot_interfaces::msg::ControlMsg>("/control_input", 10); 
        timer_ = this->create_wall_timer(100ms, std::bind(&SensorNode::timer_callback, this));
        

        // Initialize robot state
        true_state << 0.0, 0.0, 1.57; // Initial state
        waypoints = {
            Vector3f(0, 0, 0),
            Vector3f(0, 10, M_PI / 2),
            Vector3f(10, 10, M_PI),
            Vector3f(10, 0, -M_PI / 2),
            Vector3f(5, -5, 0),
            Vector3f(0.001, 0, 0)};
        
         
        current_waypoint = waypoints.begin();

        control_input << 0.0, 0.0; // Initial control inputs
    }

private:
    rclcpp::Publisher<robot_interfaces::msg::SensorMsg>::SharedPtr publisher_;
    rclcpp::Publisher<robot_interfaces::msg::ControlMsg>::SharedPtr control_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    Vector3f true_state;
    list<Vector3f> waypoints;
    list<Vector3f>::iterator current_waypoint;
    Vector2f control_input; 


    void timer_callback()
    {    
        if (current_waypoint == waypoints.end())
        {
            RCLCPP_INFO(this->get_logger(), "All waypoints reached.");
            rclcpp::shutdown(); // Shutdown the node gracefully
            return;
        }

        // Calculate control inputs
        float delta_x = (*current_waypoint)[0] - true_state[0];
        float delta_y = (*current_waypoint)[1] - true_state[1];
        float distance_to_goal = sqrt(delta_x * delta_x + delta_y * delta_y);
        float desired_theta = atan2f(delta_y, delta_x);

        float v = std::clamp(distance_to_goal * 0.02f, 0.5f, 5.0f);
        float angle_diff = fmod(desired_theta - true_state[2] + M_PI, 2 * M_PI) - M_PI;
        float w = angle_diff / 0.5;

        // Update true state
        true_state[0] += v * cos(true_state[2]) * 0.5;
        true_state[1] += v * sin(true_state[2]) * 0.5;
        true_state[2] += w * 0.5;

        // Get noisy state
        Vector3f sensor_noise = generateMultivariateNormal(Vector3f::Zero(), Matrix3f::Identity() * 0.005);
        Vector3f sensor_reading = true_state + sensor_noise;

        // Check if waypoint is reached
        if (distance_to_goal <= 0.5f)
        {
            ++current_waypoint; // Move to next waypoint
        }

        // Publish the state
        publish_state(sensor_reading);

        // Publish control input
        control_input << v, w; 
        publish_control(control_input);
    }

    void publish_state(const Vector3f &state)
    {
         

        auto message = robot_interfaces::msg::SensorMsg(); 
        message.x = state[0];
        message.y = state[1];
        message.theta = state[2];
        publisher_->publish(message);
        RCLCPP_INFO(this->get_logger(), "Publishing: x=%f, y=%f, theta=%f", message.x, message.y, message.theta);
    
    }

    void publish_control(const Vector2f &control_input){

        auto message = robot_interfaces::msg::ControlMsg();
        message.v = control_input[0]; 
        message.w = control_input[1];
        control_publisher_->publish(message); 
        RCLCPP_INFO(this->get_logger(), "Control: v=%f, w=%f", message.v, message.w);
    }

    
};


int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SensorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
