// Copyright 2016 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
using namespace std::chrono_literals;




/* This example creates a subclass of Node and uses std::bind() to register a
 * member function as a callback from the timer. */

class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher()
  : Node("minimal_publisher"), count_(0)
  { 
      this->declare_parameter<std::string>("topic_name", "Status"); //topic_name is var, "Status" is the default value
      this->get_parameter("topic_name", topic_name); // get topic_name 
      publisher_ = this->create_publisher<std_msgs::msg::String>(topic_name, 10);
      timer_ = this->create_wall_timer(1s, std::bind(&MinimalPublisher::timer_callback, this)); 
  }

private:
  std::string topic_name;
  void timer_callback()
  {
    auto message = std_msgs::msg::String(); 
    message.data = "System is running!";
    publisher_->publish(message); 
    RCLCPP_INFO(this->get_logger(), "Publishing %s", message.data.c_str() );

  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  size_t count_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}
