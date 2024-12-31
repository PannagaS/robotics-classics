cmake_minimum_required(VERSION 3.8)
project(control_of_mobile_robot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_default_runtime REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(robot_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SensorMsg.msg"
)

# Add executables and link dependencies
add_executable(talker src/publisher_member_function.cpp)
ament_target_dependencies(talker rclcpp std_msgs rosidl_default_runtime)

add_executable(listener src/subscriber_member_function.cpp)
ament_target_dependencies(listener rclcpp std_msgs rosidl_default_runtime)

add_executable(robot_state src/trajectory_filtering.cpp)
ament_target_dependencies(robot_state rclcpp std_msgs robot_interfaces geometry_msgs nav_msgs  tf2 tf2_geometry_msgs)

add_executable(sensor_reading src/publish_sensor_data.cpp)
ament_target_dependencies(sensor_reading rclcpp std_msgs robot_interfaces)
 
# Install executables
install(TARGETS
  talker
  listener
  robot_state
  sensor_reading
  DESTINATION lib/${PROJECT_NAME}
)

# Install message files
install(
  DIRECTORY msg
  DESTINATION share/${PROJECT_NAME}
)

# Export dependencies
ament_export_dependencies(rosidl_default_runtime)

ament_package()
