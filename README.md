# This is a Robotics Revision Repository (includes both Python and C++ implementation)

## LKF implemented in C++, visualized in RViz
![LKF-in-C++](https://github.com/PannagaS/robotics-classics/blob/main/assets/RVizPathforLinearKF%20gif.gif)
```
docker run -it --rm --gpus=all -v "C:\Users\panna\Documents\winter_break\robotics-classics\ws:/home/ws/"  --env="DISPLAY=192.168.50.174:0" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"   osrf/ros:humble-desktop
```

```
source /opt/ros/humble/setup.bash
```

Build the package once inside - 
```
colcon build
```

```
source install/setup.bash
```

## EKF implemented in Python 
![EKF-in-Python](https://github.com/PannagaS/robotics-classics/blob/main/assets/EKF%20in%20Python%20gif.gif)

--- 
## EKF implemented in C++
![EKF-in-C++](https://github.com/PannagaS/robotics-classics/blob/main/assets/EKF%20in%20C%2B%2B%20gif.gif)

---
### With covariance ellipse 
![KF-with-covariance-ellipse](https://github.com/PannagaS/robotics-classics/blob/main/assets/KF%20with%20ellipse%20gif.gif)

---
## Particle Filter 
### Particle Filter implementation in C++ 
![pf-gif](https://github.com/PannagaS/robotics-classics/blob/main/assets/particle_filter_gif.gif)
