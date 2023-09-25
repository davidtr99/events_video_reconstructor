#pragma once
#include <dvs_msgs/EventArray.h>
#include <ros/ros.h>
#include <ros/transport_hints.h>
#include <ros/package.h>

#include <iostream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h> 
#include <pybind11/stl.h>

#define ROS_TOPIC_BUFFER_SIZE 1

namespace event_camera_algorithms {

class EventsVideoReconstructor {
public:
  EventsVideoReconstructor(ros::NodeHandle &nh);
  virtual ~EventsVideoReconstructor();


public:

private:
  ros::NodeHandle &_nh;

  // Subscribers
  ros::Subscriber _events_subscriber;
  std::unique_ptr<pybind11::object> _test_python_object;
};

} // namespace event_camera_algorithms