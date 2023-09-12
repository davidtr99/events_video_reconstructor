#pragma once
#include <dvs_msgs/EventArray.h>
#include <ros/ros.h>
#include <ros/transport_hints.h>
#include <torch/torch.h>

#include <iostream>
#include <string>
#include <vector>

#define ROS_TOPIC_BUFFER_SIZE 1

namespace event_camera_algorithms {

class EventsVideoReconstructor {
public:
  EventsVideoReconstructor(ros::NodeHandle &nh);
  virtual ~EventsVideoReconstructor();

private:
private:
  ros::NodeHandle &_nh;

  // Subscribers
  ros::Subscriber _events_subscriber;
};

} // namespace event_camera_algorithms