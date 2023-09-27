#pragma once
#include <dvs_msgs/EventArray.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <ros/transport_hints.h>
#include <sensor_msgs/Image.h>

#include <iostream>
#include <string>
#include <vector>

#define ROS_TOPIC_BUFFER_SIZE 100

namespace event_camera_algorithms {

class EventsVideoReconstructor
{
  public:
    EventsVideoReconstructor(ros::NodeHandle& nh);
    virtual ~EventsVideoReconstructor();

  public:
  private:
    ros::NodeHandle& _nh;

    // Subscribers
    ros::Subscriber _events_subscriber;
    ros::Publisher  _image_publisher;
    ros::Timer      _inference_timer;

    std::unique_ptr<pybind11::object> _online_reconstructor;
    std::vector<dvs_msgs::Event>      _events_buffer;
    ros::Time                         _last_timestamp;
};

} // namespace event_camera_algorithms