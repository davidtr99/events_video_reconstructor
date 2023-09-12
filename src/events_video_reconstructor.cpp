#include <events_video_reconstructor/events_video_reconstructor.h>

namespace event_camera_algorithms {

EventsVideoReconstructor::EventsVideoReconstructor(ros::NodeHandle &nh)
    : _nh(nh) {
  ROS_DEBUG("[EventsVideoReconstructor::EventsVideoReconstructor]");

  _events_subscriber = _nh.subscribe<dvs_msgs::EventArray>(
      "/dvs/events", ROS_TOPIC_BUFFER_SIZE,
      [&](const dvs_msgs::EventArray::ConstPtr &msg) {
        // ROS_DEBUG("[EventsVideoReconstructor::events_subscriber])");
        ROS_DEBUG_STREAM("[C++] Message timestamp: " << msg->events[0].ts);
      },
      ros::VoidConstPtr(), ros::TransportHints().tcpNoDelay());
}

EventsVideoReconstructor::~EventsVideoReconstructor() {
  ROS_DEBUG("[EventsVideoReconstructor::~EventsVideoReconstructor]");
}

} // namespace event_camera_algorithms
