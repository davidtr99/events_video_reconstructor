#include <events_video_reconstructor/events_video_reconstructor.h>

namespace event_camera_algorithms {

EventsVideoReconstructor::EventsVideoReconstructor(ros::NodeHandle &nh)
    : _nh(nh) {
  ROS_DEBUG("[EventsVideoReconstructor::EventsVideoReconstructor]");

  _events_subscriber = _nh.subscribe<dvs_msgs::EventArray>(
      "/dvs/events", ROS_TOPIC_BUFFER_SIZE,
      [&](const dvs_msgs::EventArray::ConstPtr &msg) {
        ROS_DEBUG("[EventsVideoReconstructor::events_subscriber])");
      },
      ros::VoidConstPtr(), ros::TransportHints().tcpNoDelay());

      pybind11::initialize_interpreter();
      pybind11::module_ sys = pybind11::module_::import("sys");
      pybind11::list path = sys.attr("path");
      std::string package_path = std::string(ros::package::getPath("events_video_reconstructor"));
      path.attr("append")(package_path + "/python_bindings");

      pybind11::module_ test_module = pybind11::module_::import("test_file");
      _test_python_object = std::make_unique<pybind11::object>(test_module.attr("test_module")());
      const auto res = _test_python_object->attr("test_function")(5,10);
      std::cout << "res: " << res.cast<int>() << std::endl;
}

EventsVideoReconstructor::~EventsVideoReconstructor() {
  pybind11::finalize_interpreter();
  ROS_DEBUG("[EventsVideoReconstructor::~EventsVideoReconstructor]");
}

} // namespace event_camera_algorithms
