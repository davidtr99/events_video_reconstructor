#include <events_video_reconstructor/events_video_reconstructor.h>

namespace event_camera_algorithms {

EventsVideoReconstructor::EventsVideoReconstructor(ros::NodeHandle& nh) : _nh(nh)
{
    ROS_DEBUG("[EventsVideoReconstructor::EventsVideoReconstructor]");

    _events_subscriber = _nh.subscribe<dvs_msgs::EventArray>(
            "/dvs/events",
            ROS_TOPIC_BUFFER_SIZE,
            [&](const dvs_msgs::EventArray::ConstPtr& msg) {
                ROS_DEBUG("[EventsVideoReconstructor::events_subscriber])");
                pybind11::list events;
                for (const auto& event : msg->events) {
                    pybind11::dict event_dict;
                    event_dict["x"] = event.x;
                    event_dict["y"] = event.y;
                    event_dict["t"] = event.ts.toSec();
                    event_dict["p"] = event.polarity;
                    events.append(event_dict);
                }
                const pybind11::object           output       = _online_reconstructor->attr("process_events")(events);
                const pybind11::array_t<uint8_t> output_array = output.cast<pybind11::array_t<uint8_t>>();
                const pybind11::buffer_info      info         = output_array.request();

                sensor_msgs::Image::Ptr image_msg = boost::make_shared<sensor_msgs::Image>();
                image_msg->height                 = info.shape[0];
                image_msg->width                  = info.shape[1];
                image_msg->encoding               = "mono8";
                image_msg->step                   = info.strides[0];
                size_t size                       = info.size * info.itemsize;
                image_msg->data.resize(size);
                memcpy((char*)(&image_msg->data[0]), info.ptr, size);

                image_msg->header.stamp    = msg->header.stamp;
                image_msg->header.frame_id = "dvs";
                _image_publisher.publish(image_msg);
            },
            ros::VoidConstPtr(),
            ros::TransportHints().tcpNoDelay());

    _image_publisher = _nh.advertise<sensor_msgs::Image>("/dvs/reconstructed_image", ROS_TOPIC_BUFFER_SIZE);

    pybind11::initialize_interpreter();
    pybind11::module_ sys          = pybind11::module_::import("sys");
    pybind11::list    path         = sys.attr("path");
    std::string       package_path = std::string(ros::package::getPath("events_video_reconstructor"));
    path.attr("append")(package_path + "/python_bindings/reconstructor");

    pybind11::object online_reconstructor_class
            = pybind11::module_::import("online_reconstructor").attr("online_reconstructor");
    _online_reconstructor = std::make_unique<pybind11::object>(online_reconstructor_class());
}

EventsVideoReconstructor::~EventsVideoReconstructor()
{
    pybind11::finalize_interpreter();
    ROS_DEBUG("[EventsVideoReconstructor::~EventsVideoReconstructor]");
}

} // namespace event_camera_algorithms
