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

                const auto& events_msg = msg->events;
                _events_buffer.insert(_events_buffer.end(), msg->events.begin(), msg->events.end());
                _last_timestamp = msg->events.back().ts;
            },
            ros::VoidConstPtr(),
            ros::TransportHints().tcpNoDelay());

    _inference_timer = _nh.createTimer(ros::Duration(0.1), [&](const ros::TimerEvent& timer_event) {
        ROS_DEBUG("[EventsVideoReconstructor::inference_timer]");
        if (_events_buffer.empty())
            return;

        pybind11::array_t<double> events_array_x = pybind11::array_t<double>(_events_buffer.size());
        pybind11::array_t<double> events_array_y = pybind11::array_t<double>(_events_buffer.size());
        pybind11::array_t<double> events_array_t = pybind11::array_t<double>(_events_buffer.size());
        pybind11::array_t<double> events_array_p = pybind11::array_t<double>(_events_buffer.size());

        // Convert the events to a list of dictionaries
        for (uint32_t i = 0; i < _events_buffer.size(); ++i) {
            const auto& event            = _events_buffer[i];
            events_array_x.mutable_at(i) = event.x;
            events_array_y.mutable_at(i) = event.y;
            events_array_t.mutable_at(i) = event.ts.toSec();
            events_array_p.mutable_at(i) = event.polarity;
        }
        _events_buffer.clear();

        const pybind11::object output = _online_reconstructor->attr("process_events")(
                events_array_x, events_array_y, events_array_t, events_array_p);

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

        image_msg->header.stamp    = _last_timestamp;
        image_msg->header.frame_id = "dvs";
        _image_publisher.publish(image_msg);
        ROS_DEBUG("[EventsVideoReconstructor::inference_timer] published reconstructed image");
    });
    _image_publisher = _nh.advertise<sensor_msgs::Image>("/dvs/reconstructed_image", ROS_TOPIC_BUFFER_SIZE);

    pybind11::initialize_interpreter();
    pybind11::module_ sys          = pybind11::module_::import("sys");
    pybind11::list    path         = sys.attr("path");
    std::string       package_path = std::string(ros::package::getPath("events_video_reconstructor"));
    path.attr("append")(package_path + "/python_bindings/reconstructor");

    pybind11::module_ online_reconstructor_module = pybind11::module_::import("online_reconstructor");

    pybind11::object online_reconstructor_class = online_reconstructor_module.attr("online_reconstructor");
    _online_reconstructor                       = std::make_unique<pybind11::object>(online_reconstructor_class());
}

EventsVideoReconstructor::~EventsVideoReconstructor()
{
    pybind11::finalize_interpreter();
    ROS_DEBUG("[EventsVideoReconstructor::~EventsVideoReconstructor]");
}

} // namespace event_camera_algorithms
