#include <events_video_reconstructor/events_video_reconstructor.h>

namespace event_camera_algorithms {

EventsVideoReconstructor::EventsVideoReconstructor(ros::NodeHandle& nh) : _nh(nh)
{
    ROS_DEBUG("[EventsVideoReconstructor::EventsVideoReconstructor]");

    std::string events_topic;
    _nh.param<std::string>("events_topic", events_topic, "/dvs/events");

    std::string video_topic;
    _nh.param<std::string>("video_topic", video_topic, "/dvs/video");

    _nh.param<float>("output_frequency", _output_frequency, 10.0);

    if (_output_frequency > 0.0)
    {
        _inference_timer = _nh.createTimer(ros::Duration(1.0 / _output_frequency), [&](const ros::TimerEvent& timer_event) {
            ROS_DEBUG("[EventsVideoReconstructor::inference_timer]");
            if (_events_buffer.empty())
                return;

            pybind11::object output;
            runInference(_events_buffer, output);
            ROS_INFO_ONCE("\033[1;34m--> Neural Network Initialized! Running!.\033[0m");
            _events_buffer.clear();

            sensor_msgs::Image image_msg;
            generateRosImageMsg(output, image_msg);
            publishReconstructedImage(image_msg);
        });
    }


    _events_subscriber = _nh.subscribe<dvs_msgs::EventArray>(
            events_topic,
            ROS_TOPIC_BUFFER_SIZE,
            [&](const dvs_msgs::EventArray::ConstPtr& msg) {
                ROS_DEBUG("[EventsVideoReconstructor::events_subscriber])");

                const auto& events_msg = msg->events;
                _events_buffer.insert(_events_buffer.end(), msg->events.begin(), msg->events.end());
                _last_timestamp  = msg->events.back().ts;
                _events_frame_id = msg->header.frame_id;

                if (_output_frequency > 0.0)
                    return;

                pybind11::object output;
                runInference(_events_buffer, output);
                ROS_INFO_ONCE("\033[1;34m--> Neural Network Initialized! Running!.\033[0m");
                _events_buffer.clear();

                sensor_msgs::Image image_msg;
                generateRosImageMsg(output, image_msg);
                publishReconstructedImage(image_msg);
                },
                ros::VoidConstPtr(),
                ros::TransportHints().tcpNoDelay());

    _image_publisher = _nh.advertise<sensor_msgs::Image>(video_topic, ROS_TOPIC_BUFFER_SIZE);
    initializePythonObjects();
}

void EventsVideoReconstructor::initializePythonObjects()
{
    ROS_DEBUG("[EventsVideoReconstructor::initializePythonObjects]");
    pybind11::initialize_interpreter();

    pybind11::module_ sys          = pybind11::module_::import("sys");
    pybind11::list    path         = sys.attr("path");
    std::string       package_path = std::string(ros::package::getPath("events_video_reconstructor"));
    path.attr("append")(package_path + "/python_bindings/reconstructor");

    pybind11::module_ online_reconstructor_module = pybind11::module_::import("online_reconstructor");
    pybind11::object  online_reconstructor_class  = online_reconstructor_module.attr("online_reconstructor");

    _online_reconstructor = std::make_unique<pybind11::object>(online_reconstructor_class());
}

void EventsVideoReconstructor::runInference(
        const std::vector<dvs_msgs::Event>& input_events,
        pybind11::object&                   output_object)
{
    ROS_DEBUG("[EventsVideoReconstructor::runInference]");
    pybind11::array_t<double> events_array_x = pybind11::array_t<double>(input_events.size());
    pybind11::array_t<double> events_array_y = pybind11::array_t<double>(input_events.size());
    pybind11::array_t<double> events_array_t = pybind11::array_t<double>(input_events.size());
    pybind11::array_t<double> events_array_p = pybind11::array_t<double>(input_events.size());

    for (uint32_t i = 0; i < input_events.size(); ++i) {
        const auto& event            = input_events[i];
        events_array_x.mutable_at(i) = event.x;
        events_array_y.mutable_at(i) = event.y;
        events_array_t.mutable_at(i) = event.ts.toSec();
        events_array_p.mutable_at(i) = event.polarity;
    }

    output_object = _online_reconstructor->attr("process_events")(
            events_array_x, events_array_y, events_array_t, events_array_p);
}

void EventsVideoReconstructor::generateRosImageMsg(
        const pybind11::object& input_object,
        sensor_msgs::Image&     output_image_msg)
{
    ROS_DEBUG("[EventsVideoReconstructor::generateRosImageMsg]");
    const pybind11::array_t<uint8_t> input_array = input_object.cast<pybind11::array_t<uint8_t>>();
    const pybind11::buffer_info      info        = input_array.request();

    output_image_msg.height   = info.shape[0];
    output_image_msg.width    = info.shape[1];
    output_image_msg.encoding = "mono8";
    output_image_msg.step     = info.strides[0];

    size_t size = info.size * info.itemsize;
    output_image_msg.data.resize(size);

    memcpy((char*)(&output_image_msg.data[0]), info.ptr, size);
}

void EventsVideoReconstructor::publishReconstructedImage(sensor_msgs::Image& image_msg)
{
    ROS_DEBUG("[EventsVideoReconstructor::publishReconstructedImage]");

    if (!_events_frame_id.empty()) {
        image_msg.header.frame_id = _events_frame_id + "_optical_link";
    } else {
        image_msg.header.frame_id = "dvs_optical_link";
    }

    image_msg.header.stamp = _last_timestamp;
    _image_publisher.publish(image_msg);
}

EventsVideoReconstructor::~EventsVideoReconstructor()
{
    ROS_DEBUG("[EventsVideoReconstructor::~EventsVideoReconstructor]");
    pybind11::finalize_interpreter();
}

} // namespace event_camera_algorithms
