#include <events_video_reconstructor/events_video_reconstructor.h>
#include <ros/ros.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "events_to_video_reconstructor_node");
    ros::NodeHandle nh("~");

    event_camera_algorithms::EventsVideoReconstructor events_video_reconstructor(nh);
    ros::spin();
    return 0;
}