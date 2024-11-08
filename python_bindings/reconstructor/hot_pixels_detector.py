#!/usr/bin/env python3
import numpy as np
import rospy
from dvs_msgs.msg import Event, EventArray
import cv2
import os

EVENTS_TOPIC = "/dvs/events"
HEIGHT = 480
WIDTH = 640
TIME_RECOLECTION_SECS = 5
THRESHOLD = 0.01

counter_matrix = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
callback_counter = 0


def callback(msg):
    global callback_counter
    global counter_matrix
    global events_subscriber

    if callback_counter == 0:
        rospy.Timer(
            rospy.Duration(TIME_RECOLECTION_SECS), calculate_hot_pixels, oneshot=True
        )
        rospy.loginfo(
            "\033[1;34m--> Starting to collect events ({} secs)...\033[0m".format(
                TIME_RECOLECTION_SECS
            )
        )

    for event in msg.events:
        counter_matrix[event.y, event.x] += 1

    callback_counter += 1


def calculate_hot_pixels(time_event):
    global events_subscriber
    global counter_matrix

    events_subscriber.unregister()
    rospy.loginfo(
        "\033[1;32m--> Calculating Hot Pixels with {} packets...\033[0m".format(
            callback_counter
        )
    )

    # min / max normalization
    rospy.loginfo(
        "Min: {}, Max: {}".format(np.min(counter_matrix), np.max(counter_matrix))
    )
    counter_matrix = (counter_matrix - np.min(counter_matrix)) / (
        np.max(counter_matrix) - np.min(counter_matrix)
    )

    # thresholding
    hot_pixels = np.where(counter_matrix > THRESHOLD)
    rospy.loginfo("Hot Pixels: {}".format(len(hot_pixels[0])))

    # create image
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:, :, 0] = counter_matrix * 255
    img[:, :, 1] = counter_matrix * 255
    img[:, :, 2] = counter_matrix * 255
    for i in range(len(hot_pixels[0])):
        cv2.circle(img, (hot_pixels[1][i], hot_pixels[0][i]), 2, (0, 0, 255), -1)

    # save hot pixels
    script_path = os.path.dirname(os.path.abspath(__file__))
    rospy.loginfo(
        "\033[1;32m--> Hot Pixels saved in {}.\033[0m".format(
            script_path + "/hot_pixels.txt"
        )
    )
    hot_pixels_file = open(script_path + "/hot_pixels.txt", "w")
    for i in range(len(hot_pixels[0])):
        print("{},{}".format(hot_pixels[1][i], hot_pixels[0][i]))
        hot_pixels_file.write("{},{}\n".format(hot_pixels[1][i], hot_pixels[0][i]))
    hot_pixels_file.close()

    # counter_matrix = counter_matrix * 255 / np.max(counter_matrix)
    cv2.imshow("Hot Pixels", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rospy.loginfo("\033[1;32m--> Hot Pixels Detector Node Finished.\033[0m")
    rospy.signal_shutdown("Hot Pixels Detector Node Finished.")


def main():
    global events_subscriber
    global timer
    rospy.init_node("hot_pixels_detector")
    events_subscriber = rospy.Subscriber(
        EVENTS_TOPIC, EventArray, callback, queue_size=1000
    )

    rospy.loginfo("\033[1;32m--> Hot Pixels Detector Node Started.\033[0m")
    rospy.loginfo(
        "\033[1;34m--> Place in a static environment and don't move the camera!\033[0m"
    )
    rospy.spin()


if __name__ == "__main__":
    main()
