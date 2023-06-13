#!/usr/bin/env python3

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import PointCloud2
import sys
import pcl
from utils.utils_pcl import *

class GraspGenerator():
    def __init__(self):
        self._depth_sub = rospy.Subscriber(
            "/camera/depth/color/points",
            PointCloud2,
            self.callback_pc,
            queue_size=1,
        )

    def callback_pc(self, msg):
        _, pc_points = ros_to_pcl2(msg)
        pc = pcl.PointCloud(np.array(pc_points[:,:3], dtype=np.float32))
        pc = self.filter_pcl(pc)
        print(pc.size)

    def generate(self):
        # get clean pointcloud
        # create bounding box
        # sample grasp
        pass


    def filter_pcl(self, pc):
        '''
        Inputs: pc - pointcloud as XYZRGB format using pcl
        '''
        passthrough = pc.make_passthrough_filter()
        passthrough.set_filter_field_name("x")
        passthrough.set_filter_limits(0.278, 0.6)
        pc = passthrough.filter()

        passthrough = pc.make_passthrough_filter()
        passthrough.set_filter_field_name("y")
        passthrough.set_filter_limits(-0.15, 0.15)
        pc = passthrough.filter()	

        passthrough = pc.make_passthrough_filter()
        passthrough.set_filter_field_name("z")
        passthrough.set_filter_limits(0.112, 0.3)
        pc = passthrough.filter()	
        return pc

def main(argv):
    rospy.init_node("grasp_generator")
    node = GraspGenerator()
    # node.run()

    rospy.spin()


if __name__ == "__main__":
    main(sys.argv)

