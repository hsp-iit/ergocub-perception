import math
from geometry_msgs.msg import TransformStamped
import numpy
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from ecub_perception import eCubPerceptionInterface
import yarp
import time
#import tcpi
#from configs.rpc_config import Logging, Network, RPC



class DetectedHumanFramePublisher(Node):
    def __init__(self):
        super().__init__('detected_human_frame_publisher')
        
        self.tf_broadcaster = TransformBroadcaster(self)
        self.rpc_client = yarp.RpcClient()
        if not self.rpc_client.open('/eCubPerception/rpc:o'):
            print(" Error: cannot open RPC port.")
            exit(1)
        self.service = eCubPerceptionInterface()
        self.service.yarp().attachAsClient(self.rpc_client)
        self.service.yarp().setStreamingMode(False)

        self.timer = self.create_timer(0.1, self.pub_human_transform)
    
    def pub_human_transform(self):

        pelvis_pose = self.service.get_human_position()
        extremes = self.service.get_human_occupancy()

        print("Pelvis: " + str(yarp_vector_to_numpy(pelvis_pose)))
        print("Extremes: " + str(yarp_vector_to_numpy(extremes)))
        pelvis_pose_vec = yarp_vector_to_numpy(pelvis_pose)
        extremes_vec = yarp_vector_to_numpy(extremes)
        if pelvis_pose_vec.any() and pelvis_pose_vec[2] != -1:
            final_vec = numpy.zeros(shape = (4,))
            final_vec[1] = -pelvis_pose_vec[0]-extremes_vec[0]
            final_vec[2] = -pelvis_pose_vec[0]-extremes_vec[1]
            final_vec[0] = pelvis_pose_vec[2]
            final_vec[3] = 0.0
        
            t1 = TransformStamped()
            t1.header.stamp = self.get_clock().now().to_msg()
            t1.header.frame_id = 'head_laser_frame'
            t1.child_frame_id = 'human_left_frame'
            t1.transform.translation.x = final_vec[0]
            t1.transform.translation.y = final_vec[1]
            t1.transform.translation.z = final_vec[3]

            t2 = TransformStamped()
            t2.header.stamp = self.get_clock().now().to_msg()
            t2.header.frame_id = 'head_laser_frame'
            t2.child_frame_id = 'human_right_frame'
            t2.transform.translation.x = final_vec[0]
            t2.transform.translation.y = final_vec[2]
            t2.transform.translation.z = final_vec[3]

            self.tf_broadcaster.sendTransform(t1)
            self.tf_broadcaster.sendTransform(t2)


def yarp_matrix_to_numpy(yarp_matrix):
    matrix = numpy.zeros(shape = (yarp_matrix.rows(), yarp_matrix.cols()))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = yarp_matrix[i, j]

    return matrix


def yarp_vector_to_numpy(yarp_vector):

    vector = numpy.zeros(shape = (yarp_vector.size(),))

    for i in range(vector.shape[0]):
        vector[i] = yarp_vector[i]

    return vector


def main():
    rclpy.init()
    yarp.Network.init()

    node = DetectedHumanFramePublisher()
  
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()


