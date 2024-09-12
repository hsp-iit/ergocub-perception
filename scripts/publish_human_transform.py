import math
from geometry_msgs.msg import TransformStamped
import numpy
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from tf2_ros import TransformBroadcaster
import yarp



class HumanTransformPublisher(Node):
    def __init__(self):
        super().__init__('HumanTfPublisher')
        self.tf_broadcaster = TransformBroadcaster(self)


class DetectedHumanFramePublisher(yarp.BottleCallback):
    def __init__(self):
        super().__init__()
        
        self.tf_node = HumanTransformPublisher()
        self.tf_broadcaster = self.tf_node.tf_broadcaster
    
    def onRead(self, bot, reader):
        print("Port %s received: %s" % (reader.getName(), bot.toString()))
        pelvis_pose_vec = [bot.get(0).asFloat64(), bot.get(1).asFloat64(), bot.get(2).asFloat64()]

        extremes_vec = [bot.get(3).asFloat64(),bot.get(4).asFloat64(),bot.get(5).asFloat64(),bot.get(6).asFloat64()]

        print(pelvis_pose_vec)
        print(extremes_vec)
        t0 = TransformStamped()
        t0.header.stamp.sec = bot.get(7).asInt64()
        t0.header.stamp.nanosec = bot.get(8).asInt64()

        if pelvis_pose_vec[2] != -1:

            final_vec = numpy.zeros(shape = (4,))
            final_vec[1] = -pelvis_pose_vec[0]-extremes_vec[0]
            final_vec[2] = -pelvis_pose_vec[0]-extremes_vec[1]
            final_vec[0] = pelvis_pose_vec[2]
            final_vec[3] = 0.0
        
            t1 = TransformStamped()
            t1.header.stamp = t0.header.stamp
            t1.header.frame_id = 'realsense'
            t1.child_frame_id = 'human_left_frame'
            t1.transform.translation.x = final_vec[0]*2
            t1.transform.translation.y = final_vec[1]*2
            t1.transform.translation.z = final_vec[3]

            t2 = TransformStamped()
            t2.header.stamp = t0.header.stamp
            t2.header.frame_id = 'realsense'
            t2.child_frame_id = 'human_right_frame'
            t2.transform.translation.x = final_vec[0]*2
            t2.transform.translation.y = final_vec[2]*2
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
    human_data_port = yarp.BufferedPortBottle()
    node = DetectedHumanFramePublisher()
    human_data_port.useCallback(node)
    human_data_port.open("/humanEstimateToTf")
    yarp.Network.connect("/humanDataPort","/humanEstimateToTf")


    input("Press ENTER to quit\n")
    

    #rclpy.spin(node)
    human_data_port.interrupt()
    human_data_port.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


