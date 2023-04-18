from ecub_perception import eCubPerceptionInterface
import yarp
import time

from configs.rpc_config import Logging, Network, RPC

class eCubPerceptionServer(eCubPerceptionInterface, Network.node):

    def __init__(self):
        yarp.Network.init()
        
        log_prefix = '[perception-python-server::eCubPerceptionServer]'

        super(eCubPerceptionServer, self).__init__(**Network.Args.to_dict())

        self.rpc_server_ = yarp.RpcServer()
        if not self.rpc_server_.open(RPC.port_name):
            print(log_prefix + " Error: cannot open RPC port.")
            exit(1)

        super(eCubPerceptionServer, self).yarp().attachAsServer(self.rpc_server_)


    def get_poses(self):
        hands = self.read('from_grasp_detection')['hands']
        poses = []

        for h in range(2):
            pose = yarp.Matrix(4, 4)
            for i in range(4):
                for j in range(4):
                    pose_l[i, j] = hands[i, j, h]

            poses.append(pose)

        return poses


    def get_center(self):
        center = self.read('from_segmentation')['center']

        position = yarp.Vector(3)
        for i in range(3):
            position[i] = center[i]

        return position


    def get_distance(self):
        distance = self.read('from_segmentation')['box_distance']

        return distance



def main(): 

    service = eCubPerceptionServer()

    # Simulate main thread here
    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()
