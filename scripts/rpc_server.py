from ecub_perception import eCubPerceptionInterface
import yarp
import time
from configs.rpc_config import Logging, Network, RPC

class eCubPerceptionServer(eCubPerceptionInterface):

    def __init__(self):
        yarp.Network.init()
        
        log_prefix = '[perception-python-server::eCubPerceptionServer]'

        super(eCubPerceptionServer, self).__init__()

        self.rpc_server_ = yarp.RpcServer()
        if not self.rpc_server_.open(RPC.port_name):
            print(log_prefix + " Error: cannot open RPC port.")
            exit(1)

        super(eCubPerceptionServer, self).yarp().attachAsServer(self.rpc_server_)


        self.asd = Network.node(**Network.Args.to_dict())


    def get_poses(self):
        hands = self.asd.read('from_grasp_detection')['hands']
        poses = []

        for h in range(2):
            pose = yarp.Matrix(4, 4)
            for i in range(4):
                for j in range(4):
                    pose[i, j] = hands[i, j, h]

            poses.append(pose)

        return poses


    def get_center(self):
        center = self.asd.read('from_segmentation')['obj_center']

        position = yarp.Vector(3)
        for i in range(3):
            position[i] = center[i]

        return position


    def get_distance(self):
        distance = self.asd.read('from_segmentation')['obj_distance']

        return distance

    def is_focused(self):
        return self.asd.read('focus_to_rpc')['focus']

    def get_face_position(self):
        center = self.asd.read('focus_to_rpc')['face_point']

        face_position = yarp.Vector(3)
        for i in range(3):
            face_position[i] = center[i]

        return face_position

    def get_action(self):
        action = self.asd.read('ar_to_rpc')['action']

        return action



def main(): 

    service = eCubPerceptionServer()

    # Simulate main thread here
    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()
