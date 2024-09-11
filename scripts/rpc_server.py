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

        self.action = 'none'
        self.distance = -1
        self.focus = False
        self.center = [-1, -1, -1]

        self.human_occupancy = [-1, -1, -1, -1]
        self.human_pixels = [-1, -1, -1, -1]
        self.manual = False

    ############################################################
    ####################### Unchanged ##########################
    ############################################################
    def get_human_position(self):
        center = self.asd.read('hpe_to_rpc')['human_position']

        human_position = yarp.Vector(4)
        for i in range(3):
            human_position[i] = center[i]
        human_position[3] = self.asd.read('hpe_to_rpc')['yarp_read_time']
            
        return human_position

    def get_face_position(self):
        center = self.asd.read('focus_to_rpc')['face_point']

        face_position = yarp.Vector(3)
        for i in range(3):
            face_position[i] = center[i]

        return face_position

    def get_poses(self):
        hands = self.asd.read('from_grasp_detection')['hands_root_frame']
        print(hands)
        poses = []

        for h in range(2):
            pose = yarp.Matrix(4, 4)
            for i in range(4):
                for j in range(4):
                    pose[i, j] = hands[i, j, h]

            poses.append(pose)

        return poses

    ############################################################
    ##################### Manual Control #######################
    ############################################################

    def get_distance(self):
        if self.manual:
            distance = self.distance
        else:
            distance = self.asd.read('from_segmentation')['obj_distance']

        return distance

    def is_focused(self):
        if self.manual:
            focus =  self.focus
        else:
            focus = self.asd.read('focus_to_rpc')['focus']

        return focus

    def get_action(self):
        if self.manual:
            action = self.action 
        else:
            action = self.asd.read('ar_to_rpc')['action']

        return action

    def get_center(self):
        if self.manual:
            center = self.center
        else:
            center = self.asd.read('from_segmentation')['obj_center']

        position = yarp.Vector(3)
        for i in range(3):
            position[i] = center[i]

        return position

    def get_human_occupancy(self):
        if self.manual:
            human_occupancy = self.human_occupancy
        else:
            human_occupancy = self.asd.read('hpe_to_rpc')['human_occupancy']

        occ = yarp.Vector(5)
        for i in range(4):
            occ[i] = human_occupancy[i]
        occ[4] = self.asd.read('hpe_to_rpc')['yarp_read_time']

        return occ
    
    def get_human_pixels(self):
        if self.manual:
            human_pixels = self.human_pixels
        else:
            human_pixels = self.asd.read('hpe_to_rpc')['human_pixels']
            yarp_read_time = self.asd.read('hpe_to_rpc')['yarp_read_time']

        occ = yarp.Vector(4)
        for i in range(4):
            occ[i] = human_pixels[i]

        return occ  # TODO RETURN ALSO yarp_read_time


def main(): 

    service = eCubPerceptionServer()
    print('''
    [auto|manual]
    from manual: [wave|shake|release|grasp]
    ''')

    # Simulate main thread here
    while True:
        cmd = input('>>')

        if cmd in ['wave', 'shake', 'release']:
            service.focus = True
            service.action = cmd

            time.sleep(3)

            service.focus = False
            service.action = 'none'

        elif cmd in ['grasp']:
            service.distance = 300
            service.focus = True
            service.center = [0.40270819, -0.00950576,  0.30577174]
            time.sleep(4)
            service.distance = -1
            service.focus = False
            service.center = [-1, -1, -1]

        if cmd == 'manual':
            print('manual control')
            service.manual = True

        if cmd == 'auto':
            print('automatic')
            service.manual = False


if __name__ == '__main__':
    main()
