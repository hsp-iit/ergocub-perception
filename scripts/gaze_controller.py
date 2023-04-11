import cv2
import numpy as np

from grasping.utils.misc import pose_to_matrix
import yarp
from utils.concurrency.utils.signals import Signals

from configs.gaze_controller_config import Network, Logging
from utils.logging import setup_logger

setup_logger(**Logging.Logger.Params.to_dict())

class GazeController(Network.node):

    def __init__(self):
        super().__init__(**Network.Args.to_dict())

    def startup(self):
        props = yarp.Property()
        props.put('device', 'gazecontrollerclient')
        props.put('local', '/FollowObject/iKinGazeCtrl')
        props.put('remote', '/iKinGazeCtrl')
        self.driver = yarp.PolyDriver(props)
        self.iface = [getattr(self.driver, 'view' + v)() for v in ['IGazeControl']][0]
        # self.iface.blockEyes(0.0)
        # self.iface.blockNeckRoll(0.0)
        self.iface.setTrackingMode(False)
        self.iface.setNeckTrajTime(.5)

    def loop(self, data):
        point = data['point']
        if point in Signals:
            return

        camera_pose = data['camera_pose']
        if camera_pose in Signals:
            return

        point = np.concatenate([point, np.array([[1]])], axis=1).T
        camera_pose = pose_to_matrix(camera_pose)
        point = camera_pose @ point

        # print(point)
        self.look_at(point[:3])

    def look_at(self, point):
        yarp_vector = yarp.Vector(len(point))
        for i in range(len(point)):
            yarp_vector[i] = point[i].item()

        self.iface.lookAtFixationPoint(yarp_vector)


if __name__ == '__main__':
    gc = GazeController()
    gc.run()
