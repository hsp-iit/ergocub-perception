import yarp
from grasping.utils.misc import pose_to_matrix
from utils.concurrency.generic_node import GenericNode
from utils.concurrency.utils.signals import Signals

from configs.segmentation_config import Segmentator, Network, Logging
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
        self.iface.blockEyes(5.0)
        self.iface.blockNeckRoll(0.0)
        self.iface.setTrackingMode(False)

    def loop(self, data):
        point = data['point']
        if point is Signals.MISSING_VALUE:
            return

        camera_pose = data['camera_pose']
        if camera_pose is Signals.MISSING_VALUE:
            return

        camera_pose = pose_to_matrix(camera_pose)
        point = point @ camera_pose

        self.look_at(point)

    def look_at(self, point):
        yarp_vector = yarp.Vector(len(point))
        for i in range(len(point)):
            yarp_vector[i] = point[i]

        self.iface.lookAtFixationPoint(yarp_vector)


if __name__ == '__main__':
    gc = GazeController()
    gc.run()
