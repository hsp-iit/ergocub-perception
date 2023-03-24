import copy
import os
import sys

from grasping.utils.input import RealSense
from utils.concurrency.utils.signals import Signals

sys.path.append('/robotology-superbuild/build/install/lib/python3.8/site-packages')

from pathlib import Path

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
from vispy import scene, app
from vispy.scene import ViewBox, Markers, SurfacePlot

sys.path.insert(0, Path(__file__).parent.parent.as_posix())
from utils.logging import setup_logger
from configs.od3dviz_config import Logging, Network

setup_logger(level=Logging.level)


@logger.catch(reraise=True)
class ObjectDetection3DVisualizer(Network.node):
    def __init__(self):
        super().__init__(**Network.Args.to_dict())
        self.vis_R1 = Rotation.from_euler('x', -90, degrees=True).as_matrix()
        self.vis_R2 = Rotation.from_euler('x', 90, degrees=True).as_matrix()

        self.vb1, self.vb2 = None, None

    def startup(self):
        logger.debug('Started gui building')

        canvas = scene.SceneCanvas(keys='interactive')
        canvas.size = 1200, 600
        canvas.show()

        grid = canvas.central_widget.add_grid()

        #  Left Viewbox
        vb1 = ViewBox()
        self.vb1 = grid.add_widget(vb1)

        #  Right Viewbox
        vb2 = ViewBox()
        self.vb2 = grid.add_widget(vb2)

        #  Creating and linking the cameras
        vb1.camera = scene.TurntableCamera(elevation=0, azimuth=0, distance=1)
        vb2.camera = scene.TurntableCamera(elevation=0, azimuth=0, distance=1)
        vb1.camera.link(vb2.camera)

        self.scene_scatter = Markers(parent=vb1.scene)
        self.center = Markers(size=10, parent=vb1.scene)
        self.reconstruction_scatter = Markers(parent=vb1.scene)
        self.partial_scatter = Markers(parent=vb1.scene)

        # Adding the same element in two viewboxes is not supported
        # You need to create a different object and add it to the second viewport
        # https://github.com/vispy/vispy/issues/1992
        self.r_hand = scene.XYZAxis(parent=vb2.scene, width=10)
        self.l_hand = scene.XYZAxis(parent=vb2.scene, width=10)

        self.box = [SurfacePlot(parent=vb2.scene), SurfacePlot(parent=vb2.scene),
                    SurfacePlot(parent=vb2.scene), SurfacePlot(parent=vb2.scene),
                    SurfacePlot(parent=vb2.scene), SurfacePlot(parent=vb2.scene)]

    def loop(self, data: dict):

        offset = np.array([0, 0.5, 0])

        center = data.get('center', Signals.MISSING_VALUE)
        if center is Signals.MISSING_VALUE:
            pass
        elif center is Signals.NOT_OBSERVED:
            self.partial_scatter.parent = None
        else:
            self.vb1.add(self.partial_scatter)

            self.center.set_data((center @ self.vis_R1 - offset), edge_color='orange',
                                          face_color='green', size=50)

        rgb, depth = data.get('rgb', Signals.MISSING_VALUE), data.get('depth', Signals.MISSING_VALUE)
        if rgb is Signals.MISSING_VALUE or depth is Signals.MISSING_VALUE:
            pass
        elif rgb is Signals.NOT_OBSERVED or depth is Signals.NOT_OBSERVED:
            self.scene_scatter.parent = None
        else:
            self.vb1.add(self.scene_scatter)

            rgb, depth = data['rgb'], data['depth']
            scene = RealSense.rgb_pointcloud(depth, rgb)
            scene = np.concatenate([np.array(scene.points), np.array(scene.colors)], axis=1)

            idx = np.random.choice(scene.shape[0], 10000, replace=False)
            scene = scene[idx]

            self.scene_scatter.set_data(scene[..., :3] @ self.vis_R1 - offset, edge_color=scene[..., 3:],
                                        face_color=scene[..., 3:])

        transformation = data.get('transform', Signals.MISSING_VALUE)

        partial = data.get('partial', Signals.MISSING_VALUE)
        if partial is Signals.MISSING_VALUE:
            pass
        elif partial is Signals.NOT_OBSERVED:
            self.partial_scatter.parent = None
        else:
            self.vb1.add(self.partial_scatter)

            idx = np.random.choice(partial.shape[0], 1000, replace=False)
            partial = partial[idx]

            if transformation not in Signals:
                partial = (np.block([partial, np.ones([partial.shape[0], 1])]) @
                           transformation)[..., :3]
            self.partial_scatter.set_data((partial @ self.vis_R2 - offset), edge_color='orange',
                                          face_color='orange', size=5)

        reconstruction = data['reconstruction']
        if reconstruction is Signals.MISSING_VALUE:
            pass
        elif reconstruction is Signals.NOT_OBSERVED:
            self.reconstruction_scatter.parent = None
        else:
            self.vb1.add(self.reconstruction_scatter)

            idx = np.random.choice(reconstruction.shape[0], 1000, replace=False)
            reconstruction = reconstruction[idx]

            if transformation not in Signals:
                reconstruction = (np.block([reconstruction, np.ones([reconstruction.shape[0], 1])]) @
                                  transformation)[..., :3]

            self.reconstruction_scatter.set_data(reconstruction @ self.vis_R2 - offset, edge_color='blue',
                                                 face_color='blue', size=5)

        hands = data.get('hands', Signals.MISSING_VALUE)
        if hands is Signals.MISSING_VALUE:
            pass
        elif hands is Signals.NOT_OBSERVED:
            self.r_hand.parent = None
            self.l_hand.parent = None
        else:
            self.vb2.add(self.r_hand)
            self.vb2.add(self.l_hand)

            right_hand = np.concatenate([np.zeros([1, 3]), np.eye(3)])
            left_hand = np.concatenate([np.zeros([1, 3]), np.eye(3)])

            right_hand = (np.block([right_hand, np.ones([4, 1])]) @ hands[..., 0])[:, :3]
            left_hand = (np.block([left_hand, np.ones([4, 1])]) @ hands[..., 1])[:, :3]

            right_hand = right_hand @ self.vis_R2
            left_hand = left_hand @ self.vis_R2

            self.r_hand.set_data(right_hand[[0, 1, 0, 2, 0, 3]] - offset)
            self.l_hand.set_data(left_hand[[0, 1, 0, 2, 0, 3]] - offset)

        points = data.get('vertices', Signals.MISSING_VALUE)
        if points is Signals.MISSING_VALUE:
            pass
        elif points is Signals.NOT_OBSERVED:
            for b in self.box:
                b.parent = None
        else:
            for b in self.box:
                self.vb2.add(b)

            points = points @ self.vis_R1 - offset
            for i in range(4):
                points = np.concatenate([points, points[(0 + i) % 8][None],
                                         points[(1 + i) % 8][None],
                                         points[(4 + i) % 8][None],
                                         points[(5 + i) % 8][None], ])

            planes = points.reshape(6, 4, 3)

            for i, points in enumerate(planes):
                if i in [0, 1]:
                    aux = copy.deepcopy(points)
                    points[1], points[3], points[0] = aux[0], aux[1], aux[3]
                if i == 5:
                    aux = copy.deepcopy(points)
                    points[3], points[0], points[1] = aux[0], aux[1], aux[3]

                x, y, z = points.T
                x, y, z = x.reshape((2, 2)), y.reshape((2, 2)), z.reshape((2, 2))

                self.box[i].set_data(x=x, y=y, z=z)

        app.process_events()
        return {}


if __name__ == '__main__':
    source = ObjectDetection3DVisualizer()
    source.run()
