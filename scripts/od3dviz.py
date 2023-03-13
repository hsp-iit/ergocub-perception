import copy
import os
import sys

from grasping.utils.input import RealSense

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

    def startup(self):
        logger.debug('Started gui building')

        canvas = scene.SceneCanvas(keys='interactive')
        canvas.size = 1200, 600
        canvas.show()

        self.grid = canvas.central_widget.add_grid()
        self.canvas = canvas

        vb1 = ViewBox(name='pc')
        self.vb1 = self.grid.add_widget(vb1)

        vb2 = ViewBox(name='pc2')
        self.vb2 = self.grid.add_widget(vb2)

        vb1.camera = scene.TurntableCamera(elevation=0, azimuth=0, distance=1, name='pc')
        vb2.camera = scene.TurntableCamera(elevation=0, azimuth=0, distance=1, name='pc2')

        vb1.camera.link(vb2.camera)

        self.scatter3 = Markers(parent=vb1.scene)
        self.scatter4 = Markers(parent=vb1.scene)
        self.r_hand = scene.XYZAxis(parent=vb1.scene, width=10)
        self.l_hand = scene.XYZAxis(parent=vb1.scene)

        self.box = [SurfacePlot(parent=vb2.scene), SurfacePlot(parent=vb2.scene),
                    SurfacePlot(parent=vb2.scene), SurfacePlot(parent=vb2.scene),
                    SurfacePlot(parent=vb2.scene), SurfacePlot(parent=vb2.scene)]

        self.r_hand_2 = scene.XYZAxis(parent=vb2.scene, width=10)
        self.l_hand_2 = scene.XYZAxis(parent=vb2.scene)

        # self.r_hand = scene.XYZAxis(parent=b3.scene, width=10)
        # axis = scene.XYZAxis(parent=b3.scene, width=10)

    def loop(self, data: dict):
        vis_R1 = Rotation.from_euler('x', -90, degrees=True).as_matrix()
        vis_R2 = Rotation.from_euler('x', 90, degrees=True).as_matrix()

        if data['rgb'] is not None and data['depth'] is not None:
            rgb, depth = data['rgb'], data['depth']
            scene = RealSense.rgb_pointcloud(depth, rgb)
            scene = np.concatenate([np.array(scene.points), np.array(scene.colors)], axis=1)

            idx = np.random.choice(scene.shape[0], 10000, replace=False)
            scene = scene[idx]

            offset = np.array([0, 0.5, 0])

            self.scatter3.set_data(scene[..., :3] @ vis_R1 - offset, edge_color=scene[..., 3:],
                                   face_color=scene[..., 3:])

        # if data['partial'] is not None:
            # self.scatter1.set_data((data['partial'] @ vis_R1) * np.array([1, -1, 1]), edge_color='orange',
            #                        face_color='orange', size=5)

        reconstruction = data['reconstruction']
        if reconstruction is not None:
            if isinstance(reconstruction, int) and reconstruction == -1:
                self.scatter4.parent = None
            else:
                self.vb1.add(self.scatter4)

                idx = np.random.choice(reconstruction.shape[0], 1000, replace=False)
                reconstruction = reconstruction[idx]

                denormalized_pc = (np.block([reconstruction, np.ones([reconstruction.shape[0], 1])]) @
                                   data['transform'])[..., :3]

                self.scatter4.set_data(denormalized_pc @ vis_R2 - offset, edge_color='blue', face_color='blue', size=5)

        hands = data['hands']
        if hands is not None:
            if isinstance(hands, int) and hands == -1:
                self.r_hand.parent = None
                self.l_hand.parent = None
                self.r_hand_2.parent = None
                self.l_hand_2.parent = None
            else:
                self.vb1.add(self.r_hand)
                self.vb1.add(self.l_hand)
                self.vb2.add(self.r_hand_2)
                self.vb2.add(self.l_hand_2)

                right_hand = np.concatenate([np.zeros([1, 3]), np.eye(3)])
                left_hand = np.concatenate([np.zeros([1, 3]), np.eye(3)])

                right_hand = (np.block([right_hand, np.ones([4, 1])]) @ hands[..., 0])[:, :3]
                left_hand = (np.block([left_hand, np.ones([4, 1])]) @ hands[..., 1])[:, :3]

                right_hand = right_hand @ vis_R2
                left_hand = left_hand @ vis_R2

                self.r_hand.set_data(right_hand[[0, 1, 0, 2, 0, 3]] - offset)
                self.l_hand.set_data(left_hand[[0, 1, 0, 2, 0, 3]] - offset)

                self.r_hand_2.set_data(right_hand[[0, 1, 0, 2, 0, 3]] - offset)
                self.l_hand_2.set_data(left_hand[[0, 1, 0, 2, 0, 3]] - offset)

        points = data['vertices']
        if points is not None:
            if isinstance(points, int) and points == -1:
                for b in self.box: b.parent = None
            else:
                for b in self.box: self.vb2.add(b)

                points = points @ vis_R1 - offset
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

# import copy
# import time
# from multiprocessing import Queue, Process
# from multiprocessing.managers import BaseManager
#
# import cv2
# from vispy import app, scene, visuals
# from vispy.scene import ViewBox
# from vispy.scene.visuals import Text, Image, Markers
# import numpy as np
# import math
# import sys
# from loguru import logger
# from scipy.spatial.transform import Rotation
#
# from grasping.modules.utils.misc import draw_mask
# from gui.misc import project_hands
# from human.utils.params import RealSenseIntrinsics
#
#
# class Visualizer(Process):
#
#     def __init__(self):
#         super().__init__()
#         self.name = 'Visualizer'
#
#         self.widgets = []
#         self.builders = {}
#         self.cameras = {}
#         self.last_widget = None
#         self.fps_s = []
#         self.last_time = 0
#
#     def run(self):
#         self.build_gui()
#         app.run()
#
#     def remove_all_widgets(self):
#         for wg, _ in self.widgets:
#             self.grid.remove_widget(wg)
#             del wg
#
#     def add_all_widgets(self, name='', args=None):
#         for build in self.builders:
#             if build == name:
#                 self.builders[build](**args)
#             else:
#                 self.builders[build]()
#
#     def highlight(self, event):
#         if event.type == 'mouse_press' and event.button == 2:
#             self.canvas.central_widget.children[0].parent = None
#             self.grid = self.canvas.central_widget.add_grid()
#             self.add_all_widgets(event.visual.name, {'row': 0, 'col': 1, 'row_span': 4, 'col_span': 2})
#
#     def build_gui(self):
#         logger.debug('Started gui building')
#         self.show = True
#
#         self._timer = app.Timer('auto', connect=self.on_timer, start=True)
#
#         canvas = scene.SceneCanvas(keys='interactive')
#         canvas.size = 1200, 600
#         canvas.show()
#
#         # This is the top-level widget that will hold three ViewBoxes, which will
#         # be automatically resized whenever the grid is resized.
#         self.grid = canvas.central_widget.add_grid()
#         self.canvas = canvas
#         self.input_text = '>'
#
#         ######################
#         ##### View Boxes #####
#         ######################
#         ######################
#         ###### Grasping ######
#         ######################
#
#         # Point Cloud 1
#         def build_pc1(row=2, col=0, row_span=1, col_span=1):
#             b3 = ViewBox(name='pc1')
#             b3 = self.grid.add_widget(b3, row=row, col=col, row_span=row_span, col_span=col_span)
#
#             if 'pc1' not in self.cameras:
#                 self.cameras['pc1'] = scene.TurntableCamera(elevation=0, azimuth=0, distance=1, name='pc1')
#
#             b3.camera = self.cameras['pc1']
#             b3.border_color = (0.5, 0.5, 0.5, 1)
#             self.scatter1 = Markers(parent=b3.scene)
#             self.scatter2 = Markers(parent=b3.scene)
#
#             # self.r_hand = scene.XYZAxis(parent=b3.scene, width=10)
#             # axis = scene.XYZAxis(parent=b3.scene, width=10)
#
#             b3.events.mouse_press.connect(self.highlight)
#             self.widgets.append([b3, {'row': 2, 'col': 0}])
#
#         self.builders['pc1'] = build_pc1
#
#         # Point Cloud 2
#         def build_pc2(row=3, col=0, row_span=1, col_span=1):
#             b4 = ViewBox(name='pc2')
#             b4 = self.grid.add_widget(b4, row=row, col=col, row_span=row_span, col_span=col_span)
#
#             if 'pc2' not in self.cameras:
#                 self.cameras['pc2'] = scene.TurntableCamera(elevation=0, azimuth=0, distance=0.5, name='pc2')
#
#             b4.camera = self.cameras['pc2']
#             b4.border_color = (0.5, 0.5, 0.5, 1)
#             self.scatter3 = Markers(parent=b4.scene)
#             self.scatter4 = Markers(parent=b4.scene)
#             self.r_hand = scene.XYZAxis(parent=b4.scene, width=10)
#             self.l_hand = scene.XYZAxis(parent=b4.scene)
#             # scene.XYZAxis(parent=b4.scene, width=10)
#             b4.events.mouse_press.connect(self.highlight)
#             self.widgets.append([b4, {'row': 3, 'col': 0}])
#             self.test = b4
#
#         self.builders['pc2'] = build_pc2
#
#         self.add_all_widgets()
#
#         logger.debug('Gui built successfully')
#
#     def on_timer(self, _):
#
#         if not self.show:
#             return
#
#         start = time.perf_counter()
#         ##################
#         #### Grasping ####
#         ##################
#         if not self.grasping_in.empty():
#             data = self.grasping_in.get()
#
#             rgb_mask = draw_mask(data['rgb'], data['mask'])
#
#             depth_image = cv2.applyColorMap(cv2.convertScaleAbs(data['depth'], alpha=0.03), cv2.COLORMAP_JET)
#             depth_mask = draw_mask(depth_image, data['mask'])
#
#             font = cv2.FONT_ITALIC
#             bottomLeftCornerOfText = (10, 30)
#             fontScale = 1
#             fontColor = (255, 255, 255)
#             thickness = 1
#             lineType = 2
#
#             cv2.putText(depth_mask, 'Distance: {:.2f}'.format(data["distance"] / 1000),
#                         bottomLeftCornerOfText,
#                         font,
#                         fontScale,
#                         fontColor,
#                         thickness,
#                         lineType)
#
#             # The view in pc2 is denormalized so it needs a different rotation
#             vis_R1 = Rotation.from_euler('x', -90, degrees=True).as_matrix()
#             vis_R2 = Rotation.from_euler('x', 90, degrees=True).as_matrix()
#
#             self.image1.set_data(rgb_mask[::-1, ..., ::-1])
#             self.image2.set_data(depth_mask[::-1, ...])
#
#             if data['partial'] is not None:
#                 self.scatter1.set_data((data['partial'] @ vis_R1) * np.array([1, -1, 1]), edge_color='orange',
#                                        face_color='orange', size=5)
#
#                 if data['reconstruction'] is not None:
#                     self.scatter2.set_data((data['reconstruction'] @ vis_R1) * np.array([1, -1, 1]), edge_color='blue',
#                                            face_color='blue', size=5)
#
#                     self.scatter3.set_data(data['scene'][..., :3] @ vis_R2, edge_color=data['scene'][..., 3:],
#                                            face_color=data['scene'][..., 3:])
#
#                     denormalized_pc = (np.block(
#                         [data['reconstruction'], np.ones([data['reconstruction'].shape[0], 1])]) @ data['transform'])[
#                                       ..., :3]
#                     self.scatter4.set_data(denormalized_pc @ vis_R2, edge_color='blue', face_color='blue', size=5)
#
#                     hands = data['hands']
#                     if hands is not None:
#                         right_hand = np.concatenate([np.zeros([1, 3]), np.eye(3)])
#                         left_hand = np.concatenate([np.zeros([1, 3]), np.eye(3)])
#
#                         right_hand = (np.block([right_hand, np.ones([4, 1])]) @ hands['right'])[:, :3]
#                         left_hand = (np.block([left_hand, np.ones([4, 1])]) @ hands['left'])[:, :3]
#
#                         right_hand = right_hand @ vis_R2
#                         left_hand = left_hand @ vis_R2
#
#                         self.r_hand.set_data(right_hand[[0, 1, 0, 2, 0, 3]])
#                         self.l_hand.set_data(left_hand[[0, 1, 0, 2, 0, 3]])
#
#             # text = '\n'.join([f'{key}: {value:.2f} fps' for (key, value) in data['fps'].items()])
#             # self.avg_fps.text = text
#
#         ##################
#         ##### Human ######
#         ##################
#         if not self.human_in.empty():
#             data = self.human_in.get()
#
#             if not data:
#                 return
#
#             edges = data["edges"]
#             pose = data["pose"]
#             img = data["img"]
#             focus = data["focus"]
#             action = data["action"]
#             distance = data["distance"]
#             human_bbox = data["human_bbox"]
#             face = data["face"]
#             box_center_3d = data["box_center"]
#             pose2d = data["pose2d"]
#             grasping = data['grasping']
#
#             # POSE
#             if pose is not None:
#                 for i, edge in enumerate(edges):
#                     self.lines[i].set_data((pose[[edge[0], edge[1]]]), color='purple')
#             else:
#                 for i in list(range(len(self.lines))):
#                     self.lines[i].set_data(color='grey')
#
#             # GAZE
#             if face is not None:  # TODO FIX
#                 # head_pose = np.linalg.inv(face.normalizing_rot.as_matrix()) @ face.head_pose_rot.as_rotvec()
#                 # second_point = face.gaze_vector if  else head_pose
#                 second_point = face.gaze_vector if face.is_close else face.head_pose_rot.as_rotvec()  # TODO REMOVE DEbuG
#                 gaze = np.concatenate((face.head_position[None, ...],
#                                        (face.head_position + second_point)[None, ...]))
#                 self.focus_vector.set_data(gaze, color='orange')
#
#             # BOX
#             box_center_2d = None
#             if box_center_3d is not None and np.any(box_center_3d):
#                 # Draw box with human
#                 self.box.set_data(box_center_3d, edge_color='orange', face_color='orange', size=50)
#                 # Draw projection of box
#                 box_center = box_center_3d
#                 box_center_2d = RealSenseIntrinsics().K @ box_center.T
#                 box_center_2d = box_center_2d[0:2] / box_center_2d[2, :]
#                 box_center_2d = np.round(box_center_2d, 0).astype(int).squeeze()
#             else:
#                 self.box.set_data(np.array([0, 0, 0])[None, ...])
#
#             # IMAGE
#             if img is not None:
#                 if human_bbox is not None:
#                     x1, x2, y1, y2 = human_bbox
#                     img = cv2.rectangle(img,
#                                         (x1, y1), (x2, y2), (0, 0, 255), 1).astype(np.uint8)
#                 if face is not None:
#                     x1, y1, x2, y2 = face.bbox.reshape(-1)
#                     img = cv2.rectangle(img,
#                                         (x1, y1), (x2, y2), (255, 0, 0), 1).astype(np.uint8)
#                 if box_center_2d is not None:
#                     img = cv2.circle(img, box_center_2d, 5, (0, 255, 0)).astype(np.uint8)
#                 if pose2d is not None:
#                     for edge in edges:
#                         c1 = 0 < pose2d[edge[0]][0] < 640 and 0 < pose2d[edge[0]][1] < 480
#                         c2 = 0 < pose2d[edge[1]][0] < 640 and 0 < pose2d[edge[1]][1] < 480
#                         if c1 and c2:
#                             img = cv2.line(img, pose2d[edge[0]], pose2d[edge[1]], (255, 0, 255), 3, cv2.LINE_AA)
#
#                 if grasping['mask'] is not None:
#                     img = draw_mask(img, grasping['mask'])
#
#                 if grasping['hands'] is not None:
#                     hands = grasping['hands']
#                     img = project_hands(img, hands['right'], hands['left'])
#
#                 img = cv2.flip(img, 0)
#
#                 self.image.set_data(img)
#
#             # INFO
#             if focus is not None:
#                 if focus:
#                     self.focus.text = "FOCUS"
#                     self.focus.color = "green"
#                 else:
#                     self.focus.text = "NOT FOCUS"
#                     self.focus.color = "red"
#
#             # FPS
#             self.fps_s.append(1 / (time.time() - self.last_time))
#             self.last_time = time.time()
#             fps_s = self.fps_s[-10:]
#             fps = sum(fps_s) / len(fps_s)
#             if fps is not None:
#                 self.fps.text = "FPS: {:.2f}".format(fps)
#
#             # Distance
#             if distance is not None:
#                 self.distance.text = "DIST: {:.2f}m".format(distance)
#
#             # Actions
#             self.action.text = str(action)
#             self.action.pos = 0.5, 0.5
#             self.action.color = "green"
#
#     def on_draw(self, event):
#         pass
#
#
# if __name__ == '__main__':
#     def grasping():
#         BaseManager.register('get_queue')
#         manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
#         manager.connect()
#
#         grasping_in = manager.get_queue('vis_in_grasping')
#
#         while True:
#             res1 = np.random.randint(0, 255, [480, 640, 3], dtype=np.uint8)
#             res2 = np.random.randint(0, 255, [480, 640, 3], dtype=np.uint8)
#
#             pc1 = np.random.rand(1000, 3)
#             pc2 = np.random.rand(1000, 3)
#             grasping_in.put({'res1': res1, 'res2': res2, 'pc1': pc1, 'pc2': pc2})
#
#
#     def human():
#         BaseManager.register('get_queue')
#         manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
#         manager.connect()
#
#         human_in = manager.get_queue('vis_in_human')
#         human_out = manager.get_queue('vis_out_human')
#
#         while True:
#             elements = [{"img": np.random.random((640, 480, 3)),
#                          "pose": np.random.random((30, 3)),
#                          "edges": [(1, 2)],
#                          "fps": 0,
#                          "focus": False,
#                          "action": {},
#                          "distance": 0,  # TODO fix
#                          "box": [1, 2, 3, 4]
#                          }]
#             human_in.put(elements)
#
#
#     viz = Visualizer()
#     viz.run()
#
#     # # Testing
#     # BaseManager.register('get_queue')
#     # manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
#     # manager.connect()
#     #
#     # queue = manager.get_queue('vis_in_grasping')
#     # while True:
#     #     data = {'rgb': np.random.randint(0, 255, [480, 640, 3], dtype=np.uint8), 'depth': np.random.rand(480, 640, 3),
#     #             'mask': np.random.randint(0, 1, [480, 640], dtype=np.uint8), 'distance': 1.5,
#     #             'mean': np.array([0, 0, 0]), 'var': np.array([1, 1, 1]), 'partial': np.random.rand(2048, 3),
#     #             'reconstruction': np.random.rand(2048, 3), 'scene': np.random.rand(2048, 6), 'poses': None, 'fps': {'test': 1}}
#     #
#     #     queue.put(data)
