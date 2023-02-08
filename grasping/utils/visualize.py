import atexit
import gc
import io
import os
import pickle
import socket
import sys
import time
from functools import partial
from multiprocessing import Queue
from pathlib import Path

import cv2
import numpy as np

import open3d as o3d
#Creatreconstructionject.
from open3d.cpu.pybind.geometry import PointCloud, TriangleMesh
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import Visualizer

from grasping.modules.utils.input import RealSense
from grasping.modules.utils.misc import draw_mask

print = partial(print, flush=True)



class DebugVisualizer:
    def __init__(self):
        vis1 = Visualizer()
        vis1.create_window('Partial')
        vis2 = Visualizer()
        vis2.create_window('Scene')

        self.vis1 = vis1
        self.vis2 = vis2
        self.scene_pcd = PointCloud()
        self.part_pcd = PointCloud()
        self.pred_pcd = PointCloud()
        self.coords_mesh = [TriangleMesh.create_coordinate_frame(size=0.1) for _ in range(2)]

        self.render_setup = False

    def update(self, image, depth, mask, partial, reconstruction, poses, mean, var):
        res1, res2 = draw_mask(image, mask)
        cv2.imshow('res1', res1)
        cv2.imshow('res2', res2)

        cv2.waitKey(1)

        scene_pc = RealSense.rgb_pointcloud(depth, image)

        if reconstruction is not None:

            if poses is not None:
                poses[0] = (poses[0] * (var * 2) + mean)
                poses[2] = (poses[2] * (var * 2) + mean)
                best_centers = (poses[0], poses[2])
                best_rots = (poses[1], poses[3])
                size = 0.1
            else:
                best_centers = (np.zeros([3]), np.zeros([3]))
                best_rots = (np.zeros([3, 3]), np.zeros([3, 3]))
                size = 0.01

            # Orient poses
            for c, R, coord_mesh in zip(best_centers, best_rots, self.coords_mesh):
                coord_mesh_ = TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0]) \
                    .rotate(R, center=[0, 0, 0]).translate(c, relative=False)

                # Update mesh
                coord_mesh.triangles = coord_mesh_.triangles
                coord_mesh.vertices = coord_mesh_.vertices

            part_pc = PointCloud()
            part_pc.points = Vector3dVector(partial)  # + [0, 0, 1]
            part_pc.paint_uniform_color([0, 1, 0])
            pred_pc = PointCloud()
            pred_pc.points = Vector3dVector(reconstruction)
            pred_pc.paint_uniform_color([1, 0, 0])

            self.scene_pcd.clear()
            self.part_pcd.clear()
            self.pred_pcd.clear()

            self.scene_pcd += scene_pc
            self.part_pcd += part_pc
            self.pred_pcd += pred_pc



            if not self.render_setup:
                self.vis2.add_geometry(self.scene_pcd)
                self.vis1.add_geometry(self.part_pcd)
                self.vis1.add_geometry(self.pred_pcd)
                for pose in self.coords_mesh:
                    self.vis2.add_geometry(pose)

                render_setup = True

            self.vis2.update_geometry(self.scene_pcd)
            self.vis1.update_geometry(self.part_pcd)
            self.vis1.update_geometry(self.pred_pcd)
            for pose in self.coords_mesh:
                self.vis2.update_geometry(pose)

            self.vis1.poll_events()
            self.vis1.update_renderer()

            self.vis2.poll_events()
            self.vis2.update_renderer()

        # if data['poses'] is not None:
        #     poses = data['poses']
        #     centers = (poses[0], poses[2])
        #     rotations = (poses[1], poses[3])
        #     size = 0.1
        # else:
        #     centers, rotations, size = (np.zeros([3]), np.zeros([3])), (np.zeros([3, 3]), np.zeros([3, 3])), 0.01
        #
        # names = ['right', 'left']
        # for r, c, n in zip(rotations, centers, names):
        #     hand = TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0]) \
        #         .rotate(r, center=[0, 0, 0]).translate(c, relative=False)
        #
        #     update_mesh(vis2, (n, hand))

        # vis2.poll_events()
        # vis2.update_renderer()


