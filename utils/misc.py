import os

import numpy as np


class Shell:
    def __init__(self, type='cmd'):
        self.process = None
        self.type = type
        self.cmds = []

    def add_pane(self, cmd):
        self.cmds += [cmd]

    def start(self):
        cmd = ''
        for i in range((len(self.cmds) - 1) // 4 + 1):
            cmd += 'wt -M' if i == 0 else ';'

            if len(self.cmds) >= (i * 4) + 1:
                cmd += f' --title ecub-visual-pipeline --suppressApplicationTitle -d {os.getcwd()} --colorScheme "Solarized Dark" {self.cmds[(i * 4) + 0]}'
            if len(self.cmds) >= (i * 4) + 2:
                cmd += f' ;split-pane -V -d {os.getcwd()} --colorScheme "Solarized Dark" {self.cmds[(i * 4) + 1]}' \
                       f' ;move-focus left'
            if len(self.cmds) >= (i * 4) + 3:
                cmd += f' ;split-pane -H -d {os.getcwd()} --colorScheme "Solarized Dark" {self.cmds[(i * 4) + 2]}' \
                       f' ;move-focus right'
            if len(self.cmds) >= (i * 4) + 4:
                cmd += f' ;split-pane -H -d {os.getcwd()} --colorScheme "Solarized Dark" {self.cmds[(i * 4) + 3]}'

        os.system(cmd)

def plot_plane(a, b, c, d):
    xy = (np.random.rand(1000000, 2) - 0.5) * 2
    z = - ((a * xy[..., 0] + b * xy[..., 1] + d) / c)

    xy = xy[(-1 < z) & (z < 1)]
    z = z[(-1 < z) & (z < 1)]

    plane = np.concatenate([xy, z[..., None]], axis=1)

    # aux = PointCloud()
    # aux.points = Vector3dVector(plane)
    return plane