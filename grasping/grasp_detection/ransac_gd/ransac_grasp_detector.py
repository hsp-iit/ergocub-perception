import copy

import numpy as np
from vispy import scene
from vispy.scene import Markers
from vispy.visuals import LineVisual

from ...utils.misc import plot_plane


# TODO NOTE
# In order to see the visualization correctly, one should stay in the coordinate frame, looking towards +z with +x
# facing towards -x and +y facing towards -y
from ...utils.misc import plot_line


class RansacGraspDetectorTRT:

    def __call__(self, box):
        """
        Get a complete point cloud and return the point cloud with good grasping spot
        Args:
            up:
            var:
            mean:
            debug: if to draw each iteration with open3d.draw_geometries
            dist: distance of points in RANSAC algorithm to belong to plane
            iterations: number of iterations to do in segment plane
            n_points: number of points to use in each iteration of segment plane
            pc: Complete Point Cloud
        Returns:
            poses: first best center, first best normal, second-best center, second-best normal
        """

        planes, face_points = box

        # Take the normals of the planes and the mean of the extracted points
        normals = planes[..., :3]
        centers = np.array([np.mean(surface, axis=0) for surface in face_points])

        right, left, front, back, top, bottom = order_planes(centers, planes)

        if len(np.unique([right, left, front, back, top, bottom])) != 6:
            return None

        # Rotation
        align_hand = True  # Hand aligned with the front normal of the box

        right_normal, front_normal = normals[[right, front]]
        x_axis, y_axis, z_axis = np.eye(3)

        cst1 = {'from': z_axis, 'to': -right_normal}
        cst2 = {'from': x_axis, 'to': -front_normal} if align_hand else {'from': y_axis, 'to': -y_axis}

        # the z axis is rotated to the normal of the left plane
        rotation = origin2pose([cst1, cst2])

        # Translation
        x = 0.15
        y = 0.5

        lines = []
        for idx in [front, top, back, bottom]:
            lines += [plane_plane(planes[right], planes[idx])]
        front_line, top_line, back_line, bottom_line = lines

        right_vertices = []
        for line, plane in [[front_line, planes[top]], [top_line, planes[back]],
                            [back_line, planes[bottom]], [bottom_line, planes[front]]]:
            right_vertices += [line_plane(line, plane)]
        top_left, top_right, bottom_right, bottom_left = right_vertices

        # This way (1, 1) is the top-right vertex of the face
        u, v, = x * np.linalg.norm(bottom_right - bottom_left), y * np.linalg.norm(top_left - bottom_left)

        r = (bottom_right - bottom_left) / np.linalg.norm(bottom_right - bottom_left)
        s = (top_left - bottom_left) / np.linalg.norm(top_left - bottom_left)

        # + bottom left is for translating the point to the correct position
        # the scalar product correct for the rotation of the new point
        right_center = (u * r + v * s) + bottom_left

        # u = bottom_left + x * np.linalg.norm(bottom_right - bottom_left)
        # v = bottom_left + y * np.linalg.norm(top_left - bottom_left)
        # TODO Remove
        # scatter3 = Markers(pos=points, face_color=(1, 0, 0, .5))
        # axis = scene.XYZAxis(width=5)
        #
        # r_hand = scene.XYZAxis(width=10)
        # r_hand._pos = (rotation @ r_hand._pos.T).T + right_center

        # normal = scene.Line(pos=np.array([[0, 0, 0], right_normal]), width=5, color=(1, 1, 0, 1))
        # prediction = scene.Line(pos=np.array([[0, 0, 0], z_axis]) @ R1.T, width=5, color=(1, 0, 1, 1))

        # draw_geometries(
        #     [scatter3, axis, r_hand] + [Markers(pos=v[None, ...], face_color=c + (1,), size=20)
        #                         for c, v in
        #                         zip([(1, 0, 0)], [right_center])])

        # right_center = centers[right]
        left_center = line_plane([right_center, right_normal], planes[left])

        left_vertices = right_vertices - (np.linalg.norm(left_center - right_center)) * right_normal
        vertices = np.concatenate([right_vertices, left_vertices])
        return [right_center, rotation, left_center, copy.deepcopy(rotation), planes, lines, vertices]


def origin2pose(constraints):  # , c1, c2, points, viz
    """
        Return the transformation from the origin frame to the
        hand pose satisfying the given constraints.

        The second rotation is around the axis that was adjusted by the first one.
        The source of the first constraint must be a vector that ended up on the plane after the first rotation.
        The plane is the one defined by the "dest" of the first rotation.
        The dest of the second rotation is projected on the plane and then the rotation is computed.
    """

    R1 = create_rotation_matrix(np.array(constraints[0]['from']), constraints[0]['to'])

    y = (R1 @ constraints[1]['from']) / np.linalg.norm(R1 @ constraints[1]['from'])
    projected = project_onto_plane(constraints[1]['to'], constraints[0]['to'])  # TODO CAREFull TO POINT DOWN
    projected = np.array(projected) / np.linalg.norm(projected)

    # Compute angle between projected y axe and actual y axe
    ort = np.cross(y, projected) / np.linalg.norm(np.cross(y, projected))
    sign = round(ort @ constraints[0]['to'])  # this either get rounded to 1 or -1

    rotation_radians = angle_between(y, projected) * sign
    # print(np.degrees(rotation_radians))  # TODO remove debug

    # Rotate mesh
    C = np.array([[0, -constraints[0]['to'][2], constraints[0]['to'][1]],
                  [constraints[0]['to'][2], 0, -constraints[0]['to'][0]],
                  [-constraints[0]['to'][1], constraints[0]['to'][0], 0]])
    R2 = np.eye(3) + C * np.sin(rotation_radians) + C @ C * (1 - np.cos(rotation_radians))

    # viz(points, R1, R2, c1, c2)

    return R2 @ R1


def order_planes(centers, planes):
    normals = planes[..., :3]

    # position: used to determine the position of the planes.
    #   It can be the mean of the ransac points, the normal of the
    #   planes, the closest points on each plane to the origin
    # position = normals * -planes[..., 3:]
    position = centers
    right = np.argmax(position[:, 0], axis=0)
    left = np.argmin(normals @ normals[right])

    # The front plane is the one with the closest to the camera (highest z).
    # Sometimes the right/left-most plane also be the closest to the camera.
    # We don't want that so we removed the one we've already assigned.
    position[[right, left]] = np.array([-1, -1, -1])  # Because 0 can still be greater than every other elements
    front = np.argmax(position[:, 2], axis=0)
    back = np.argmin(normals @ normals[front])

    position[[front, back]] = np.array([-1, -1, -1])
    top = np.argmax(position[:, 1], axis=0)
    bottom = np.argmin(normals @ normals[top])

    return right, left, front, back, top, bottom


def line_plane(line, plane):
    a = np.argmax(plane[:3])
    if plane[a] == 0:  # TODO FIX FIX FIX FIX FIX FIX FIX FIX FIX
        raise ValueError('Division by zero')
    p0 = np.zeros([3])
    p0[a] = -plane[3] / plane[a]

    n0 = plane[:3] / np.linalg.norm(plane[:3])

    l0, l = line

    d = ((p0 - l0) @ n0) / (l @ n0)
    return l0 + l * d


def plane_plane(plane1, plane2):
    a, b = np.argsort(plane1[:3])[:2]
    a = 0
    b = 1
    if plane1[a] == 0:
        raise ValueError('Division by zero')

    l0 = np.zeros([3])

    l0[b] = (plane1[3] * plane2[a] / plane1[a] - plane2[3]) / (plane1[b] * -plane2[a] / plane1[a] + plane2[b])
    l0[a] = (-plane1[b] * l0[b] - plane1[3]) / plane1[a]

    l = np.cross(plane1[:3], plane2[:3])

    return [l0, l]

def project_onto_plane(x, n):
    d = np.dot(x, n) / np.linalg.norm(n)
    n = n / np.linalg.norm(n)
    p = d * n
    return x - p


def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def create_rotation_matrix(a, b):
    """
    Creates the rotation matrix such that b = R @ a
    from https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    Args:
        a: np.array (,3), unit vector already normalized
        b: np.array (,3), unit vector already normalized
    Returns:
        R: np.array(3, 3), transformation matrix
    """
    v = np.cross(a, b)

    c = np.dot(a, b)

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    R = np.eye(3) + vx + np.dot(vx, vx) * (1 / (1 + c))
    return R