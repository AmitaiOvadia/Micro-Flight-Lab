import numpy as np
from sklearn.decomposition import PCA
import visualize
from sklearn.preprocessing import normalize
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


class FlightAnalysis:
    def __init__(self, points_3D_path):
        self.points_3D_path = points_3D_path
        self.points_3D = self.load_points()
        self.num_joints = self.points_3D.shape[1]
        self.num_frames = self.points_3D.shape[0]
        self.num_wings_points = self.num_joints - 2
        self.num_points_per_wing = self.num_wings_points // 2
        self.left_inds = np.arange(0, self.num_points_per_wing)
        self.right_inds = np.arange(self.num_points_per_wing, self.num_wings_points)
        self.wings_pnts_inds = np.array([self.left_inds, self.right_inds])
        self.head_tail_inds = [self.num_wings_points, self.num_wings_points + 1]
        self.planes = self.extract_wings_planes()
        self.head_tail_points = self.points_3D[:, self.head_tail_inds, :]
        self.wings_joints_points = self.points_3D[:, [7, 15], :]
        self.head_tail_vec = self.get_head_tail_vec()
        self.wings_joints_vec = self.get_wings_joints_vec()
        self.yaw_angle, self.pitch_angle, self.roll_angle = self.get_body_angles()
        self.stroke_planes = self.get_stroke_planes()

    def load_points(self):
        return np.load(self.points_3D_path)

    def get_head_tail_vec(self):
        head_tail_vec = self.head_tail_points[:, 1, :] - self.head_tail_points[:, 0, :]
        head_tail_vec = normalize(head_tail_vec, axis=1, norm='l2')
        return head_tail_vec

    def get_wings_joints_vec(self):
        wings_joints_vec = self.wings_joints_points[:, 1] - self.wings_joints_points[:, 0]
        wings_joints_vec = normalize(wings_joints_vec, axis=1, norm='l2')
        return wings_joints_vec

    def get_body_pitch(self):
        pitch = np.rad2deg(np.arcsin(self.head_tail_vec[:, 2]))
        return pitch

    def get_body_yaw(self):
        only_xy = normalize(self.head_tail_vec[:, :-1], axis=1, norm='l2')
        yaw = np.rad2deg(np.arcsin(only_xy[:, 1]))
        return yaw

    def get_body_roll(self):
        roll_angle = np.rad2deg(np.arcsin(self.wings_joints_vec[:, 2]))
        return roll_angle

    def get_body_angles(self):
        yaw_angle = self.get_body_yaw()
        pitch_angle = self.get_body_pitch()
        roll_angle = self.get_body_roll()
        return yaw_angle, pitch_angle, roll_angle

    def get_stroke_planes(self):
        body_y = self.wings_joints_vec
        theta = np.pi / 4
        stroke_normal = self.rodrigues_rot(self.head_tail_vec, body_y, theta)
        body_center = np.mean(self.head_tail_points, axis=1)
        d = - np.sum(np.multiply(stroke_normal, body_center), axis=1)
        stroke_planes = np.column_stack((stroke_normal, d))
        return stroke_planes

    @staticmethod
    def rodrigues_rot(V, K, theta):
        num_frames, ndims = V.shape[0], V.shape[1]
        V_rot = np.zeros_like(V)
        for frame in range(num_frames):
            vi = V[frame, :]
            ki = K[frame, :]
            vi_rot = np.cos(theta) * vi + np.cross(ki, vi) * np.sin(theta) + ki * np.dot(ki, vi) * (1 - np.cos(theta))
            V_rot[frame, :] = vi_rot
        return V_rot

    def extract_wings_planes(self):

        upper_plane_points = [0, 1, 2, 3]
        lower_plane_points = [3, 4, 5, 6]

        all_4_planes = np.zeros((self.num_frames, 4, 4))
        all_2_planes = np.zeros((self.num_frames, 2, 4))
        all_planes_errors = np.zeros((self.num_frames, 6))
        for frame in range(self.num_frames):

            # smaller planes, fit a plane to half the wing
            # left wing
            up_left_pnts = self.points_3D[frame, self.left_inds[upper_plane_points], :]
            plane_P, error = self.fit_plane(up_left_pnts)
            all_4_planes[frame, 0, :] = plane_P
            all_planes_errors[frame, 0] = error

            down_left_pnts = self.points_3D[frame, self.left_inds[lower_plane_points], :]
            plane_P, error = self.fit_plane(down_left_pnts)
            all_4_planes[frame, 1, :] = plane_P
            all_planes_errors[frame, 1] = error

            # right wing
            up_right_pnts = self.points_3D[frame, self.right_inds[upper_plane_points], :]
            plane_P, error = self.fit_plane(up_right_pnts)
            all_4_planes[frame, 2, :] = plane_P
            all_planes_errors[frame, 2] = error

            down_right_pnts = self.points_3D[frame, self.right_inds[lower_plane_points], :]
            plane_P, error = self.fit_plane(down_right_pnts)
            all_4_planes[frame, 3, :] = plane_P
            all_planes_errors[frame, 3] = error

            # fit a plane to the hole wing
            all_left_pnts = self.points_3D[frame, self.left_inds[:-1], :]
            plane_P, error = self.fit_plane(all_left_pnts)
            all_2_planes[frame, 0, :] = plane_P
            all_planes_errors[frame, 4] = error

            all_right_pnts = self.points_3D[frame, self.right_inds[:-1], :]
            plane_P, error = self.fit_plane(all_right_pnts)
            all_2_planes[frame, 1, :] = plane_P
            all_planes_errors[frame, 5] = error

        return all_4_planes, all_2_planes, all_planes_errors

    @staticmethod
    def fit_plane(points):

        # fit a plane
        pca = PCA(n_components=3)
        pca.fit(points)
        normal = pca.components_[-1]  # the z component of the data
        mean = pca.mean_
        d = - normal.T @ mean  # ax + by + cz + d = 0 -> ax + by + cz = -d
        plane_P = np.append(normal, d)

        # find the fit error
        error = np.mean(np.abs(points @ normal + d))
        return plane_P, error


if __name__ == '__main__':
    point_numpy_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D " \
                       r"code\example datasets\points_3D.npy"  # get the first argument
    a = FlightAnalysis(point_numpy_path)
    points_3D = a.points_3D
    # all_4_planes, all_2_planes, all_planes_errors = a.extract_wings_planes()
    strock_planes = a.stroke_planes[:, np.newaxis, :]
    visualize.Visualizer.show_points_and_wing_planes_3D(points_3D, strock_planes)




