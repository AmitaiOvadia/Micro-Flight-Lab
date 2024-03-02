import numpy as np
from sklearn.decomposition import PCA
import visualize
from sklearn.preprocessing import normalize
import matplotlib
import matplotlib.pyplot as plt
import h5py
import plotly.graph_objects as go
import plotly.io as pio
import scipy
import plotly
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from scipy.interpolate import make_smoothing_spline
matplotlib.use('TkAgg')


dt = 1/16000
LEFT = 0
RIGHT = 1
NUM_TIPS_FOR_PLANE = 10
WINGS_JOINTS_INDS = [7, 15]


class FlightAnalysis:
    def __init__(self, points_3D_path):

        # set attributes
        self.points_3D_path = points_3D_path
        self.points_3D = self.load_points()
        self.num_joints = self.points_3D.shape[1]
        self.num_frames = self.points_3D.shape[0]
        self.num_wings_points = self.num_joints - 2
        self.num_points_per_wing = self.num_wings_points // 2
        self.left_inds = np.arange(0, self.num_points_per_wing)
        self.right_inds = np.arange(self.num_points_per_wing, self.num_wings_points)
        self.left_wing_inds = self.left_inds[:-1]
        self.right_wing_inds = self.right_inds[:-1]
        self.all_wing_inds = np.array([self.left_wing_inds, self.right_wing_inds])
        self.all_non_body_points = np.array([self.left_inds, self.right_inds])
        self.head_tail_inds = [self.num_wings_points, self.num_wings_points + 1]
        self.wings_tips_inds = [3, 3 + len(self.right_inds)]

        # calculate things
        self.head_tail_points = self.get_head_tail_points(smooth=True)
        self.x_body = self.get_head_tail_vec()
        self.set_right_left()
        self.wings_tips = self.get_wing_tips()
        self.wings_CM = self.get_wing_CM()
        self.wings_joints_points = self.get_wings_joints()
        self.planes = self.extract_wings_planes()
        self.wings_span_vecs = self.get_wings_spans()
        self.y_body = self.get_roni_y_body()
        self.z_body = self.get_z_body()
        self.wings_joints_vec = self.get_wings_joints_vec(smooth=True)
        self.yaw_angle, self.pitch_angle, self.roll_angle = self.get_body_angles()
        self.stroke_planes = self.get_stroke_planes()
        self.center_of_mass = self.get_center_of_mass()
        self.body_speed = self.get_body_speed()
        self.get_wing_tips_speed = self.get_wing_tips_speed()
        self.auto_correlation_axis_angle = self.get_auto_correlation_axis_angle()
        self.auto_correlation_x_body = self.get_auto_correlation_x_body()

    def get_head_tail_points(self, smooth=True):
        head_tail_points = self.points_3D[:, self.head_tail_inds, :]
        if smooth:
            head_tail_smoothed = np.zeros_like(head_tail_points)
            for pnt in range(head_tail_points.shape[1]):
                points_orig = head_tail_points[:, pnt, :]
                points = np.apply_along_axis(medfilt, axis=0, arr=points_orig, kernel_size=21)
                A = np.arange(len(points))
                for axis in range(3):
                    spline = make_smoothing_spline(y=points[:, axis], x=A, lam=1000000)
                    points[:, axis] = spline(A)
                head_tail_smoothed[:, pnt, :] = points
            head_tail_points = head_tail_smoothed
        return head_tail_points

    def get_wings_joints(self):
        return self.points_3D[:, WINGS_JOINTS_INDS, :]

    def get_wing_CM(self):
        CM_left = np.mean(self.points_3D[:, self.left_wing_inds, :], axis=1)[:, np.newaxis, :]
        CM_right = np.mean(self.points_3D[:, self.right_wing_inds, :], axis=1)[:, np.newaxis, :]
        wings_CMs = np.concatenate((CM_left, CM_right), axis=1)
        return wings_CMs

    def get_wing_tips(self):
        return self.points_3D[:, self.wings_tips_inds, :]

    def get_center_of_mass(self):
        CM = np.mean(self.head_tail_points, axis=1)
        return CM

    def get_body_speed(self):
        return self.get_speed(self.center_of_mass)

    def get_wing_tips_speed(self):
        left_tip_speed = self.get_speed(self.wings_tips[:, LEFT, :])
        right_tip_speed = self.get_speed(self.wings_tips[:, RIGHT, :])
        wing_tips_speed = np.concatenate((right_tip_speed[:, np.newaxis], left_tip_speed[:, np.newaxis]), axis=1)
        return wing_tips_speed


    @staticmethod
    def get_speed(points_3d):
        T = np.arange(len(points_3d))
        derivative_3D = np.zeros((len(points_3d), 3))
        for axis in range(3):
            spline = make_smoothing_spline(y=points_3d[:, axis], x=T)
            derivative = spline.derivative()(T)
            derivative_3D[:, axis] = derivative
        derivative_3D = derivative_3D / dt  # find the real speed
        body_speed = np.linalg.norm(derivative_3D, axis=1)
        return body_speed

    def get_wings_spans(self):
        """
        calculates the wing spans as the normalized vector from the wing center of mass to the wing tip
        Returns: an array of size (num_frames, 2, 3), axis 1 is left and right
        """
        wing_spans = np.zeros((self.num_frames, 2, 3))
        for wing in range(2):
            CMs = np.mean(self.points_3D[:, self.all_wing_inds[wing, :], :], axis=1)
            tip = self.points_3D[:, self.all_wing_inds[wing, 3], :]
            wing_span = normalize(tip - CMs, axis=1)
            wing_spans[:, wing, :] = wing_span
        return wing_spans

    def get_z_body(self):
        z_body = np.cross(self.x_body, self.y_body)
        z_body = normalize(z_body, 'l2')
        return z_body

    def set_right_left(self):
        wing_CM_left = np.mean(self.points_3D[0, self.left_inds[:-1], :], axis=0)
        wing_CM_right = np.mean(self.points_3D[0, self.right_inds[:-1], :], axis=0)
        wings_vec = wing_CM_right - wing_CM_left
        wings_vec = wings_vec / np.linalg.norm(wings_vec)
        cross = np.cross(wings_vec, self.x_body[0])
        z = 2
        need_flip = False
        if cross[z] < 0:
            need_flip = True
        if need_flip:
            left = self.points_3D[:, self.left_inds, :]
            right = self.points_3D[:, self.right_inds, :]
            self.points_3D[:, self.left_inds, :] = right
            self.points_3D[:, self.right_inds, :] = left

    def load_points(self):
        return np.load(self.points_3D_path)

    def get_head_tail_vec(self, smooth=False):
        head_tail_vec = self.head_tail_points[:, 1, :] - self.head_tail_points[:, 0, :]
        head_tail_vec = normalize(head_tail_vec, axis=1, norm='l2')
        return head_tail_vec

    def get_wings_joints_vec(self, smooth=False):
        wings_joints_vec = self.wings_joints_points[:, LEFT] - self.wings_joints_points[:, RIGHT]
        if smooth:
            wings_joints_vec_smoothed = np.zeros_like(self.wings_joints_points)
            for pnt in range(self.wings_joints_points.shape[1]):
                points_orig = self.wings_joints_points[:, pnt, :]
                points = np.apply_along_axis(medfilt, axis=0, arr=points_orig, kernel_size=21)
                A = np.arange(len(points))
                for axis in range(3):
                    spline = make_smoothing_spline(y=points[:, axis], x=A, lam=100000)
                    points[:, axis] = spline(A)

                wings_joints_vec_smoothed[:, pnt, :] = points
            wings_joints_vec = wings_joints_vec_smoothed[:, LEFT, :] - wings_joints_vec_smoothed[:, RIGHT, :]

        wings_joints_vec = normalize(wings_joints_vec, axis=1, norm='l2')
        return wings_joints_vec

    def get_body_pitch(self):
        pitch = np.rad2deg(np.arcsin(self.x_body[:, 2]))
        return pitch

    def get_body_yaw(self):
        only_xy = normalize(self.x_body[:, :-1], axis=1, norm='l2')
        yaw = np.rad2deg(np.arctan2(only_xy[:, 1], only_xy[:, 0]))
        yaw = np.unwrap(yaw + 180, period=360) - 180
        return yaw

    def get_body_roll(self):
        roll_angle = np.rad2deg(np.arcsin(self.y_body[:, 2]))
        return roll_angle

    def get_body_angles(self):
        yaw_angle = self.get_body_yaw()
        pitch_angle = self.get_body_pitch()
        roll_angle = self.get_body_roll()
        return yaw_angle, pitch_angle, roll_angle

    def get_stroke_planes(self):
        theta = np.pi / 4
        stroke_normal = self.rodrigues_rot(self.x_body, self.y_body, theta)
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

    def get_auto_correlation_axis_angle(self):
        first_nonzero_index = np.argmax((self.y_body != 0).any(axis=1))
        reversed_mat = np.flip(self.y_body, axis=0)
        last_index_reversed = np.argmax((reversed_mat != 0).any(axis=1))
        last_nonzero_index = self.y_body.shape[0] - 1 - last_index_reversed

        T = (last_nonzero_index - first_nonzero_index) // 2
        x_body, y_body, z_body = (self.x_body[first_nonzero_index:last_nonzero_index],
                                  self.y_body[first_nonzero_index:last_nonzero_index],
                                  self.z_body[first_nonzero_index:last_nonzero_index])
        AC = np.zeros(T)
        AC[0] = 1
        for df in range(1, T):
            xb, yb, zb = x_body[:-df], y_body[:-df], z_body[:-df]
            Rs = np.stack([xb, yb, zb], axis=-1)

            xb_pair, yb_pair, zb_pair = x_body[df:], y_body[df:], z_body[df:]
            Rs_pair = np.stack([xb_pair, yb_pair, zb_pair], axis=-1)

            angels_radiance = np.array([self.get_rotation_axis_angle(Rs[i], Rs_pair[i]) for i in range(Rs.shape[0])])
            cosines = np.cos(angels_radiance)
            AC[df] = np.mean(cosines)
        return AC

    def get_auto_correlation_x_body(self):
        # Define M
        T = len(self.x_body) // 2
        AC = np.zeros(T)
        AC[0] = 1
        for df in range(1, T):
            x_bodies = self.x_body[:-df]
            x_bodies_pair = self.x_body[df:]
            cosines = self.dot(x_bodies, x_bodies_pair)
            AC[df] = np.mean(cosines)
        return

    @staticmethod
    def get_rotation_axis_angle(vectors1, vectors2):
        # vectors1, vectors2 (3, 3) matrices of perpendicular unit vectors (each row is a vector)
        # Create rotation matrix
        R = np.dot(vectors2, vectors1.T)

        # Calculate rotation axis
        # _, v = np.linalg.eig(R)
        # axis = np.real(v[:, np.isclose(np.linalg.eigvals(R), 1)])

        # Calculate rotation angle
        angle_rad = np.arccos((np.trace(R) - 1) / 2)
        return angle_rad

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

    @staticmethod
    def auto_correlation_1D(angles, is_radians=False):
        """
        computes the auto-correlation between angles:
        C(dt) =  <cos[Theta(t) - theta(t + dt)]>
        """
        T = len(angles) // 2
        AC = np.zeros(T)
        AC[0] = 1
        if not is_radians:
            angles = np.deg2rad(angles)
        for df in range(1, T):
            d_theta = angles[df:] - angles[:-df]
            cosines = np.cos(d_theta)
            AC[df] = np.mean(cosines)
        return AC

    @staticmethod
    def auto_correlation_3D(unit_vectors):
        T = len(unit_vectors) // 2
        AC = np.zeros(T)
        AC[0] = 1
        for df in range(1, T):
            first_vecs = unit_vectors[df:]
            second_vecs = unit_vectors[:-df]
            cosines = FlightAnalysis.dot(first_vecs, second_vecs)
            AC[df] = np.mean(cosines)
        return AC

    @staticmethod
    def dot(arr1, arr2):
        """
        Args:
            arr1: size (N, d)
            arr2: size (N, d)
        Returns:
            row-wize dot product
        """
        return np.sum(arr1 * arr2, axis=1)

    @staticmethod
    def find_zero_crossings_up(data):
        # data is 1D
        # find when sign is changing from negative to positive
        return np.where(np.diff(np.sign(data)) > 0)[0]

    @staticmethod
    def remove_close_elements(data, threshold=60):
        diffs = np.abs(np.diff(data))
        indices = np.insert(diffs >= threshold, 0, True)
        return data[indices]

    def choose_span(self):
        dotspanAx_wing1 = self.dot(self.wings_span_vecs[:, LEFT, :], self.x_body)
        dotspanAx_wing2 = self.dot(self.wings_span_vecs[:, RIGHT, :], self.x_body)

        distSpans = np.arccos(self.dot(self.wings_span_vecs[:, LEFT, :], self.wings_span_vecs[:, RIGHT, :]))
        angBodSp = np.rad2deg(np.arccos(self.dot(self.wings_span_vecs[:, RIGHT, :], self.x_body)))
        mean_strks = np.mean(np.array([dotspanAx_wing1, dotspanAx_wing2]), axis=0)
        idx4StrkPln = self.find_zero_crossings_up(mean_strks)

        idx4StrkPln = self.remove_close_elements(idx4StrkPln)
        idx4StrkPln = np.array([i for i in idx4StrkPln if abs(angBodSp[i] - 90) <= 20])
        return idx4StrkPln

    def get_roni_y_body(self):
        # should be used after head_tail_vec is obtained
        idx4StrkPln = self.choose_span()
        idx4StrkPln = idx4StrkPln[(NUM_TIPS_FOR_PLANE <= idx4StrkPln) & (idx4StrkPln <= self.num_frames - NUM_TIPS_FOR_PLANE)]
        y_bodies = []
        for i, ind in enumerate(idx4StrkPln):
            left = self.wings_tips[ind - NUM_TIPS_FOR_PLANE:ind + NUM_TIPS_FOR_PLANE, LEFT, :]
            right = self.wings_tips[ind - NUM_TIPS_FOR_PLANE:ind + NUM_TIPS_FOR_PLANE, RIGHT, :]
            points = np.concatenate((left, right), axis=0)
            wing_tips_plane = self.fit_plane(points)[0]
            plane_normal = wing_tips_plane[:-1]
            y_body = np.cross(plane_normal, self.x_body[ind])
            y_body = y_body / np.linalg.norm(y_body)
            if (i > 0) and (np.dot(y_body, y_bodies[i - 1]) < 0):
                y_body = - y_body
            y_bodies.append(y_body)
            # self.plot_plane_and_points(ind, wing_tips_plane, points, y_body)
            pass
        y_bodies = np.array(y_bodies)
        all_y_bodies = np.zeros_like(self.x_body)
        start = np.min(idx4StrkPln)
        end = np.max(idx4StrkPln)
        x = np.arange(start, end)
        f1 = interp1d(idx4StrkPln, y_bodies[:, 0], kind='cubic')
        f2 = interp1d(idx4StrkPln, y_bodies[:, 1], kind='cubic')
        f3 = interp1d(idx4StrkPln, y_bodies[:, 2], kind='cubic')
        Ybody_inter = np.vstack((f1(x), f2(x), f3(x))).T
        Ybody_inter = normalize(Ybody_inter, axis=1, norm='l2')
        all_y_bodies[start:end, :] = Ybody_inter
        # make sure that the all_y_bodies are (1) unit vectors and (2) perpendicular to x_body
        y_bodies_corrected = all_y_bodies - self.x_body * self.dot(self.x_body, all_y_bodies).reshape(-1, 1)
        y_bodies_corrected = normalize(y_bodies_corrected, 'l2')

        # make sure that the all_y_bodies are (1) unit vectors and (2) perpendicular to x_body
        # y_bodies_corrected_1 = np.zeros_like(y_bodies_corrected)
        # all_angles_errors = np.zeros((self.num_frames,))
        # for i in range(start, end):
        #     # Find the rotation axis, which is the cross product of A[i] and B[i]
        #     yb_i = all_y_bodies[i]
        #     yb_i = yb_i / np.linalg.norm(yb_i)
        #     xb_i = self.x_body[i]
        #     axis = np.cross(yb_i, xb_i)
        #     axis = axis / np.linalg.norm(axis)
        #     # Find the angle between A[i] and B[i]
        #     angle = np.arccos(np.dot(yb_i, xb_i)) - np.pi / 2
        #     all_angles_errors[i] = np.rad2deg(angle)
        #     yb_i_corrected = self.rodrigues_rot(yb_i[np.newaxis, :], axis[np.newaxis, :], angle)
        #     y_bodies_corrected_1[i] = yb_i_corrected
        return y_bodies_corrected

    def plot_plane_and_points(self, ind, plane, points, y_body):
        size = 0.005
        cm = np.mean(self.head_tail_points[ind], axis=0)
        wing_points = points
        center = np.mean(points, axis=0)
        a, b, c, d = plane
        x = np.linspace(-size / 2, size / 2, 10) + center[0]
        y = np.linspace(-size / 2, size / 2, 10) + center[1]
        xx, yy = np.meshgrid(x, y)
        zz = (-a * xx - b * yy - d) / c
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # plot surface
        ax.plot_surface(xx, yy, zz, color='green', alpha=0.5, label='Plane')
        # plot points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        # plot center of mass
        ax.scatter(*cm, color="r", s=100)
        # plot head and tail
        ax.scatter(self.head_tail_points[ind, :, 0], self.head_tail_points[ind, :, 1], self.head_tail_points[ind, :, 2])
        # Plot the vectors
        ax.quiver(*cm, *(y_body * 0.001), color="b", alpha=.8)
        # ax.quiver(*cm, *(np.array([a, b, c]) * 0.001), color="g", alpha=.8)
        ax.set_box_aspect([1, 1, 1])
        plt.show()


def plot_movie_html(movie_num):
    if movie_num == 1:
        title = "mov1 dark disturbance smoothed.html"
        point_numpy_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov61_d\points_3D_smoothed_ensemble.npy"
        start_frame = 10
    elif movie_num == 2:
        title = "mov2 dark disturbance smoothed.html"
        point_numpy_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov62_d\points_3D_smoothed_ensemble.npy"
        start_frame = 160
    elif movie_num == 3:
        title = "mov1 free flight smoothed.html"
        point_numpy_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov10_u\points_3D_smoothed_ensemble.npy"
        start_frame = 130
    else:
        title = "mov2 free flight smoothed.html"
        point_numpy_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov11_u\points_3D_smoothed_ensemble.npy"
        start_frame = 10

    FA = FlightAnalysis(point_numpy_path)

    # plt.plot(FA.wings_tips[:, 0, :])
    # plt.show()

    # ybody1 = FA.y_body
    # plt.plot(ybody1)
    # plt.show()
    # ybody2 = FA.wings_joints_vec
    # plt.figure()

    com = FA.center_of_mass
    x_body = FA.x_body
    y_body = FA.y_body
    points_3D = FA.points_3D

    plt.plot(FA.head_tail_points[:, 1, :])
    plt.show()

    visualize.Visualizer.create_movie_plot(com, x_body, y_body, points_3D, start_frame, title)


if __name__ == '__main__':
    # plot_movie_html(1)
    # plot_movie_html(2)
    # plot_movie_html(3)
    # plot_movie_html(4)

    # mov10
    # path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov10_u\points_3D_ensemble.npy"
    # h5 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov10_u\body_segmentations.h5"

    # mov11
    # path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov11_u\points_3D_ensemble.npy"
    # h5 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov11_u\body_segmentations.h5"

    # mov62
    path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov62_d\points_3D_ensemble.npy"
    h5 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov62_d\body_segmentations.h5"

    FA = FlightAnalysis(path)
    x_body1 = h5py.File(h5, "r")["/x_body"][:]
    x_body2 = FA.x_body

    x_body3 = FA.points_3D[:, -1, :] - FA.head_tail_points[:, -2, :]
    x_body3 = normalize(x_body3, axis=1, norm='l2')

    plt.plot(x_body1[72:-100])
    # plt.plot(x_body2[72:-100])
    # plt.plot(x_body3[72:-72])
    plt.show()
    pass

