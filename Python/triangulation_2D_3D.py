import numpy as np
import scipy.io
import itertools
from scipy.io import savemat
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.ndimage
from PIL import Image
import cv2
IMAGE_HEIGHT = 800
IMAGE_WIDTH = 1280


def get_all_couples():
    s = {0, 1, 2, 3}
    all_subs = []
    for i in range(2, 3):
        subs = Triangulate.findsubsets(s, i)
        all_subs += subs
    return all_subs


class Triangulate:
    def __init__(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            self.rotation_matrix = f["rotation_matrix"][:]
            self.camera_matrices = f["cameras_dlt_array"][:].T
            self.inv_camera_matrices = f["cameras_inv_dlt_array"][:].T
            self.camera_centers = f["camera_centers"][:].T
            self.box = f["box"][:].T[:,:,:,:,:3]
            self.points_2D = f["joints"][:].T
            self.cropzone = f["cropZone"][:]
        self.num_frames = self.points_2D.shape[0]
        self.num_cams = self.camera_matrices.shape[0]
        self.num_points = self.points_2D.shape[2]
        self.points_2D_uncropped = None
        self.all_subs = get_all_couples()
        self.all_points_3D = None

    def triangulate_2D_to_3D_points(self):
        new_shape = list(self.points_2D.shape)
        new_shape[-1] += 1
        self.points_2D_uncropped = self.get_uncropped_xy1(new_shape)
        self.all_points_3D = np.zeros((self.num_points, self.num_frames, 6, 3))
        for frame in range(self.num_frames):
            for point in range(self.num_points):
                for i, couple in enumerate(self.all_subs):
                    cam_a, cam_b = couple

                    inv_cam_mat_a = self.inv_camera_matrices[cam_a]  # get the inverse camera matrix
                    center_a = self.camera_centers[cam_a]  # camera center
                    pa = self.points_2D_uncropped[frame, cam_a, point, :]  # get point
                    PB_a = inv_cam_mat_a @ pa  # get projection
                    PB_a_n = PB_a[:-1] / PB_a[-1]  # normalize projection

                    inv_cam_mat_b = self.inv_camera_matrices[cam_b]
                    pb = self.points_2D_uncropped[frame, cam_b, point, :]
                    center_b = self.camera_centers[cam_b]
                    PB_b = inv_cam_mat_b @ pb
                    PB_b_n = PB_b[:-1] / PB_b[-1]

                    PA = np.vstack((center_a, center_b))
                    PB = np.vstack((PB_a_n, PB_b_n))

                    p = np.squeeze(Triangulate.lineIntersect3D(PA, PB))
                    self.all_points_3D[point, frame, i, :] = self.rotation_matrix @ p
        return self.all_points_3D

    def get_uncropped_xy1(self, new_shape):
        points_2D_uncropped = np.zeros(new_shape)
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                for pnt in range(self.num_points):
                    x = self.cropzone[frame, cam, 1] + self.points_2D[frame, cam, pnt, 0]
                    y = self.cropzone[frame, cam, 0] + self.points_2D[frame, cam, pnt, 1]
                    y = IMAGE_HEIGHT + 1 - y
                    point = [x, y, 1]
                    points_2D_uncropped[frame, cam, pnt, :] = point
        return points_2D_uncropped

    # def test_augmentations(self):
    #     image = self.box[0, 0, :, :, :]
    #     plt.scatter(self.points_2D[0, 0, :, 0], self.points_2D[0, 0, :, 1], color='red')
    #     plt.imshow(image)
    #     plt.show()
    #
    #
    #     x1 = self.cropzone[0, 0, 1]
    #     y1 = self.cropzone[0, 0, 0]
    #     x2 = x1 + 192
    #     y2 = y1 + 192
    #     orig_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    #     orig_image[y1:y2, x1:x2, :] = image
    #     orig_image_flipped = np.flipud(orig_image)
    #
    #     x = self.cropzone[0, 0, 1] + self.points_2D[0, 0, :, 0]
    #     y = self.cropzone[0, 0, 0] + self.points_2D[0, 0, :, 1]
    #     y = IMAGE_HEIGHT + 1 - y
    #
    #     plt.scatter(x, y, color='red')
    #     plt.imshow(orig_image_flipped)
    #     plt.show()
    #
    #     theta = 30
    #     # Calculate the center of the bounding box
    #     center = ((x1 + x2) / 2, (y1 + y2) / 2)
    #
    #     # Get the rotation matrix
    #     rotation_matrix = cv2.getRotationMatrix2D(center, theta, 1)
    #
    #     # Perform the rotation
    #     rotated_image = cv2.warpAffine(orig_image, rotation_matrix, (IMAGE_WIDTH, IMAGE_HEIGHT), flags=cv2.INTER_CUBIC)
    #     plt.imshow(rotated_image)
    #     plt.show()
    #     pass



    @staticmethod
    def decompose_camera_matrix(P):
        # Compute the camera center
        C = -np.linalg.inv(P[:, :3]) @ P[:, 3]
        # Compute the rotation matrix
        M = P[:, :3]
        Q, R = np.linalg.qr(np.linalg.inv(M))
        K = np.linalg.inv(R)
        R = Q.T
        # Ensure that the diagonal elements of K are positive
        if np.linalg.det(R) < 0:
                R = -R
                K = -K
        # Compute the translation vector
        t = -R @ C

        return K, R, t, C



    @staticmethod
    def lineIntersect3D(PA, PB):
        # Ensure all inputs are tensors
        PA = tf.convert_to_tensor(PA, dtype=tf.float32)
        PB = tf.convert_to_tensor(PB, dtype=tf.float32)

        # N lines described as vectors
        Si = PB - PA

        # Normalize vectors
        ni = tf.linalg.l2_normalize(Si, axis=-1)
        nx, ny, nz = tf.unstack(ni, axis=-1)

        # Calculate S matrix
        SXX = tf.reduce_sum(nx ** 2 - 1)
        SYY = tf.reduce_sum(ny ** 2 - 1)
        SZZ = tf.reduce_sum(nz ** 2 - 1)
        SXY = tf.reduce_sum(nx * ny)
        SXZ = tf.reduce_sum(nx * nz)
        SYZ = tf.reduce_sum(ny * nz)
        S = tf.stack([[SXX, SXY, SXZ], [SXY, SYY, SYZ], [SXZ, SYZ, SZZ]])

        # Calculate C vector
        CX = tf.reduce_sum(PA[:, 0] * (nx ** 2 - 1) + PA[:, 1] * (nx * ny) + PA[:, 2] * (nx * nz))
        CY = tf.reduce_sum(PA[:, 0] * (nx * ny) + PA[:, 1] * (ny ** 2 - 1) + PA[:, 2] * (ny * nz))
        CZ = tf.reduce_sum(PA[:, 0] * (nx * nz) + PA[:, 1] * (ny * nz) + PA[:, 2] * (nz ** 2 - 1))
        C = tf.stack([CX, CY, CZ])

        # Solve for intersection point
        P_intersect = tf.linalg.solve(S, tf.expand_dims(C, axis=1))
        return P_intersect.numpy()

    @staticmethod
    def findsubsets(s, n):
        return list(itertools.combinations(s, n))


if __name__ == '__main__':

    h5path = r"C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\main datasets\head_tail_dataset\training\pre_train_1000_frames_5_channels_ds_3tc_7tj.h5"
    tr = Triangulate(h5path)
    # tr.test_augmentations()
    all_points_3D = tr.triangulate_2D_to_3D_points()
    xy = all_points_3D[0,0,:,:]
    mean = np.mean(all_points_3D, axis=2)[0]

    print("finished")
