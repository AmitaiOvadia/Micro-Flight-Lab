
import json
import h5py
import numpy as np
import os
import scipy.signal
from skimage.morphology import convex_hull_image
from scipy.stats import median_abs_deviation
from traingulate import Triangulate
import visualize
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from validation import Validation
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import disk, erosion, dilation
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import make_smoothing_spline
from scipy.signal import medfilt
from constants import *

WHICH_TO_FLIP = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]).astype(bool)
SIDE_POINTS = [7, 15]
ALPHA = 0.7


class From2Dto3D:
    """
    this is a predictions class that will encapsulate all the data arrays and the functions regarding the 2D and 3D
    predicted points.
    It will hold the data of:
    * 2D raw predictions
    * related fly train_images
    * wings1 train_masks
    * predictions scores for each point in 2D

    It will perform the following functionalities:
    * body segmentation
    * orienting the 2D right and left to 3D world right and left
    * getting the 3D triangulations of each point from all the 6 possible pairs
    * get 'best' 3D point for each 2D point from multiple views synthesis
    * apply smoothing and outlier removal
    """

    def __init__(self, load_from=CONFIG, h5_file_path="", configuration_path=""):
        """

        """
        if load_from == CONFIG:
            with open(configuration_path) as C:
                config = json.load(C)
                self.configuration_path = configuration_path
                self.config = config
                self.points_2D_h5_path = self.config["2D predictions path"]
                self.save_path = self.config["out path"]
                self.num_cams = self.config["number of cameras"]
                self.need_left_right_alignment = bool(self.config["align right left"])
            self.preds_2D = self.load_preds_2D()
            self.cropzone = self.load_cropzone()
            self.box = self.load_box()
            self.conf_preds = self.load_conf_pred()
            self.set_attributes()
            self.triangulate = Triangulate(self.config)
            self.body_masks, self.body_distance_transform = self.set_body_masks()
            if self.need_left_right_alignment:
                self.fix_wings_3D_per_frame()
            self.neto_wings_masks, self.wings_size = self.get_neto_wings_masks()

        elif load_from == H5_FILE:
            self.h5_path = h5_file_path
            self.points_2D_h5_path = self.h5_path
            self.box = self.load_box()
            self.preds_2D = self.load_preds_2D()
            self.cropzone = self.load_cropzone()
            self.conf_preds = self.load_conf_pred()
            self.set_attributes()
            f = h5py.File(self.h5_path, "r")
            self.configuration_path = f.attrs["configuration_path"]
            with open(self.configuration_path) as C:
                self.config = json.load(C)
            self.triangulate = Triangulate(self.config)
            self.num_cams = self.config["number of cameras"]
            self.body_masks = h5py.File(self.h5_path, "r")["/body_masks"][:]
            self.body_distance_transform = h5py.File(self.h5_path, "r")["/body_distance_transform"][:]
            self.neto_wings_masks = h5py.File(self.h5_path, "r")["/neto_wings_masks"][:]
            self.wings_size = h5py.File(self.h5_path, "r")["/wings_size"][:]
            pass



    def set_attributes(self):
        self.image_size = self.box.shape[-2]
        self.num_frames = self.preds_2D.shape[0]
        self.num_joints = self.preds_2D.shape[2]
        self.num_time_channels = self.box.shape[-1] - 2
        self.num_wings_points = self.num_joints - 2
        self.num_points_per_wing = self.num_wings_points // 2
        self.left_inds = np.arange(0, self.num_points_per_wing)
        self.right_inds = np.arange(self.num_points_per_wing, self.num_wings_points)
        self.wings_pnts_inds = np.array([self.left_inds, self.right_inds])
        self.head_tail_inds = [self.num_wings_points, self.num_wings_points + 1]
        self.left_mask_ind = self.box.shape[-1] - 2
        self.right_mask_ind = self.box.shape[-1] - 1

    def save_data_to_h5(self):
        h5_file_name = os.path.join(self.save_path, "preprocessed_2D_to_3D.h5")
        with h5py.File(h5_file_name, "w") as f:
            f.attrs["configuration_path"] = self.configuration_path
            ds_pos = f.create_dataset("positions_pred", data=self.preds_2D, compression="gzip",
                                      compression_opts=1)
            ds_cropzone = f.create_dataset("cropzone", data=self.cropzone, compression="gzip", compression_opts=1)
            ds_box = f.create_dataset("box", data=self.box, compression="gzip", compression_opts=1)
            body_masks = f.create_dataset("body_masks", data=self.body_masks, compression="gzip", compression_opts=1)
            ds_box = f.create_dataset("body_distance_transform", data=self.body_distance_transform, compression="gzip",
                                      compression_opts=1)
            ds_box = f.create_dataset("neto_wings_masks", data=self.neto_wings_masks, compression="gzip",
                                      compression_opts=1)
            ds_box = f.create_dataset("wings_size", data=self.wings_size, compression="gzip",
                                      compression_opts=1)
            ds_box = f.create_dataset("conf_pred", data=self.conf_preds, compression="gzip",
                                      compression_opts=1)
        print(f"saved data to file in path:\n{h5_file_name}")

    def load_data_from_h5(self, h5_path):
        pass

    def get_points_3D(self):
        points_3D = self.choose_best_score_2_cams()
        return points_3D

    def do_smooth_3D_points(self, points_3D):
        return self.smooth_3D_points(points_3D)

    @staticmethod
    def get_validation_score(points_3D):
        return Validation.get_wings_distances_variance(points_3D)[0]

    @staticmethod
    def visualize_3D(points_3D):
        visualize.Visualizer.show_points_in_3D(points_3D)

    def visualize_2D(self, points_2D):
        visualize.Visualizer.show_predictions_all_cams(np.copy(self.box), points_2D)

    def reprojected_2D_points(self, points_3D):
        points_2D_reprojected = self.triangulate.get_reprojections(points_3D, self.cropzone)
        return points_2D_reprojected

    def get_all_3D_pnts_pairs(self, points_2D, cropzone):
        points_3D_all, reprojection_errors, triangulation_errors = \
            self.triangulate.triangulate_2D_to_3D_reprojection_optimization(points_2D, cropzone)
        return points_3D_all, reprojection_errors, triangulation_errors

    def fix_wings_3D_per_frame(self):
        """
        fix the right and left wings1 order in all the cameras according to the 3D ground truth
        one camera (camera 0) is fixed and all cameras are tested (all the options of right-left are considered)

        # step 1: make sure the right-left of chosen camera stays consistent between frames, and flip if needed
        # step 2: find which cameras needed to be flipped to minimize triangulation error, and flip them
        """
        chosen_camera = 0
        cameras_to_check = np.arange(1, 4)
        for frame in range(self.num_frames):
            # step 1
            if frame > 0:
                switch_flag = self.deside_if_switch(chosen_camera, frame)
                if switch_flag:
                    self.flip_camera(chosen_camera, frame)

            # step 2
            cameras_to_flip = self.find_which_cameras_to_flip(cameras_to_check, frame)
            # print(f"frame {frame}, camera to flip {cameras_to_flip}")
            for cam in cameras_to_flip:
                self.flip_camera(cam, frame)

    def get_reprojection_masks(self, points_3D):
        points_2D_reprojected = self.triangulate.get_reprojections(points_3D, self.cropzone)
        reprojected_masks = np.zeros_like(self.neto_wings_masks)
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                for wing in range(2):
                    points_inds = self.wings_pnts_inds[wing, :]
                    mask = np.zeros((self.image_size, self.image_size))
                    wing_pnts = np.round(points_2D_reprojected[frame, cam, points_inds, :]).astype(int)
                    mask[wing_pnts[:, 1], wing_pnts[:, 0]] = 1
                    mask = convex_hull_image(mask)
                    mask = dilation(mask, footprint=np.ones((4, 4)))
                    reprojected_masks[frame, cam, :, :, wing] = mask
        return reprojected_masks

    def set_body_masks(self, opening_rad=6):
        """
        find the fly's body, and the distance transform for later analysis in every camera in 2D using segmentation
        """
        body_masks = np.zeros((self.num_frames, self.num_cams, self.image_size, self.image_size))
        body_distance_transform = np.zeros(body_masks.shape)
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                fly_3_ch = self.box[frame, cam, :, :, :self.num_time_channels]
                fly_3_ch_av = np.sum(fly_3_ch, axis=-1) / self.num_time_channels
                binary_body = fly_3_ch_av >= 0.8
                selem = disk(opening_rad)
                # Perform dilation
                dilated = dilation(binary_body, selem)
                # Perform erosion
                mask = erosion(dilated, selem)
                distance_transform = distance_transform_edt(mask)
                body_masks[frame, cam, ...] = mask
                body_distance_transform[frame, cam, ...] = distance_transform
        return body_masks, body_distance_transform

    def get_neto_wings_masks(self):
        neto_wings = np.zeros((self.num_frames, self.num_cams, self.image_size, self.image_size, 2))
        wings_size = np.zeros((self.num_frames, self.num_cams, 2))
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                body_mask = self.body_masks[frame, cam, ...]
                for wing_num in range(2):
                    other_wing_mask = self.box[frame, cam, :, :, self.num_time_channels + (not wing_num)]
                    wing_mask = self.box[frame, cam, :, :, self.num_time_channels + wing_num]
                    body_and_other_wing_mask = np.bitwise_or(body_mask.astype(bool), other_wing_mask.astype(bool))
                    intersection = np.logical_and(wing_mask, body_and_other_wing_mask)
                    neto_wing = wing_mask - intersection
                    neto_wings[frame, cam, :, :, wing_num] = neto_wing
                    wings_size[frame, cam, wing_num] = np.count_nonzero(neto_wing)
        return neto_wings, wings_size

    def smooth_3D_points(self, points_3D):
        points_3D_smoothed = np.zeros_like(points_3D)
        for pnt in range(self.num_joints):
            for axis in range(3):
                vals = points_3D[:, pnt, axis]
                # set lambda as the regularising parameters: smoothing vs close to data
                lam = 300 if pnt in SIDE_POINTS else None
                A = np.arange(vals.shape[0])
                # weight outliers
                filtered_data = medfilt(vals, kernel_size=3)
                # Compute the absolute difference between the original data and the filtered data
                diff = np.abs(vals - filtered_data)
                # make the diff into weights in [0,1]
                diff = diff / np.max(diff)
                W = 1 - diff
                W[W == 0] = 0.00001
                spline = make_smoothing_spline(A, vals, w=W, lam=lam)
                smoothed = spline(A)
                points_3D_smoothed[:, pnt, axis] = smoothed
        return points_3D_smoothed

    def choose_best_score_2_cams(self, alpha=0.7):
        """
        for each point rank the 4 different cameras by visibility, noise, size of mask, and choose the best 2
        """
        points_3D_all, reprojection_errors, triangulation_errors = self.get_all_3D_pnts_pairs(self.preds_2D, self.cropzone)
        envelope_2D = self.get_derivative_envelope_2D()
        points_3D = np.zeros((self.num_frames, self.num_joints, 3))
        for frame in range(self.num_frames):
            for wing_num in range(2):
                for pnt_num in self.wings_pnts_inds[wing_num, :]:
                    candidates = points_3D_all[frame, pnt_num, :, :]
                    wings_size = self.wings_size[frame, :, wing_num]
                    max_size = np.max(wings_size)
                    masks_sizes_score = wings_size / max_size
                    noise = envelope_2D[frame, :, pnt_num] / np.max(envelope_2D[frame, :, pnt_num])
                    noise_score = 1 - noise

                    # compute the visibility of the point
                    # visibilities = np.zeros(self.num_cams,)
                    # for cam in range(self.num_cams):
                    #     body_dist_trn = self.body_distance_transform[frame, cam, :, :]
                    #     point = self.preds_2D[frame, cam, pnt_num, :]
                    #     px, py = point[0], point[1]
                    #     visibility = body_dist_trn[py, px]
                    #     visibilities[cam] = visibility
                    # visibilities = visibilities / np.max(visibilities)
                    # visibility_score = 1 - visibilities

                    scores = alpha * masks_sizes_score + (1 - alpha) * noise_score
                    # scores = scores * visibility_score
                    cameras_ind = np.sort(np.argpartition(scores, -2)[-2:])
                    best_pair_ind = self.triangulate.all_subs.index(tuple(cameras_ind))
                    best_3D_point = candidates[best_pair_ind]
                    points_3D[frame, pnt_num, :] = best_3D_point
        points_3D[:, self.head_tail_inds, :] = self.choose_average_points()[:, self.head_tail_inds, :]
        return points_3D

    def get_derivative_envelope_2D(self):
        derivative_2D = self.get_2D_derivatives()
        envelope_2D = np.zeros(shape=self.preds_2D.shape[:-1])
        for cam in range(self.num_cams):
            for joint in range(self.num_joints):
                signal = derivative_2D[:, cam, joint]
                # Define the cutoff frequency for the low-pass filter (between 0 and 1)
                # Calculate the analytic signal, from which the envelope is the magnitude
                analytic_signal = hilbert(signal)
                envelope = np.abs(analytic_signal)
                smooth_envelope = gaussian_filter1d(envelope, 0.7)
                envelope_2D[:, cam, joint] = smooth_envelope
        return envelope_2D

    def get_2D_derivatives(self):
        derivative_2D = np.zeros(shape=self.preds_2D.shape[:-1])
        derivative_2D[1:, ...] = np.linalg.norm(self.preds_2D[1:, ...] - self.preds_2D[:-1, ...], axis=-1)
        return derivative_2D

    def choose_best_reprojection_error_points(self):
        points_3D_all, reprojection_errors, triangulation_errors = self.get_all_3D_pnts_pairs(self.preds_2D,
                                                                                               self.cropzone)
        points_3D = np.zeros((self.num_frames, self.num_joints, 3))
        for frame in range(self.num_frames):
            for joint in range(self.num_joints):
                candidates = points_3D_all[frame, joint, ...]
                best_candidate_ind = np.argmin(reprojection_errors[frame, joint, ...])
                point_3d = candidates[best_candidate_ind]
                points_3D[frame, joint, :] = point_3d
        return points_3D

    def choose_average_points(self):
        points_3D_all, reprojection_errors, triangulation_errors = self.get_all_3D_pnts_pairs(self.preds_2D, self.cropzone)
        points_3D = np.zeros(shape=(self.num_frames, self.num_joints, 3))
        for frame in range(self.num_frames):
            for joint in range(self.num_joints):
                candidates = points_3D_all[frame, joint, ...]
                candidates_inliers = self.find_outliers_MAD(candidates, 3)
                point_3d = candidates_inliers.mean(axis=0)
                points_3D[frame, joint, :] = point_3d
        return points_3D

    def choose_best_conf_points(self):
        points_3D_all, reprojection_errors, triangulation_errors = self.get_all_3D_pnts_pairs(self.preds_2D,
                                                                                              self.cropzone)
        points_3D = np.zeros(shape=(self.num_frames, self.num_joints, 3))
        for frame in range(self.num_frames):
            for joint in range(self.num_joints):
                candidates = points_3D_all[frame, joint, ...]
                confidence_scores = self.conf_preds[frame, :, joint]
                indices = np.argpartition(confidence_scores, -2)[-2:]
                indices = tuple(np.sort(indices))
                best_pair_ind = self.triangulate.all_subs.index(indices)
                best_conf_3D_point = candidates[best_pair_ind]
                points_3D[frame, joint, :] = best_conf_3D_point
        return points_3D

    @staticmethod
    def find_outliers_MAD(candidates, threshold):
        median = np.median(candidates, axis=0)
        MAD = median_abs_deviation(candidates)
        inliers = np.linalg.norm((candidates - median) / MAD, axis=-1) < threshold
        candidates_inliers = candidates[inliers]
        return candidates_inliers

    def find_which_cameras_to_flip(self, cameras_to_check, frame):
        num_of_options = len(WHICH_TO_FLIP)
        switch_scores = np.zeros(num_of_options, )
        for i, option in enumerate(WHICH_TO_FLIP):
            points_2D, cropzone = self.get_orig_2d_points_and_cropzone(frame)
            cameras_to_flip = cameras_to_check[option]
            for cam in cameras_to_flip:
                left_points = points_2D[0, cam, self.left_inds, :]
                right_points = points_2D[0, cam, self.right_inds, :]
                points_2D[0, cam, self.left_inds, :] = right_points
                points_2D[0, cam, self.right_inds, :] = left_points
            _, reprojection_errors, _ = self.get_all_3D_pnts_pairs(points_2D, cropzone)
            score = np.mean(reprojection_errors)
            switch_scores[i] = score
        cameras_to_flip = cameras_to_check[WHICH_TO_FLIP[np.argmin(switch_scores)]]
        return cameras_to_flip

    def deside_if_switch(self, chosen_camera, frame):
        cur_left_points = self.preds_2D[frame, chosen_camera, self.left_inds, :]
        cur_right_points = self.preds_2D[frame, chosen_camera, self.right_inds, :]
        prev_left_points = self.preds_2D[frame - 1, chosen_camera, self.left_inds, :]
        prev_right_points = self.preds_2D[frame - 1, chosen_camera, self.right_inds, :]
        l2l_dist = np.linalg.norm(cur_left_points - prev_left_points)
        r2r_dist = np.linalg.norm(cur_right_points - prev_right_points)
        r2l_dist = np.linalg.norm(cur_right_points - prev_left_points)
        l2r_dist = np.linalg.norm(cur_left_points - prev_right_points)
        do_switch = l2l_dist + r2r_dist > r2l_dist + l2r_dist
        return do_switch

    def flip_camera(self, camera_to_flip, frame):
        left_points = self.preds_2D[frame, camera_to_flip, self.left_inds, :]
        right_points = self.preds_2D[frame, camera_to_flip, self.right_inds, :]
        self.preds_2D[frame, camera_to_flip, self.left_inds, :] = right_points
        self.preds_2D[frame, camera_to_flip, self.right_inds, :] = left_points
        # switch train_masks in box
        self.box[frame, camera_to_flip, :, :, [self.left_mask_ind, self.right_mask_ind]] = \
            self.box[frame, camera_to_flip, :, :, [self.right_mask_ind, self.left_mask_ind]]
        # switch confidence scores
        left_conf_scores = self.conf_preds[frame, camera_to_flip, self.left_inds]
        right_conf_scores = self.conf_preds[frame, camera_to_flip, self.right_inds]
        self.conf_preds[frame, camera_to_flip, self.left_inds] = left_conf_scores
        self.conf_preds[frame, camera_to_flip, self.right_inds] = right_conf_scores

    def get_orig_2d_points_and_cropzone(self, frame):
        orig_2d_points = self.preds_2D[frame, :, np.concatenate((self.left_inds, self.right_inds)), :]
        orig_2d_points = orig_2d_points[np.newaxis, ...]
        orig_2d_points = np.transpose(orig_2d_points, [0, 2, 1, 3])

        cropzone = self.cropzone[frame]
        cropzone = cropzone[np.newaxis, ...]
        return orig_2d_points, cropzone

    def load_preds_2D(self):
        return h5py.File(self.points_2D_h5_path, "r")["/positions_pred"][:]

    def load_cropzone(self):
        return h5py.File(self.points_2D_h5_path, "r")["/cropzone"][:]

    def load_conf_pred(self):
        return h5py.File(self.points_2D_h5_path, "r")["/conf_pred"][:]

    def load_box(self):
        return h5py.File(self.points_2D_h5_path, "r")["/box"][:]

    @staticmethod
    def save_points_3D(save_dir, points_to_save):
        save_path = os.path.join(save_dir, "points_3D.npy")
        np.save(save_path, points_to_save)


if __name__ == '__main__':
    config_path = r"2D_to_3D_config.json"  # get the first argument
    # predictor = From2Dto3D(configuration_path=config_path, load_from=CONFIG)
    # predictor.save_data_to_h5()

    h5_file = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\example " \
              r"datasets\movie_14_800_1799_ds_3tc_7tj_WINGS_AND_BODY_SAME_MODEL_Jan 18_06\preprocessed_2D_to_3D.h5 "
    save_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\example " \
              r"datasets\movie_14_800_1799_ds_3tc_7tj_WINGS_AND_BODY_SAME_MODEL_Jan 18_06"
    predictor_3D = From2Dto3D(load_from=H5_FILE, h5_file_path=h5_file)
    points_3D = predictor_3D.get_points_3D()
    smoothed_points_3D = predictor_3D.smooth_3D_points(points_3D)
    predictor_3D.save_points_3D(save_path, smoothed_points_3D)
    predictor_3D.visualize_3D(smoothed_points_3D)
    pass
    # predictor.get_all_rays_intersections_3D()
