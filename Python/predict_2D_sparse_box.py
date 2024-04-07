import matplotlib.pyplot as plt
import scipy.ndimage
import scipy
import yaml
import h5py
from skimage.morphology import disk, erosion, dilation
from validation import Validation
import json
import h5py
import numpy as np
from visualize import Visualizer
import os
from tensorflow import keras
from tensorflow.keras.layers import Lambda
from scipy.interpolate import make_smoothing_spline
from skimage import util, measure
import tensorflow as tf
# from training import preprocess
# from leap.utils import find_weights, find_best_weights, preprocess
# from leap.layers import Maxima2D
from scipy.spatial.distance import cdist
from scipy.io import loadmat
# imports of the wings1 detection
from time import time
from ultralytics import YOLO
# import open3d as o3d
from scipy.signal import medfilt
from scipy.ndimage import binary_dilation, binary_closing, center_of_mass, shift, gaussian_filter, binary_opening
from datetime import date
import shutil
from skimage.morphology import convex_hull_image

# from scipy.spatial.distance import pdist
# from scipy.ndimage.measurements import center_of_mass
# from scipy.spatial import ConvexHull
# import matplotlib
# import cv2
# import preprocessor
from constants import *
import predictions_2Dto3D
import sys
from predictions_2Dto3D import From2Dto3D

sys.path.append(r'C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\predict_2D_pytorch')
from BoxSparse import BoxSparse
from traingulate import Triangulate
# initial_gpus = tf.config.list_physical_devices('GPU')
# Hide all GPUs from TensorFlow
# tf.config.set_visible_devices([], 'GPU')
# print("GPUs have been hidden.")
# print(f"Initially available GPUs: {initial_gpus}")


WHICH_TO_FLIP = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                          [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]).astype(bool)
ALL_COUPLES = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

class Predictor2D:
    def __init__(self, configuration_path):
        self.neto_wings_sparse = None
        self.wings_size = None
        self.body_masks_sparse = None
        self.run_path = None
        self.triangulation_errors = None
        self.reprojection_errors = None
        self.points_3D_all = None
        self.conf_preds = None
        self.preds_2D = None
        with open(configuration_path) as C:
            config = json.load(C)
            self.config = config
            self.triangulate = Triangulate(self.config)
            self.box_path = config["box path"]
            self.wings_pose_estimation_model_path = config["wings pose estimation model path"]
            self.wings_pose_estimation_model_path_second_pass = config["wings pose estimation model path second path"]
            self.head_tail_pose_estimation_model_path = config["head tail pose estimation model path"]
            # self.out_path = config["out path"]
            self.wings_detection_model_path = config["wings detection model path"]
            self.model_type = config["model type"]
            self.model_type_second_pass = config["model type second pass"]
            self.is_video = bool(config["is video"])
            self.batch_size = config["batch size"]
            self.points_to_predict = config["body parts to predict"]
            self.num_cams = config["number of cameras"]
            self.num_times_channels = config["number of time channels"]
            self.mask_increase_initial = config["mask increase initial"]
            self.mask_increase_reprojected = config["mask increase reprojected"]
            self.predict_again_using_reprojected_masks = bool(config["predict again using reprojected masks"])
            self.base_output_path = config["base output path"]
            self.json_2D_3D_path = config["2D to 3D config path"]

        print("creating sparse box object")
        self.box_sparse = BoxSparse(self.box_path)
        print("finish creating sparse box object")
        # self.box = self.get_box()[first_frame:last_frame]
        self.masks_flag = False
        # Visualizer.display_movie_from_box(np.copy(self.box))
        self.cropzone = self.get_cropzone()
        self.im_size = self.box_sparse.shape[2]
        self.num_frames = self.box_sparse.shape[0]
        self.num_pass = 0

        self.wings_pose_estimation_model = \
            Predictor2D.get_pose_estimation_model(self.wings_pose_estimation_model_path)
        if self.model_type != WINGS_AND_BODY_SAME_MODEL:
            self.head_tail_pose_estimation_model = \
                Predictor2D.get_pose_estimation_model(self.head_tail_pose_estimation_model_path)
        self.wings_detection_model = self.get_wings_detection_model()
        self.scores = np.zeros((self.num_frames, self.num_cams, 2))
        self.predict_method = self.choose_predict_method()

        self.run_name = self.get_run_name()

        self.total_runtime = None
        self.prediction_runtime = None
        self.predicted_points = None
        self.out_path_h5 = None

        self.num_joints = 18
        self.left_mask_ind, self.right_mask_ind = 3, 4
        self.image_size = self.box_sparse.shape[-2]
        self.num_time_channels = self.box_sparse.shape[-1] - 2
        self.num_wings_points = self.num_joints - 2
        self.num_points_per_wing = self.num_wings_points // 2
        self.left_inds = np.arange(0, self.num_points_per_wing)
        self.right_inds = np.arange(self.num_points_per_wing, self.num_wings_points)
        self.wings_pnts_inds = np.array([self.left_inds, self.right_inds])
        self.head_tail_inds = [self.num_wings_points, self.num_wings_points + 1]
        self.left_mask_ind = 3
        self.right_mask_ind = 4

    def run_predict_2D(self):
        """
        creates an array of pose estimation predictions
        """
        t0 = time()
        self.run_path = self.create_run_folders()

        if not self.masks_flag:
            print("cleaning the images")
            self.clean_images()
            print("aligning time channels")
            self.align_time_channels()

        print("find body masks")
        self.set_body_masks()
        print("finding wings sizes")
        self.get_neto_wings_masks()
        print("preprocessing masks")
        self.preprocess_masks()

        preprocessing_time = time() - t0
        preds_time = time()
        print("preprocess [%.1fs]" % preprocessing_time)
        self.predicted_points = self.predict_method()
        self.preds_2D = self.predicted_points[..., :-1]
        self.conf_preds = self.predicted_points[..., -1]

        print("finish predict")
        self.prediction_runtime = 0
        print("enforcing 3D consistency")
        self.enforce_3D_consistency()
        print("done")

        # box = self.box_sparse.retrieve_dense_box()
        # points_2D = self.preds_2D
        # Visualizer.show_predictions_all_cams(box, points_2D)

        print("predicting 3D points")
        self.points_3D_all, self.reprojection_errors, self.triangulation_errors = (
            self.get_all_3D_pnts_pairs(self.preds_2D, self.cropzone))

        print("saving")
        self.save_predictions_to_h5()
        # box = self.box_sparse.retrieve_dense_box()
        # Visualizer.show_predictions_all_cams(box, self.predicted_points)
        print("done")
        if self.predict_again_using_reprojected_masks:
            print("starting the reprojected masks creation")
            self.num_pass += 1
            self.model_type = self.model_type_second_pass
            self.predict_method = self.choose_predict_method()
            return_model_peaks = False if self.model_type == ALL_CAMS_PER_WING else True
            self.wings_pose_estimation_model = \
                self.get_pose_estimation_model(self.wings_pose_estimation_model_path_second_pass,
                                               return_model_peaks=return_model_peaks)
            points_3D = self.choose_best_score_2_cams()
            smoothed_3D = From2Dto3D.smooth_3D_points(points_3D)
            self.get_reprojection_masks(smoothed_3D, self.mask_increase_reprojected)
            print("created reprojection masks")
            print("predicting second round, now with reprojected masks")
            self.predicted_points = self.predict_method()
            self.preds_2D = self.predicted_points[..., :-1]
            self.conf_preds = self.predicted_points[..., -1]
            self.points_3D_all, self.reprojection_errors, self.triangulation_errors = (
                self.get_all_3D_pnts_pairs(self.preds_2D, self.cropzone))
            print("saving")
            self.save_predictions_to_h5()

        # Visualizer.show_predictions_all_cams(self.box, self.predicted_points)
        self.prediction_runtime = time() - preds_time
        self.total_runtime = time() - t0
        print("Predicted [%.1fs]" % self.prediction_runtime)
        print("Prediction performance: %.3f FPS" % (self.num_frames * self.num_cams / self.prediction_runtime))

    @staticmethod
    def get_median_point(all_points_3D):
        median = np.median(all_points_3D, axis=2)
        return median

    @staticmethod
    def get_validation_score(points_3D):
        return Validation.get_wings_distances_variance(points_3D)[0]

    @staticmethod
    def choose_best_reprojection_error_points(points_3D_all, reprojection_errors):
        num_frames, num_joints, _, _ = points_3D_all.shape
        points_3D = np.zeros((num_frames, num_joints, 3))
        for frame in range(num_frames):
            for joint in range(num_joints):
                candidates = points_3D_all[frame, joint, ...]
                best_candidate_ind = np.argmin(reprojection_errors[frame, joint, ...])
                point_3d = candidates[best_candidate_ind]
                points_3D[frame, joint, :] = point_3d
        return points_3D

    def choose_best_score_2_cams(self):
        """
        for each point rank the 4 different cameras by visibility, noise, size of mask, and choose the best 2
        """
        points_3D = np.zeros((self.num_frames, self.num_joints, 3))
        for frame in range(self.num_frames):
            for wing_num in range(2):
                for pnt_num in self.wings_pnts_inds[wing_num, :]:
                    candidates = self.points_3D_all[frame, pnt_num, :, :]
                    wings_size = self.wings_size[frame, :, wing_num]
                    max_size = np.max(wings_size)
                    masks_sizes_score = wings_size / max_size
                    scores = masks_sizes_score
                    cameras_ind = np.sort(np.argpartition(scores, -2)[-2:])
                    best_pair_ind = self.triangulate.all_subs.index(tuple(cameras_ind))
                    best_3D_point = candidates[best_pair_ind]
                    points_3D[frame, pnt_num, :] = best_3D_point
        # now find the
        points_3D[:, self.head_tail_inds, :] = self.choose_best_reprojection_error_points(self.points_3D_all, self.reprojection_errors)[:, self.head_tail_inds, :]
        return points_3D


    def clean_images(self):
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                for channel in range(self.num_times_channels):
                    image = self.box_sparse.get_frame_camera_channel_dense(frame, cam, channel)
                    # image = self.box[frame, cam, :, :, channel]
                    binary = np.where(image >= 0.1, 1, 0)
                    label = measure.label(binary)
                    props = measure.regionprops(label)
                    sizes = [prop.area for prop in props]
                    largest = np.argmax(sizes)
                    fly_component = np.where(label == largest + 1, 1, 0)
                    image = image * fly_component
                    # self.box[frame, cam, :, :, channel] = image
                    self.box_sparse.set_frame_camera_channel_dense(frame, cam, channel, image)

    def enforce_3D_consistency(self):
        chosen_camera = 0
        cameras_to_check = np.arange(0, 4)
        cameras_to_check = cameras_to_check[np.where(cameras_to_check != chosen_camera)]
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

    def get_reprojection_masks(self, points_3D, extend_mask_radius=5):
        points_2D_reprojected = self.triangulate.get_reprojections(points_3D, self.cropzone)
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                fly = self.box_sparse.get_frame_camera_channel_dense(frame, cam, channel_idx=1)
                for wing in range(2):
                    points_inds = self.wings_pnts_inds[wing, :]
                    mask = np.zeros((self.image_size, self.image_size))
                    wing_pnts = np.round(points_2D_reprojected[frame, cam, points_inds, :]).astype(int)
                    wing_pnts[wing_pnts >= self.image_size] = self.image_size - 1
                    wing_pnts[wing_pnts < 0] = 0
                    mask[wing_pnts[:, 1], wing_pnts[:, 0]] = 1
                    mask = convex_hull_image(mask)  # todo switch
                    mask = binary_dilation(mask, iterations=extend_mask_radius)
                    mask = np.logical_and(mask, fly)
                    mask = binary_dilation(mask, iterations=1)
                    self.box_sparse.set_frame_camera_channel_dense(frame, cam, self.num_times_channels + wing, mask)

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

    def get_all_3D_pnts_pairs(self, points_2D, cropzone):
        points_3D_all, reprojection_errors, triangulation_errors = \
            self.triangulate.triangulate_2D_to_3D_reprojection_optimization(points_2D, cropzone)
        return points_3D_all, reprojection_errors, triangulation_errors

    def find_which_cameras_to_flip(self, cameras_to_check, frame):
        num_of_options = len(WHICH_TO_FLIP)
        switch_scores = np.zeros(num_of_options, )
        cropzone = self.cropzone[frame][np.newaxis, ...]
        for i, option in enumerate(WHICH_TO_FLIP):
            points_2D = np.copy(self.preds_2D[frame])
            cameras_to_flip = cameras_to_check[option]
            for cam in cameras_to_flip:
                left_points = points_2D[cam, self.left_inds, :]
                right_points = points_2D[cam, self.right_inds, :]
                points_2D[cam, self.left_inds, :] = right_points
                points_2D[cam, self.right_inds, :] = left_points
            points_2D = points_2D[np.newaxis, ...]
            _, reprojection_errors, _ = self.get_all_3D_pnts_pairs(points_2D, cropzone)
            score = np.mean(reprojection_errors)
            switch_scores[i] = score
        cameras_to_flip = cameras_to_check[WHICH_TO_FLIP[np.argmin(switch_scores)]]
        return cameras_to_flip

    def flip_camera(self, camera_to_flip, frame):
        left_points = self.preds_2D[frame, camera_to_flip, self.left_inds, :]
        right_points = self.preds_2D[frame, camera_to_flip, self.right_inds, :]
        self.preds_2D[frame, camera_to_flip, self.left_inds, :] = right_points
        self.preds_2D[frame, camera_to_flip, self.right_inds, :] = left_points
        # switch train_masks in box
        left_mask = self.box_sparse.get_frame_camera_channel_dense(frame, camera_to_flip, self.left_mask_ind)
        right_mask = self.box_sparse.get_frame_camera_channel_dense(frame, camera_to_flip, self.right_mask_ind)
        self.box_sparse.set_frame_camera_channel_dense(frame, camera_to_flip, self.left_mask_ind, right_mask)
        self.box_sparse.set_frame_camera_channel_dense(frame, camera_to_flip, self.right_mask_ind, left_mask)
        # switch confidence scores
        left_conf_scores = self.conf_preds[frame, camera_to_flip, self.left_inds]
        right_conf_scores = self.conf_preds[frame, camera_to_flip, self.right_inds]
        self.conf_preds[frame, camera_to_flip, self.left_inds] = left_conf_scores
        self.conf_preds[frame, camera_to_flip, self.right_inds] = right_conf_scores

    def align_time_channels(self):
        all_shifts = np.zeros((self.num_frames, self.num_cams, 2, 2))
        all_shifts_smoothed = np.zeros((self.num_frames, self.num_cams, 2, 2))
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                # present = self.box[frame, cam, :, :, 1]
                present = self.box_sparse.get_frame_camera_channel_dense(frame, cam, channel_idx=1)
                cm_present = self.get_fly_cm(present)
                for i, time_channel in enumerate([0, 2]):
                    # fly = self.box[frame, cam, :, :, time_channel]
                    fly = self.box_sparse.get_frame_camera_channel_dense(frame, cam, channel_idx=time_channel)
                    CM = self.get_fly_cm(fly)
                    shift_to_do = cm_present - CM
                    all_shifts[frame, cam, i, :] = shift_to_do

        # do shiftes
        for cam in range(self.num_cams):
            for time_channel in range(all_shifts.shape[2]):
                for axis in range(all_shifts.shape[3]):
                    vals = all_shifts[:, cam, time_channel, axis]
                    A = np.arange(vals.shape[0])
                    filtered = medfilt(vals, kernel_size=11)
                    # all_shifts_smoothed[:, cam, time_channel, axis] = filtered
                    try:
                        spline = make_smoothing_spline(A, filtered, lam=10000)
                        smoothed = spline(A)
                    except:
                        smoothed = filtered
                        print(f"spline failed in cam {cam} time channel {time_channel} and axis {axis}")
                    all_shifts_smoothed[:, cam, time_channel, axis] = smoothed
                    pass

        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                for i, time_channel in enumerate([0, 2]):
                    # fly = self.box[frame, cam, :, :, time_channel]
                    fly = self.box_sparse.get_frame_camera_channel_dense(frame, cam, channel_idx=time_channel)
                    shift_to_do = all_shifts_smoothed[frame, cam, i, :]
                    shifted_fly = shift(fly, shift_to_do, order=2)
                    # self.box[frame, cam, :, :, time_channel] = shifted_fly
                    self.box_sparse.set_frame_camera_channel_dense(frame, cam, time_channel, shifted_fly)

    @staticmethod
    def get_fly_cm(im_orig):
        im = gaussian_filter(im_orig, sigma=2)
        im[im < 0.8] = 0
        # im = binary_opening(im, iterations=1)
        CM = center_of_mass(im)
        return np.array(CM)

    def preprocess_masks(self):
        if self.points_to_predict == WINGS or self.points_to_predict == WINGS_AND_BODY:
            if not self.masks_flag:
                self.add_masks()
            self.adjust_masks_size()
            if self.is_video:
                self.fix_masks()

    def get_run_name(self):
        box_path_file = os.path.basename(self.box_path)
        name, ext = os.path.splitext(box_path_file)
        run_name = f"{name}_{self.model_type}_{date.today().strftime('%b %d')}"
        return run_name

    def create_2D_3D_config(self):
        json_path = self.json_2D_3D_path
        with open(json_path, "r") as jsonFile:
            data = json.load(jsonFile)

        # Change the values of some variables
        data["2D predictions path"] = self.out_path_h5
        data["align right left"] = 1

        new_json_path = os.path.join(self.run_path, "2D_to_3D_config.json")
        # Save the JSON string to a different file
        with open(new_json_path, "w") as jsonFile:
            json.dump(data, jsonFile)
        return new_json_path

    def create_run_folders(self):
        """ Creates subfolders necessary for outputs of vision. """
        run_path = os.path.join(self.base_output_path, self.run_name)

        initial_run_path = run_path
        i = 1
        while os.path.exists(run_path):  # and not is_empty_run(run_path):
            run_path = "%s_%02d" % (initial_run_path, i)
            i += 1

        if os.path.exists(run_path):
            shutil.rmtree(run_path)

        os.makedirs(run_path)
        print("Created folder:", run_path)

        return run_path

    def save_predictions_to_h5(self):
        """ save the predictions and the box of train_images to h5 file"""
        if self.num_pass > 0:
            name = "predicted_points_and_box_reprojected.h5"
        else:
            name = "predicted_points_and_box.h5"
        self.out_path_h5 = os.path.join(self.run_path, name)
        with open(f"{self.run_path}/configuration.json", 'w') as file:
            json.dump(self.config, file, indent=4)
        with h5py.File(self.out_path_h5, "w") as f:
            f.attrs["num_frames"] = self.box_sparse.shape[0]
            f.attrs["img_size"] = self.im_size
            f.attrs["box_path"] = self.box_path
            f.attrs["box_dset"] = "/box"
            f.attrs["pose_estimation_model_path"] = self.wings_pose_estimation_model_path
            f.attrs["wings_detection_model_path"] = self.wings_detection_model_path

            positions = self.predicted_points[..., :2]
            confidence_val = self.predicted_points[..., 2]

            ds_pos = f.create_dataset("positions_pred", data=positions.astype("int32"), compression="gzip",
                                      compression_opts=1)
            ds_pos.attrs["description"] = "coordinate of peak at each sample"
            ds_pos.attrs["dims"] = "(sample, joint, [x, y])"

            ds_conf = f.create_dataset("conf_pred", data=confidence_val.squeeze(), compression="gzip",
                                       compression_opts=1)
            ds_conf.attrs["description"] = "confidence map value in [0, 1.0] at peak"
            ds_conf.attrs["dims"] = "(frame, cam, joint)"
            if self.num_frames < 2000:
                box = self.box_sparse.retrieve_dense_box()
                ds_conf = f.create_dataset("box", data=box, compression="gzip", compression_opts=1)
                ds_conf.attrs["description"] = "The predicted box and the wings1 if the wings1 were detected"
                ds_conf.attrs["dims"] = f"{box.shape}"

            if self.points_to_predict == WINGS or self.points_to_predict == WINGS_AND_BODY:
                ds_conf = f.create_dataset("scores", data=self.scores, compression="gzip", compression_opts=1)
                ds_conf.attrs["description"] = "the score (0->1) assigned to each wing during wing detection"
                ds_conf.attrs["dims"] = f"{self.scores.shape}"

            ds_conf = f.create_dataset("cropzone", data=self.cropzone, compression="gzip", compression_opts=1)
            ds_conf.attrs["description"] = "cropzone of every image for 2D to 3D projection"
            ds_conf.attrs["dims"] = f"{self.cropzone.shape}"

            ds_conf = f.create_dataset("points_3D_all", data=self.points_3D_all, compression="gzip", compression_opts=1)
            ds_conf.attrs["description"] = "all the points triangulations"
            ds_conf.attrs["dims"] = f"{self.points_3D_all.shape}"

            ds_conf = f.create_dataset("reprojection_errors", data=self.reprojection_errors, compression="gzip", compression_opts=1)
            ds_conf.attrs["dims"] = f"{self.reprojection_errors.shape}"

            ds_conf = f.create_dataset("triangulation_errors", data=self.triangulation_errors, compression="gzip", compression_opts=1)
            ds_conf.attrs["dims"] = f"{self.triangulation_errors.shape}"

    def choose_predict_method(self):
        if self.points_to_predict == WINGS:
            return self.predict_wings
        elif self.points_to_predict == BODY:
            return self.predict_body
        elif self.points_to_predict == WINGS_AND_BODY:
            if self.model_type == WINGS_AND_BODY_SAME_MODEL:
                return self.predict_wings_and_body_same_model
            if self.model_type == ALL_POINTS or self.model_type == ALL_POINTS_REPROJECTED_MASKS:
                return self.predict_all_points
            if self.model_type == ALL_CAMS_PER_WING:
                return self.predict_all_cams_per_wing
            return self.predict_wings_and_body

    def predict_all_cams_per_wing(self, n=100):
        print(f"started predicting projected masks, split box into {n} parts")
        all_points = []
        all_frames = np.arange(self.num_frames)
        n = min(self.num_frames, n)
        splited_frames = np.array_split(all_frames, n)
        for i in range(n):
            print(f"predicting part number {i + 1}")
            all_points_i = []
            for wing in range(2):
                input_wing_cams = []
                for cam_idx in range(self.num_cams):
                    input_wing_cam = self.box_sparse.get_camera_dense(camera_idx=cam_idx,
                                                                      channels=[0, 1, 2,
                                                                                self.num_times_channels + wing],
                                                                      frames=splited_frames[i])
                    input_wing_cams.append(input_wing_cam)
                input_wing = np.concatenate(input_wing_cams, axis=-1)
                output = self.wings_pose_estimation_model(input_wing)
                peaks = self.tf_find_peaks(output)
                peaks_list = [peaks[..., 0:10],
                              peaks[..., 10:20],
                              peaks[..., 20:30],
                              peaks[..., 30:40]]
                for cam in range(self.num_cams):
                    peaks_list[cam] = np.expand_dims(peaks_list[cam], axis=1)
                peaks_wing = np.concatenate(peaks_list, axis=1)
                all_points_i.append(peaks_wing)
            all_points_i = np.concatenate((all_points_i[0], all_points_i[1]), axis=-1)
            tail_points = all_points_i[..., [8, 18]]
            tail_points = np.expand_dims(np.mean(tail_points, axis=-1), axis=-1)
            head_points = all_points_i[..., [9, 19]]
            head_points = np.expand_dims(np.mean(head_points, axis=-1), axis=-1)
            wings_points = all_points_i[..., [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17]]
            wings_and_body_pnts = np.concatenate((wings_points, tail_points, head_points), axis=-1)
            wings_and_body_pnts = np.transpose(wings_and_body_pnts, [0, 1, 3, 2])
            all_points.append(wings_and_body_pnts)
        all_wing_and_body_points = np.concatenate(all_points, axis=0)
        print("done predicting projected masks")
        return all_wing_and_body_points

    def predict_all_points(self):
        all_points = []
        for cam in range(self.num_cams):
            input = self.box[:, cam, ...]
            points_cam_i, _, _, _ = self.predict_Ypk(input, self.batch_size, self.wings_pose_estimation_model)
            all_points.append(points_cam_i[np.newaxis, ...])
        wings_and_body_pnts = np.concatenate(all_points, axis=0)
        wings_and_body_pnts = np.transpose(wings_and_body_pnts, [1, 0, 3, 2])
        return wings_and_body_pnts

    def predict_wings_and_body_same_model(self):
        all_pnts = self.predict_wings()
        tail_points = all_pnts[:, :, [8, 18], :]
        tail_points = np.expand_dims(np.mean(tail_points, axis=2), axis=2)
        head_points = all_pnts[:, :, [9, 19], :]
        head_points = np.expand_dims(np.mean(head_points, axis=2), axis=2)
        wings_points = all_pnts[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17], :]
        wings_and_body_pnts = np.concatenate((wings_points, tail_points, head_points), axis=2)
        return wings_and_body_pnts

    def predict_wings_and_body(self):
        wings_points = self.predict_wings()
        body_points = self.predict_body()
        wings_and_body_pnts = np.concatenate((wings_points, body_points), axis=2)
        return wings_and_body_pnts

    def predict_wings(self, n=100):
        Ypks = []
        all_frames = np.arange(self.num_frames)
        n = min(n, self.num_frames)
        splited_frames = np.array_split(all_frames, n)
        for cam in range(self.num_cams):
            print(f"predict camera {cam + 1}")
            Ypks_per_wing = []
            for wing in range(2):
                # split for memory limit
                Ypk = []
                for i in range(n):
                    input_i = self.box_sparse.get_camera_dense(cam, channels=[0, 1, 2, self.num_times_channels + wing],
                                                             frames=splited_frames[i])
                    Ypk_i, _, _, _ = self.predict_Ypk(input_i, self.batch_size, self.wings_pose_estimation_model)
                    Ypk.append(Ypk_i)
                Ypk = np.concatenate(Ypk, axis=0)
                Ypks_per_wing.append(Ypk)
            Ypk_cam = np.concatenate((Ypks_per_wing[0], Ypks_per_wing[1]), axis=-1)
            Ypk_cam = np.expand_dims(Ypk_cam, axis=1)
            Ypks.append(Ypk_cam)
        Ypk_all = np.concatenate(Ypks, axis=1)
        Ypk_all = np.transpose(Ypk_all, [0, 1, 3, 2])
        return Ypk_all

    def predict_body(self):
        Ypks = []
        for cam in range(self.num_cams):
            input = self.box[:, cam, :, :, :self.num_times_channels]
            Ypk_cam, _, _, _ = self.predict_Ypk(input, self.batch_size, self.head_tail_pose_estimation_model)
            Ypk_cam = np.expand_dims(Ypk_cam, axis=1)
            Ypks.append(Ypk_cam)
        Ypk_all = np.concatenate(Ypks, axis=1)
        Ypk_all = np.transpose(Ypk_all, [0, 1, 3, 2])
        return Ypk_all

    def add_masks(self, n=100):
        """ Add train_masks to the dataset using yolov8 segmentation model """
        all_frames = np.arange(self.num_frames)
        n = min(n, self.num_frames)
        all_frames_split = np.array_split(all_frames, n)
        for cam in range(self.num_cams):
            print(f"finds wings for camera number {cam + 1}")
            results = []
            for i in range(n):
                print(f"processing n = {i}")
                img_3_ch_i = self.box_sparse.get_camera_dense(cam, [0, 1, 2], frames=all_frames_split[i])
                img_3_ch_input = np.round(img_3_ch_i * 255)
                img_3_ch_input = [img_3_ch_input[i] for i in range(img_3_ch_input.shape[0])]
                with tf.device('/CPU:0'):  # Forces the operation to run on the CPU
                    results_i = self.wings_detection_model(img_3_ch_input)
                results.append(results_i)
            results = sum(results, [])
            for frame in range(self.num_frames):
                masks_2 = np.zeros((self.im_size, self.im_size, 2))
                result = results[frame]
                boxes = result.boxes.data.numpy()
                inds_to_keep = self.eliminate_close_vectors(boxes, 10)
                num_wings_found = np.count_nonzero(inds_to_keep)
                if num_wings_found > 0:
                    masks_found = result.masks.data.numpy()[inds_to_keep, :, :]
                else:
                    assert f"no masks found for this frame {frame} and camera {cam}"
                for wing in range(min(num_wings_found, 2)):
                    mask = masks_found[wing, :, :]
                    masks_2[:, :, wing] = mask
                self.box_sparse.set_camera_dense(camera_idx=cam, frames=[frame],
                                                 dense_camera_data=masks_2[np.newaxis, ...], channels=[3, 4])

    def adjust_masks_size(self):
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                mask_1 = self.box_sparse.get_frame_camera_channel_dense(frame, cam, self.num_times_channels)
                mask_2 = self.box_sparse.get_frame_camera_channel_dense(frame, cam, self.num_times_channels + 1)
                mask_1 = self.adjust_mask(mask_1, self.mask_increase_initial)
                mask_2 = self.adjust_mask(mask_2, self.mask_increase_initial)
                self.box_sparse.set_frame_camera_channel_dense(frame, cam, self.num_times_channels, mask_1)
                self.box_sparse.set_frame_camera_channel_dense(frame, cam, self.num_times_channels + 1, mask_2)

    def fix_masks(self):  # todo find out if there are even train_masks to be fixed
        """
            goes through each frame, if there is no mask for a specific wing, unite train_masks of the closest times before and after
            this frame.
            :param X: a box of size (num_frames, 20, 192, 192)
            :return: same box
            """
        search_range = 5
        problematic_masks = []
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                for mask_num in range(2):
                    # mask = self.box[frame, cam, :, :, self.num_times_channels + mask_num]
                    mask = self.box_sparse.get_frame_camera_channel_dense(frame, cam,
                                                                          self.num_times_channels + mask_num)
                    if np.all(mask == 0):  # check if all 0:
                        problematic_masks.append((frame, cam, mask_num))
                        # find previous matching mask
                        prev_mask = np.zeros(mask.shape)
                        next_mask = np.zeros(mask.shape)
                        for prev_frame in range(frame - 1, max(0, frame - search_range - 1), -1):
                            prev_mask_i = self.box_sparse.get_frame_camera_channel_dense(prev_frame, cam,
                                                                                         self.num_times_channels + mask_num)
                            if not np.all(prev_mask_i == 0):  # there is a good mask
                                prev_mask = prev_mask_i
                                break
                        # find next matching mask
                        for next_frame in range(frame + 1, min(self.num_frames, frame + search_range)):
                            next_mask_i = self.box_sparse.get_frame_camera_channel_dense(next_frame, cam,
                                                                                         self.num_times_channels + mask_num)
                            if not np.all(next_mask_i == 0):  # there is a good mask
                                next_mask = next_mask_i
                                break
                        # combine the 2 train_masks

                        new_mask = prev_mask + next_mask
                        new_mask[new_mask >= 1] = 1

                        sz_prev_mask = np.count_nonzero(prev_mask)
                        sz_next_mask = np.count_nonzero(next_mask)
                        sz_new_mask = np.count_nonzero(new_mask)
                        if sz_prev_mask + sz_next_mask == sz_new_mask:
                            # it means that the train_masks are not overlapping
                            new_mask = prev_mask if sz_prev_mask > sz_next_mask else next_mask

                        # replace empty mask with new mask
                        self.box_sparse.set_frame_camera_channel_dense(frame, cam, self.num_times_channels + mask_num,
                                                                       new_mask)

    def get_cropzone(self):
        return h5py.File(self.box_path, "r")["/cropzone"]

    def set_body_masks(self, opening_rad=6):
        """
        find the fly's body, and the distance transform for later analysis in every camera in 2D using segmentation
        """
        self.body_masks_sparse = BoxSparse(box_path=None, shape=(self.num_frames, self.num_cams, self.image_size, self.image_size, 1))
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                # fly_3_ch = self.box[frame, cam, :, :, :self.num_time_channels]
                fly_3_ch = np.zeros((self.image_size, self.image_size, 3))
                fly_3_ch[..., 0] = self.box_sparse.get_frame_camera_channel_dense(frame, cam, 0)
                fly_3_ch[..., 1] = self.box_sparse.get_frame_camera_channel_dense(frame, cam, 1)
                fly_3_ch[..., 2] = self.box_sparse.get_frame_camera_channel_dense(frame, cam, 2)

                fly_3_ch_av = np.sum(fly_3_ch, axis=-1) / self.num_time_channels
                binary_body = fly_3_ch_av >= 0.7
                selem = disk(opening_rad)
                # Perform dilation
                dilated = dilation(binary_body, selem)
                # Perform erosion
                mask = erosion(dilated, selem)
                # body_masks[frame, cam, ...] = mask
                self.body_masks_sparse.set_frame_camera_channel_dense(frame, cam, 0, mask)

    def get_neto_wings_masks(self):
        self.neto_wings_sparse = BoxSparse(box_path=None, shape=(self.num_frames, self.num_cams, self.image_size, self.image_size, 2))
        self.wings_size = np.zeros((self.num_frames, self.num_cams, 2))
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                body_mask = self.body_masks_sparse.get_frame_camera_channel_dense(frame, cam, 0)
                for wing_num in range(2):
                    other_wing_mask = self.box_sparse.get_frame_camera_channel_dense(frame, cam, self.num_time_channels + (not wing_num))
                    wing_mask = self.box_sparse.get_frame_camera_channel_dense(frame, cam, self.num_time_channels + wing_num)
                    body_and_other_wing_mask = np.bitwise_or(body_mask.astype(bool), other_wing_mask.astype(bool))
                    intersection = np.logical_and(wing_mask, body_and_other_wing_mask)
                    neto_wing = wing_mask - intersection
                    self.neto_wings_sparse.set_frame_camera_channel_dense(frame, cam, wing_num, neto_wing)
                    self.wings_size[frame, cam, wing_num] = np.count_nonzero(neto_wing)

    def get_wings_detection_model(self):
        """ load a pretrained YOLOv8 segmentation model"""
        if self.num_pass == 0:
            model = YOLO(self.wings_detection_model_path)
            model.fuse()
        elif self.num_pass == 1:
            model = YOLO(self.wings_pose_estimation_model_path_second_pass)
            model.fuse()
        try:
            return model.cpu()
        except:
            return model


    @staticmethod
    def adjust_mask(mask, radius=3):
        mask = binary_closing(mask).astype(int)
        mask = binary_dilation(mask, iterations=radius).astype(int)
        return mask

    @staticmethod
    def get_pose_estimation_model(pose_estimation_model_path, return_model_peaks=True):
        """ load a pretrained LEAP pose estimation model model"""
        model = keras.models.load_model(pose_estimation_model_path)
        if return_model_peaks:
            model = Predictor2D.convert_to_peak_outputs(model, include_confmaps=False)
        print("weights_path:", pose_estimation_model_path)
        print("Loaded model: %d layers, %d params" % (len(model.layers), model.count_params()))
        return model

    @staticmethod
    def convert_to_peak_outputs(model, include_confmaps=False):
        """ Creates a new Keras model with a wrapper to yield channel peaks from rank-4 tensors. """
        if type(model.output) == list:
            confmaps = model.output[-1]
        else:
            confmaps = model.output

        if include_confmaps:
            return keras.Model(model.input, [Lambda(Predictor2D.tf_find_peaks)(confmaps), confmaps])
        else:
            return keras.Model(model.input, Lambda(Predictor2D.tf_find_peaks)(confmaps))


    @staticmethod
    def tf_find_peaks(x):
        """ Finds the maximum value in each channel and returns the location and value.
        Args:
            x: rank-4 tensor (samples, height, width, channels)

        Returns:
            peaks: rank-3 tensor (samples, [x, y, val], channels)
        """

        # Store input shape
        in_shape = tf.shape(x)

        # Flatten height/width dims
        flattened = tf.reshape(x, [in_shape[0], -1, in_shape[-1]])

        # Find peaks in linear indices
        idx = tf.argmax(flattened, axis=1)

        # Convert linear indices to subscripts
        rows = tf.math.floordiv(tf.cast(idx, tf.int32), in_shape[1])
        cols = tf.math.floormod(tf.cast(idx, tf.int32), in_shape[1])

        # Dumb way to get actual values without indexing
        vals = tf.math.reduce_max(flattened, axis=1)

        # Return N x 3 x C tensor
        pred = tf.stack([
            tf.cast(cols, tf.float32),
            tf.cast(rows, tf.float32),
            vals],
            axis=1)
        return pred

    @staticmethod
    def predict_Ypk(X, batch_size, model_peaks, save_confmaps=False):
        """ returns a predicted dataset"""
        confmaps, confmaps_min, confmaps_max = None, None, None
        if save_confmaps:
            Ypk, confmaps = model_peaks.predict(X, batch_size=batch_size)

            # Quantize
            confmaps_min = confmaps.min()
            confmaps_max = confmaps.max()

            # Reshape
            confmaps = np.transpose(confmaps, (0, 3, 2, 1))
        else:
            Ypk = model_peaks.predict(X, batch_size=batch_size)
        return Ypk, confmaps, confmaps_min, confmaps_max

    @staticmethod
    def eliminate_close_vectors(matrix, threshold):
        # calculate pairwise Euclidean distances
        distances = cdist(matrix, matrix, 'euclidean')

        # create a mask to identify which vectors to keep
        inds_to_del = np.ones(len(matrix), dtype=bool)
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                if distances[i, j] < threshold:
                    # eliminate one of the vectors
                    inds_to_del[j] = False

        # return the new matrix with close vectors eliminated
        return inds_to_del


