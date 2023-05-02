import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.layers import Lambda
import tensorflow as tf
# from training import preprocess
# from leap.utils import find_weights, find_best_weights, preprocess
# from leap.layers import Maxima2D
from scipy.spatial.distance import cdist
from scipy.io import loadmat
# imports of the wings detection
from time import time
from ultralytics import YOLO
# import open3d as o3d
from scipy.ndimage import binary_dilation, binary_closing
from scipy.spatial.distance import pdist
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial import ConvexHull
import matplotlib

import preprocessor
from traingulation.triangulation_2D_3D import calculate_all_possible_triangulations
from constants import *

TWO_WINGS_TOGATHER = "TWO_WINGS_TOGATHER"
PER_WING = "PER WING"
ALL_POINTS = "ALL POINTS"
BODY_POINTS = "BODY_POINTS"
ALL_CAMS = "ALL_CAMS"

PER_WING_SPLIT_TO_3 = "PER WING SPLIT TO 3"
PER_WING_PER_POINT = "PER_WING_PER_POINT"
ENSEMBLE = "ENSEMBLE"
NO_MASKS = "NO_MASKS"
HEAD_TAIL = "NO_MASKS"
NUM_CAMS = 4
NUM_CHANNELS_PER_IMAGE = 5
CAMERAS_MATRICES_PATH = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\traingulation\datasets\projection_matrices.mat"
ROTATION_MATRIX_PATH = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\traingulation\datasets\rotation_mat.mat"

CAM_1 = [0, 1, 2, 3, 4]
CAM_2 = [5, 6, 7, 8, 9]
CAM_3 = [10, 11, 12, 13, 14]
CAM_4 = [15, 16, 17, 18, 19]
LEFT_INDS = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18]
RIGHT_INDS = [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19]
CAM1_LEFT = [0, 1, 2, 3]
CAM2_LEFT = [5, 6, 7, 8]
CAM3_LEFT = [10, 11, 12, 13]
CAM4_LEFT = [15, 16, 17, 18]
CAM1_RIGHT = [0, 1, 2, 4]
CAM2_RIGHT = [5, 6, 7, 9]
CAM3_RIGHT = [10, 11, 12, 14]
CAM4_RIGHT = [15, 16, 17, 19]


# def adjust_masks_size(box, train_or_predict):
#     """ adjust the size of the wings masks """
#     if train_or_predict == "TRAIN":
#         num_training_samples = box.shape[0]
#         for image_num in range(num_training_samples):
#             mask = box[image_num, :, :, 3]
#             # non_0_1 = np.count_nonzero(mask)
#             adjusted_mask = adjust_mask(mask)
#             # non_0_2 = np.count_nonzero(adjusted_mask)
#             box[image_num, :, :, 3] = adjusted_mask
#             # matplotlib.use('TkAgg')
#             # plt.imshow(adjusted_mask - mask)
#             # plt.show()
#     if train_or_predict == "PREDICT":
#         pass
#     return box


class Predictions:

    def __init__(self,
                 box_path: str,
                 model_type: str,
                 out_path: str,
                 pose_estimation_model_path: str,
                 wings_detection_model_path: str,
                 cameras_path: str = CAMERAS_MATRICES_PATH,
                 is_video: bool = True,
                 threshold: float = 2,
                 times_channels=3):

        self.box_path = box_path
        self.model_type = model_type
        self.pose_estimation_model_path = pose_estimation_model_path
        self.out_path = out_path
        self.wings_detection_model_path = wings_detection_model_path
        self.is_video = is_video
        self.cam_matrices = loadmat(CAMERAS_MATRICES_PATH)['projection_matrices']
        self.rotation_matrix = loadmat(ROTATION_MATRIX_PATH)['rotation_mat']
        # self.desired_times_channels = times_channels
        # self.time_span = 7 if self.desired_times_channels == 3 else 14
        self.box = self.get_box()
        # self.init_times_channels = self.box.shape[1] // NUM_CAMS

        # debug
        # self.box = self.box[:20, :, :, :]
        # train set only
        # self.box = self.box[:, [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17, ], :, :]

        # if self.init_times_channels == 1:
        #     self.add_times_channels()
        self.cropzone = self.get_cropzone()
        self.pose_estimation_model = Predictions.get_pose_estimation_model(self.pose_estimation_model_path)
        self.wings_detection_model = self.get_wings_detection_model()

        self.im_size = self.box.shape[-1]
        self.num_frames = self.box.shape[0]
        self.num_cams = NUM_CAMS
        self.num_times_channels = self.box.shape[1] // self.num_cams
        # self.num_times_channels = 3  # todo remove this line, only for trainset
        self.num_channels = self.num_times_channels + 2
        self.threshold = threshold

        self.box_to_save = None
        self.Ypk = None
        self.left_input = None
        self.right_input = None
        self.prediction_runtime = None
        self.total_runtime = None

    def add_times_channels(self):
        times = np.linspace(-self.time_span, self.time_span, self.desired_times_channels).astype(int)
        new_box = np.zeros(
            (self.box.shape[0], self.box.shape[1] * self.desired_times_channels, self.box.shape[2], self.box.shape[3]))
        for frame in range(self.time_span, self.box.shape[0] - self.time_span):
            for cam in range(self.box.shape[1]):
                new_img_with_time_channels = self.box[frame + times, cam, :, :]
                new_box[frame, cam * self.desired_times_channels + np.arange(self.desired_times_channels), :,
                :] = new_img_with_time_channels
        self.box = new_box

    def get_box(self):
        return h5py.File(self.box_path, "r")["/box"]

    def get_cropzone(self):
        return h5py.File(self.box_path, "r")["/cropzone"]

    @staticmethod
    def get_pose_estimation_model(pose_estimation_model_path):
        """ load a pretrained LEAP pose estimation model model"""
        model = keras.models.load_model(pose_estimation_model_path)
        model_peaks = Predictions.convert_to_peak_outputs(model, include_confmaps=False)
        print("weights_path:", pose_estimation_model_path)
        print("Loaded model: %d layers, %d params" % (len(model.layers), model.count_params()))
        return model_peaks

    def get_wings_detection_model(self):
        """ load a pretrained YOLOv8 segmentation model"""
        model = YOLO(self.wings_detection_model_path)
        model.fuse()
        return model

    def preprocess_box(self):
        # Add singleton dim for single images
        if self.box.ndim == 3:
            self.box = self.box[None, ...]
        if self.box.dtype == "uint8" or np.max(self.box) > 1:
            self.box = self.box.astype("float32") / 255

    def add_masks(self):
        """ Add masks to the dataset using yolov8 segmentation model """
        new_box = np.zeros((self.num_frames, self.num_cams * self.num_channels, self.im_size, self.im_size))
        if self.num_times_channels == 3:
            for frame in range(self.num_frames):
                for cam in range(self.num_cams):
                    print(f"frame number {frame}, cam number {cam}")
                    img_3_ch = self.box[frame, np.array([0, 1, 2]) + self.num_times_channels * cam, :, :]
                    img_3_ch = np.transpose(img_3_ch, [1, 2, 0])
                    masks = self.get_masks(img_3_ch)
                    new_box[frame, self.num_times_channels + np.array([0, 1]) + self.num_channels * cam, :, :] = masks
                    new_box[frame, np.array([0, 1, 2]) + self.num_channels * cam, :, :] = np.transpose(img_3_ch,
                                                                                                       [2, 0, 1])
                    # show the image
                    # if frame >= 6:
                    #     imtoshow = new_box[frame, np.array([0, 1, 2]) + self.num_channels * cam, :, :]
                    #     imtoshow = np.transpose(imtoshow, [1, 2, 0])
                    #     mask_1 = new_box[frame, 3 + self.num_channels * cam, :, :]
                    #     mask_2 = new_box[frame, 4 + self.num_channels * cam, :, :]
                    #     imtoshow[:, :, 1] += mask_1
                    #     imtoshow[:, :, 1] += mask_2
                    #     matplotlib.use('TkAgg')
                    #     plt.imshow(imtoshow)
                    #     plt.show()

        elif self.num_times_channels == 5:
            for frame in range(self.num_frames):
                for cam in range(self.num_cams):
                    print(f"frame number {frame}, cam number {cam}")
                    img_5_ch = self.box[frame, np.array([0, 1, 2, 3, 4]) + self.num_times_channels * cam, :, :]
                    img_5_ch = np.transpose(img_5_ch, [1, 2, 0])
                    img_3_ch = img_5_ch[:, :, [1, 2, 3]]
                    masks = self.get_masks(img_3_ch)
                    new_box[frame, self.num_times_channels + np.array([0, 1]) + self.num_channels * cam, :, :] = masks
                    new_box[frame, np.array([0, 1, 2, 3, 4]) + self.num_channels * cam, :, :] = np.transpose(img_5_ch,
                                                                                                             [2, 0, 1])
                    # show the image
                    # imtoshow = new_box[frame, np.array([0, 2, 4]) + self.num_channels * cam, :, :]
                    # imtoshow = np.transpose(imtoshow, [1, 2, 0])
                    # mask_1 = new_box[frame, 5 + self.num_channels * cam, :, :]
                    # mask_2 = new_box[frame, 6 + self.num_channels * cam, :, :]
                    # imtoshow[:, :, 1] += mask_1
                    # imtoshow[:, :, 1] += mask_2
                    # matplotlib.use('TkAgg')
                    # plt.imshow(imtoshow)
                    # plt.show()
        self.box = new_box

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

    def get_masks(self, img_3_ch):
        net_input = img_3_ch
        masks_2 = np.zeros((2, self.im_size, self.im_size))
        if np.max(img_3_ch) <= 1:
            net_input = np.round(255 * img_3_ch)
        results = self.wings_detection_model(net_input)[0]

        # find if the masks detected are overlapping
        boxes = results.boxes.data.numpy()
        inds_to_keep = self.eliminate_close_vectors(boxes, 10)

        # get number of masks detected
        num_wings_found = np.count_nonzero(inds_to_keep)
        if num_wings_found > 0:
            masks_found = results.masks.data.numpy()[inds_to_keep, :, :]
        # add masks
        for wing in range(min(num_wings_found, 2)):
            mask = masks_found[wing, :, :]
            score = results.boxes.data[wing, 4]
            masks_2[wing, :, :] = mask
            # else:
            # print(f"score = {score}")
            # matplotlib.use('TkAgg')
            # img_3_ch[:, :, 2] += mask
            # plt.imshow(img_3_ch)
            # plt.show()
        return masks_2

    def fix_masks(self):  # todo find out if there are even masks to be fixed
        """
            goes throw each frame, if there is no mask for a specific wing, unite masks of the closest times before and after
            this frame.
            :param X: a box of size (num_frames, 20, 192, 192)
            :return: same box
            """
        search_range = 5
        problematic_masks = []
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                for mask_num in range(2):
                    mask = self.box[frame, self.num_times_channels + mask_num + self.num_channels * cam, :, :]
                    if np.all(mask == 0):  # check if all 0:
                        problematic_masks.append((frame, cam, mask_num))
                        # find previous matching mask
                        prev_mask = np.zeros(mask.shape)
                        next_mask = np.zeros(mask.shape)
                        for prev_frame in range(frame - 1, max(0, frame - search_range - 1), -1):
                            prev_mask_i = self.box[prev_frame,
                                          self.num_times_channels + mask_num + self.num_channels * cam, :, :]
                            if not np.all(prev_mask_i == 0):  # there is a good mask
                                prev_mask = prev_mask_i
                                break
                        # find next matching mask
                        for next_frame in range(frame + 1, min(self.num_frames, frame + search_range)):
                            next_mask_i = self.box[next_frame,
                                          self.num_times_channels + mask_num + self.num_channels * cam, :, :]
                            if not np.all(next_mask_i == 0):  # there is a good mask
                                next_mask = next_mask_i
                                break
                        # combine the 2 masks

                        new_mask = prev_mask + next_mask  # todo changed it from : prev_mask + next_mask
                        new_mask[new_mask >= 1] = 1

                        sz_prev_mask = np.count_nonzero(prev_mask)
                        sz_next_mask = np.count_nonzero(next_mask)
                        sz_new_mask = np.count_nonzero(new_mask)
                        if sz_prev_mask + sz_next_mask == sz_new_mask:
                            # it means that the masks are not overlapping
                            new_mask = prev_mask if sz_prev_mask > sz_next_mask else next_mask

                        # replace empty mask with new mask
                        self.box[frame, self.num_times_channels + mask_num + self.num_channels * cam, :, :] = new_mask

        # do dilation to all masks

    def adjust_dimensions(self):  # todo: do it in a smarter less ugly way
        """ return a box with dimensions (num_frames, imsize, imsize, num_channels)"""
        if self.model_type == ALL_CAMS:
            # make input:
            self.box = np.transpose(self.box, [0, 2, 3, 1])
            self.left_input = self.box[..., LEFT_INDS]
            self.right_input = self.box[..., RIGHT_INDS]
            return
        if self.model_type == ALL_CAMS_AND_3_GOOD_CAMS:
            self.box = np.transpose(self.box, [0, 2, 3, 1])
            self.left_input = self.box[..., LEFT_INDS]
            self.right_input = self.box[..., RIGHT_INDS]

            x1 = np.expand_dims(self.left_input[:, :, :, 0:4], axis=1)
            x2 = np.expand_dims(self.left_input[:, :, :, 4:8], axis=1)
            x3 = np.expand_dims(self.left_input[:, :, :, 8:12], axis=1)
            x4 = np.expand_dims(self.left_input[:, :, :, 12:16], axis=1)
            self.left_input = np.concatenate((x1, x2, x3, x4), axis=1)

            x1 = np.expand_dims(self.right_input[:, :, :, 0:4], axis=1)
            x2 = np.expand_dims(self.right_input[:, :, :, 4:8], axis=1)
            x3 = np.expand_dims(self.right_input[:, :, :, 8:12], axis=1)
            x4 = np.expand_dims(self.right_input[:, :, :, 12:16], axis=1)
            self.right_input = np.concatenate((x1, x2, x3, x4), axis=1)

            self.left_input_3_cams, _, self.left_input_small_wing_cam, _, self.left_input_small_wing_cams_inds = preprocessor.Preprocessor.take_n_good_cameras(
                self.left_input, np.zeros(self.left_input.shape), 3)
            self.right_input_3_cams, _, self.right_input_small_wing_cam, _, self.right_input_small_wing_cams_inds = preprocessor.Preprocessor.take_n_good_cameras(
                self.right_input, np.zeros(self.right_input.shape), 3)
            self.left_input_3_cams = np.concatenate((self.left_input_3_cams[:, 0, ...],
                                                     self.left_input_3_cams[:, 1, ...],
                                                     self.left_input_3_cams[:, 2, ...]), axis=-1)
            self.right_input_3_cams = np.concatenate((self.right_input_3_cams[:, 0, ...],
                                                      self.right_input_3_cams[:, 1, ...],
                                                      self.right_input_3_cams[:, 2, ...]), axis=-1)
            return

        self.box = np.transpose(self.box, (0, 3, 2, 1))
        if self.box.shape[-1] == 20:  # if number of channels is 5
            x1 = np.expand_dims(self.box[:, :, :, 0:5], axis=1)
            x2 = np.expand_dims(self.box[:, :, :, 5:10], axis=1)
            x3 = np.expand_dims(self.box[:, :, :, 10:15], axis=1)
            x4 = np.expand_dims(self.box[:, :, :, 15:20], axis=1)
            self.box = np.concatenate((x1, x2, x3, x4), axis=1)

        if self.box.shape[-1] == 28:  # if number of channels is 7
            x1 = np.expand_dims(self.box[:, :, :, 0:7], axis=1)
            x2 = np.expand_dims(self.box[:, :, :, 7:14], axis=1)
            x3 = np.expand_dims(self.box[:, :, :, 14:21], axis=1)
            x4 = np.expand_dims(self.box[:, :, :, 21:28], axis=1)
            self.box = np.concatenate((x1, x2, x3, x4), axis=1)

    def save_predictions_to_h5(self):
        """ save the predictions and the box of images to h5 file"""
        with h5py.File(self.out_path, "w") as f:
            f.attrs["num_samples"] = self.box.shape[0]
            f.attrs["img_size"] = self.im_size
            f.attrs["box_path"] = self.box_path
            f.attrs["box_dset"] = "/box"
            f.attrs["pose_estimation_model_path"] = self.pose_estimation_model_path
            f.attrs["wings_detection_model_path"] = self.wings_detection_model_path

            # positions = self.Ypk[:, :2, :]
            # confidence_val = self.Ypk[:, 2, :]
            positions = self.Ypk[:, :, :2, :]
            confidence_val = self.Ypk[:, :, 2, :]

            # positions = np.transpose(positions, (0,2,1))
            ds_pos = f.create_dataset("positions_pred", data=positions.astype("int32"), compression="gzip",
                                      compression_opts=1)
            ds_pos.attrs["description"] = "coordinate of peak at each sample"
            ds_pos.attrs["dims"] = "(sample, [x, y], joint) === (sample, [column, row], joint)"

            ds_conf = f.create_dataset("conf_pred", data=confidence_val.squeeze(), compression="gzip",
                                       compression_opts=1)
            ds_conf.attrs["description"] = "confidence map value in [0, 1.0] at peak"
            ds_conf.attrs["dims"] = "(sample, joint)"

            ds_conf = f.create_dataset("box", data=self.box_to_save, compression="gzip", compression_opts=1)
            ds_conf.attrs["description"] = "The predicted box"
            ds_conf.attrs["dims"] = f"{self.box_to_save.shape}"

            ds_conf = f.create_dataset("cropzone", data=self.cropzone, compression="gzip", compression_opts=1)
            ds_conf.attrs["description"] = "cropzone of every image for 2D to 3D projection"
            ds_conf.attrs["dims"] = f"{self.cropzone.shape}"
            f.attrs["prediction_runtime_secs"] = self.prediction_runtime
            f.attrs["total_runtime_secs"] = self.total_runtime

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
    def convert_to_peak_outputs(model, include_confmaps=False):
        """ Creates a new Keras model with a wrapper to yield channel peaks from rank-4 tensors. """
        if type(model.output) == list:
            confmaps = model.output[-1]
        else:
            confmaps = model.output

        if include_confmaps:
            return keras.Model(model.input, [Lambda(Predictions.tf_find_peaks)(confmaps), confmaps])
        else:
            return keras.Model(model.input, Lambda(Predictions.tf_find_peaks)(confmaps))

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
    def adjust_mask(mask, radius=3):
        mask = binary_closing(mask).astype(int)
        mask = binary_dilation(mask, iterations=radius).astype(int)
        return mask

    def adjust_masks_size(self):
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                mask_1 = self.box[frame, self.num_times_channels + cam * self.num_channels, :, :]
                mask_2 = self.box[frame, self.num_times_channels + 1 + cam * self.num_channels, :, :]
                mask_1 = self.adjust_mask(mask_1)
                mask_2 = self.adjust_mask(mask_2)
                self.box[frame, self.num_times_channels + cam * self.num_channels, :, :] = mask_1
                self.box[frame, self.num_times_channels + 1 + cam * self.num_channels, :, :] = mask_2

    def run_predict_box(self, batch_size=8):  # todo: consider do wings and body points predictions togather
        t0 = time()
        if os.path.exists(self.out_path):
            print("Error: Output path already exists.")
            return

        self.preprocess_box()
        self.add_masks()

        # adding the using the right left right order
        # self.box = np.transpose(h5py.File("box_to_save_movie_17.h5", "r")["/box"][:], [0, 3, 1, 2])

        self.adjust_masks_size()
        if self.is_video:
            self.fix_masks()
        self.box_to_save = self.box
        self.adjust_dimensions()
        preprocessing_time = time() - t0

        preds_time = time()
        print("preprocess [%.1fs]" % preprocessing_time)
        if self.model_type == PER_WING:
            Ypks = []
            for cam in range(self.num_cams):
                print(f"predict wing 1 points cam {cam + 1}")
                if self.num_times_channels == 3:
                    left_channels = [0, 1, 2, 3]
                elif self.num_times_channels == 5:
                    left_channels = [0, 1, 2, 3, 4, 5]
                input_left = np.transpose(self.box[:, cam, :, :, left_channels], [1, 2, 3, 0])
                Ypk_left, _, _, _ = Predictions.predict_Ypk(input_left,
                                                            batch_size,
                                                            self.pose_estimation_model)
                print(f"predict wing 2 points cam {cam + 1}")

                if self.num_times_channels == 3:
                    right_channels = [0, 1, 2, 4]
                elif self.num_times_channels == 5:
                    right_channels = [0, 1, 2, 3, 4, 6]
                input_right = np.transpose(self.box[:, cam, :, :, right_channels], [1, 2, 3, 0])
                Ypk_right, _, _, _ = Predictions.predict_Ypk(input_right,
                                                             batch_size,
                                                             self.pose_estimation_model)
                Ypk_cam = np.concatenate((Ypk_left, Ypk_right), axis=2)
                Ypk_cam = np.expand_dims(Ypk_cam, axis=1)
                Ypks.append(Ypk_cam)
            Ypk = np.concatenate(Ypks, axis=1)

            # visualize_box_confmaps(self.box, confmaps, self.model_type)

        elif self.model_type == ALL_POINTS or self.model_type == BODY_POINTS or model_type == TWO_WINGS_TOGATHER:
            Ypks = []
            for cam in range(self.num_cams):
                input_per_cam = self.box[:, cam, :, :, :]
                Ypk_cam, _, _, _ = Predictions.predict_Ypk(input_per_cam,
                                                           batch_size,
                                                           self.pose_estimation_model)
                Ypk_cam = np.expand_dims(Ypk_cam, axis=1)
                Ypks.append(Ypk_cam)
            Ypk = np.concatenate(Ypks, axis=1)
        elif self.model_type == ALL_CAMS:
            Ypk_left, _, _, _ = Predictions.predict_Ypk(self.left_input,
                                                        batch_size,
                                                        self.pose_estimation_model)
            Ypk_left_1 = np.expand_dims(Ypk_left[:, :, 0:7], 1)
            Ypk_left_2 = np.expand_dims(Ypk_left[:, :, 7:14], 1)
            Ypk_left_3 = np.expand_dims(Ypk_left[:, :, 14:21], 1)
            Ypk_left_4 = np.expand_dims(Ypk_left[:, :, 21:28], 1)

            Ypk_left = np.concatenate((Ypk_left_1, Ypk_left_2, Ypk_left_3, Ypk_left_4), 1)

            Ypk_right, _, _, _ = Predictions.predict_Ypk(self.right_input,
                                                         batch_size,
                                                         self.pose_estimation_model)

            Ypk_right_1 = np.expand_dims(Ypk_right[:, :, 0:7], 1)
            Ypk_right_2 = np.expand_dims(Ypk_right[:, :, 7:14], 1)
            Ypk_right_3 = np.expand_dims(Ypk_right[:, :, 14:21], 1)
            Ypk_right_4 = np.expand_dims(Ypk_right[:, :, 21:28], 1)

            Ypk_right = np.concatenate((Ypk_right_1, Ypk_right_2, Ypk_right_3, Ypk_right_4), axis=1)
            Ypk = np.concatenate((Ypk_left, Ypk_right), axis=-1)
        elif self.model_type == ALL_CAMS_AND_3_GOOD_CAMS:
            per_wing_model_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per wing model\PER_WING_SMALL_WINGS_MODEL_Apr 29\best_model.h5"
            per_wing_model = Predictions.get_pose_estimation_model(per_wing_model_path)
            Ypk_left_small_wing_cam, _, _, _ = Predictions.predict_Ypk(self.left_input_small_wing_cam,
                                                        batch_size,
                                                        per_wing_model)
            Ypk_left_3cams, _, _, _ = Predictions.predict_Ypk(self.left_input_3_cams,
                                                        batch_size,
                                                        self.pose_estimation_model)

            Ypk_left_3cams_1 = np.expand_dims(Ypk_left_3cams[:, :, 0:7], 1)
            Ypk_left_3cams_2 = np.expand_dims(Ypk_left_3cams[:, :, 7:14], 1)
            Ypk_left_3cams_3 = np.expand_dims(Ypk_left_3cams[:, :, 14:21], 1)
            Ypk_left_3cams = np.concatenate((Ypk_left_3cams_1, Ypk_left_3cams_2, Ypk_left_3cams_3), axis=1)

            Ypk_left = np.zeros((Ypk_left_small_wing_cam.shape[0], self.num_cams, Ypk_left_small_wing_cam.shape[1],
                                  Ypk_left_small_wing_cam.shape[2]))
            for frame in range(self.num_frames):
                small_wing_ind = self.left_input_small_wing_cams_inds[frame]
                Ypk_left[frame, small_wing_ind, ...] = Ypk_left_small_wing_cam[frame, ...]
                cams_inds_3 = np.delete(np.arange(self.num_cams), np.where(np.arange(self.num_cams) == small_wing_ind))
                Ypk_left[frame, cams_inds_3, ...] = Ypk_left_3cams[frame, ...]
            ## right input
            Ypk_right_small_wing_cam, _, _, _ = Predictions.predict_Ypk(self.right_input_small_wing_cam,
                                                                       batch_size,
                                                                       per_wing_model)
            Ypk_right_3cams, _, _, _ = Predictions.predict_Ypk(self.right_input_3_cams,
                                                              batch_size,
                                                              self.pose_estimation_model)

            Ypk_right_3cams_1 = np.expand_dims(Ypk_right_3cams[:, :, 0:7], 1)
            Ypk_right_3cams_2 = np.expand_dims(Ypk_right_3cams[:, :, 7:14], 1)
            Ypk_right_3cams_3 = np.expand_dims(Ypk_right_3cams[:, :, 14:21], 1)
            Ypk_right_3cams = np.concatenate((Ypk_right_3cams_1, Ypk_right_3cams_2, Ypk_right_3cams_3), axis=1)

            Ypk_right = np.zeros((Ypk_right_small_wing_cam.shape[0], self.num_cams, Ypk_right_small_wing_cam.shape[1],
                                 Ypk_right_small_wing_cam.shape[2]))
            for frame in range(self.num_frames):
                small_wing_ind = self.right_input_small_wing_cams_inds[frame]
                Ypk_right[frame, small_wing_ind, ...] = Ypk_right_small_wing_cam[frame, ...]
                cams_inds_3 = np.delete(np.arange(self.num_cams), np.where(np.arange(self.num_cams) == small_wing_ind))
                Ypk_right[frame, cams_inds_3, ...] = Ypk_right_3cams[frame, ...]

            Ypk = np.concatenate((Ypk_left, Ypk_right), axis=-1)

        self.Ypk = Ypk
        self.prediction_runtime = time() - preds_time
        self.total_runtime = time() - t0
        print("Predicted [%.1fs]" % self.prediction_runtime)
        print("Prediction performance: %.3f FPS" % (self.num_frames * self.num_cams / self.prediction_runtime))
        self.save_predictions_to_h5()


if __name__ == "__main__":
    # box_path_no_masks = r"movie_1_1701_2200_500_frames_3tc_7tj_no_masks.h5"
    # wings

    model_type = PER_WING
    # model_type = TWO_WINGS_TOGATHER
    # model_type = ALL_CAMS
    # model_type = BODY_POINTS
    # model_type = ALL_CAMS_AND_3_GOOD_CAMS

    wings_detection_model_path = "wings_detection_yolov8_weights_13_3.pt"
    # box_path_no_masks = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 14\dataset_movie_14_frames_1301_2300_ds_3tc_7tj.h5"

    box_path_no_masks = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 60 roni\movie_60_1501_2500_ds_3tc_7tj.h5"
    pose_estimation_model_path_wings = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\train on 3 good cameras\TRAIN_ON_3_GOOD_CAMERAS_MODEL_Mar 30\best_model.h5"

    out_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 60 roni\PER_WING_3_good_cams_predictions.h5"
    predictions = Predictions(
        box_path_no_masks,
        model_type,
        out_path,
        pose_estimation_model_path_wings,
        wings_detection_model_path,
        is_video=True
    )
    predictions.run_predict_box()

    model_type = BODY_POINTS
    pose_estimation_model_path_body = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\body parts model\body_parts_model_dilation_2\best_model.h5"
    out_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 60 roni\BODY_PNTS_predictions.h5"
    predictions = Predictions(
        box_path_no_masks,
        model_type,
        out_path,
        pose_estimation_model_path_body,
        wings_detection_model_path
    )
    predictions.run_predict_box()

# path = r"C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\main datasets\random frames\box_random_frames_dataset_agreed_masks.h5"
# self.box = h5py.File(path, "r")["/box"][:]
# self.box = self.box.T
# self.box_to_save = self.box
# self.left_input = self.box[:, :, :, :, [0, 1, 2, 3]]
# self.left_input = np.concatenate((self.left_input[:, 0, :, :, :],
#                                   self.left_input[:, 1, :, :, :],
#                                   self.left_input[:, 2, :, :, :],
#                                   self.left_input[:, 3, :, :, :]), axis=-1)
#
# self.right_input = self.box[:, :, :, :, [0, 1, 2, 4]]
# self.right_input = np.concatenate((self.right_input[:, 0, :, :, :],
#                                    self.right_input[:, 1, :, :, :],
#                                    self.right_input[:, 2, :, :, :],
#                                    self.right_input[:, 3, :, :, :]), axis=-1)
