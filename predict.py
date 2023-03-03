import h5py
import numpy as np
import os
from time import time
from tensorflow import keras
import tensorflow.keras.models
from tensorflow.keras.layers import Lambda
import tensorflow as tf
from preprocessing_utils import adjust_masks_size
# from training import preprocess
from training_with_masks import visualize_box_confmaps
# from leap.utils import find_weights, find_best_weights, preprocess
# from leap.layers import Maxima2D
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import matplotlib

# imports of the wings detection
import cv2
from time import time
from ultralytics import YOLO

PER_WING = "PER WING"
ALL_POINTS = "ALL POINTS"
BODY_POINTS = "BODY_POINTS"
PER_WING_SPLIT_TO_3 = "PER WING SPLIT TO 3"
PER_WING_PER_POINT = "PER_WING_PER_POINT"
ENSEMBLE = "ENSEMBLE"
NO_MASKS = "NO_MASKS"
HEAD_TAIL = "NO_MASKS"
NUM_CAMS = 4
NUM_CHANNELS_PER_IMAGE = 5


class Predictions:

    def __init__(self,
                 box_path: str,
                 model_type: str,
                 out_path: str,
                 pose_estimation_model_path: str,
                 wings_detection_model_path: str,
                 is_video: bool = True):

        self.box_path = box_path
        self.model_type = model_type
        self.pose_estimation_model_path = pose_estimation_model_path
        self.out_path = out_path
        self.wings_detection_model_path = wings_detection_model_path
        self.is_video = is_video

        self.box = self.get_box()
        self.cropzone = self.get_cropzone()
        self.pose_estimation_model = self.get_pose_estimation_model()
        self.wings_detection_model = self.get_wings_detection_model()

        self.im_size = self.box.shape[-1]
        self.num_frames = self.box.shape[0]
        self.num_cams = NUM_CAMS
        self.num_times_channels = self.box.shape[1] // self.num_cams
        self.num_channels = NUM_CHANNELS_PER_IMAGE

        self.Ypk = None
        self.prediction_runtime = None
        self.total_runtime = None

    def get_box(self):
        return h5py.File(self.box_path, "r")["/box"]

    def get_cropzone(self):
        return h5py.File(self.box_path, "r")["/cropzone"]

    def get_pose_estimation_model(self):
        """ load a pretrained LEAP pose estimation model model"""
        model = keras.models.load_model(self.pose_estimation_model_path)
        model_peaks = Predictions.convert_to_peak_outputs(model, include_confmaps=False)
        print("weights_path:", self.pose_estimation_model_path)
        print("Loaded model: %d layers, %d params" % (len(model.layers), model.count_params()))
        return model_peaks

    def get_wings_detection_model(self):
        """ load a pretrained YOLOv8 segmentation model"""
        model = YOLO(self.wings_detection_model_path)
        model.fuse()
        return model

    def preprocess_box(self):
        if self.box.dtype == "uint8" or np.max(self.box) > 1:
            self.box = self.box.astype("float32") / 255

    def add_masks(self):   # todo do it in one swoop : insert a matrix in size (num_frames, 192, 192, 3), can be much faster
        """ Add masks to the dataset using yolov8 segmentation model """
        new_box = np.zeros((self.num_frames, self.num_cams * self.num_channels, self.im_size, self.im_size))
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                print(f"frame number {frame}, cam number {cam}")
                img_3_ch = self.box[frame, np.array([0, 1, 2]) + self.num_times_channels * cam, :, :]
                img_3_ch = np.transpose(img_3_ch, [1, 2, 0])
                net_input = np.round(255 * img_3_ch)
                results = self.wings_detection_model(net_input)[0]
                num_masks = results.masks.shape[0]
                for mask_num in range(min(num_masks, 2)):
                    mask = results.masks[mask_num, :, :].masks.numpy()
                    # mask = cv2.resize(mask, (im_size, im_size), interpolation=cv2.INTER_AREA)
                    new_box[frame, self.num_times_channels + mask_num + self.num_channels * cam, :, :] = mask
                new_box[frame, np.array([0, 1, 2]) + self.num_channels * cam, :, :] = np.transpose(img_3_ch, [2, 0, 1])

                # show the image
                # imtoshow = new_box[frame, np.array([0,1,2]) + self.num_channels * cam, :, :]
                # imtoshow = np.transpose(imtoshow, [1, 2, 0])
                # mask_1 = new_box[frame, 3 + self.num_channels * cam, :, :]
                # mask_2 = new_box[frame, 4 + self.num_channels * cam, :, :]
                # imtoshow[:, :, 1] += mask_1
                # imtoshow[:, :, 1] += mask_2
                # matplotlib.use('TkAgg')
                # plt.imshow(imtoshow)
                # plt.show()
        self.box = new_box

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
                    mask = self.box[frame, 3 + mask_num + self.num_channels * cam, :, :]
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
                        new_mask = prev_mask + next_mask
                        new_mask[new_mask >= 1] = 1
                        # replace empty mask with new mask
                        self.box[frame, 3 + mask_num + self.num_channels * cam, :, :] = new_mask

        # do dilation to all masks
        self.box = adjust_masks_size(self.box, "PREDICT")

    def adjust_dimensions(self):  # todo: do it in a smarter less ugly way
        """ return a box with dimensions (num_frames, imsize, imsize, num_channels)"""
        self.box = np.transpose(self.box, (0, 3, 2, 1))
        self.box = np.transpose(self.box, (1, 2, 3, 0))
        if self.box.shape[2] == 20:  # if number of channels is 5

            x1 = self.box[:, :, 0:5, :]
            x2 = self.box[:, :, 5:10, :]
            x3 = self.box[:, :, 10:15, :]
            x4 = self.box[:, :, 15:20, :]
            self.box = np.concatenate((x1, x2, x3, x4), axis=3)
        self.box = np.transpose(self.box, (3, 0, 1, 2))

    def save_predictions_to_h5(self):
        with h5py.File(self.out_path, "w") as f:
            f.attrs["num_samples"] = self.box.shape[0]
            f.attrs["img_size"] = self.im_size
            f.attrs["box_path"] = self.box_path
            f.attrs["box_dset"] = "/box"
            f.attrs["pose_estimation_model_path"] = self.pose_estimation_model_path
            f.attrs["wings_detection_model_path"] = self.wings_detection_model_path
            # f.attrs["model_name"] = model_name
            positions = self.Ypk[:, :2, :]
            confidence_val = self.Ypk[:, 2, :]
            # positions = np.transpose(positions, (0,2,1))
            ds_pos = f.create_dataset("positions_pred", data=positions.astype("int32"), compression="gzip",
                                      compression_opts=1)
            ds_pos.attrs["description"] = "coordinate of peak at each sample"
            ds_pos.attrs["dims"] = "(sample, [x, y], joint) === (sample, [column, row], joint)"

            ds_conf = f.create_dataset("conf_pred", data=confidence_val.squeeze(), compression="gzip",
                                       compression_opts=1)
            ds_conf.attrs["description"] = "confidence map value in [0, 1.0] at peak"
            ds_conf.attrs["dims"] = "(sample, joint)"

            ds_conf = f.create_dataset("box", data=self.box, compression="gzip", compression_opts=1)
            ds_conf.attrs["description"] = "The predicted box"
            ds_conf.attrs["dims"] = f"{self.box.shape}"

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

    def run_predict_box(self, batch_size=8):  # todo: consider do wings and body points predictions togather
        t0 = time()
        if os.path.exists(self.out_path):
            print("Error: Output path already exists.")
            return
        self.preprocess_box()
        self.add_masks()
        if self.is_video:
            self.fix_masks()
        self.adjust_dimensions()
        preprocessing_time = time() - t0
        preds_time = time()
        print("preprocess [%.1fs]" % preprocessing_time)
        if self.model_type == PER_WING:
            input_left = self.box[:, :, :, [0, 1, 2, 3]]
            input_right = self.box[:, :, :, [0, 1, 2, 4]]
            print("predict wing 1 points")
            Ypk_left, _, _, _ = Predictions.predict_Ypk(input_left,
                                                        batch_size,
                                                        self.pose_estimation_model)
            print("predict wing 2 points")
            Ypk_right, _, _, _ = Predictions.predict_Ypk(input_right,
                                                         batch_size,
                                                         self.pose_estimation_model)


            Ypk = np.concatenate((Ypk_left, Ypk_right), axis=2)
            # visualize_box_confmaps(self.box, confmaps, self.model_type)

        elif self.model_type == ALL_POINTS or self.model_type == BODY_POINTS:
            Ypk, confmaps, confmaps_max, confmaps_min = Predictions.predict_Ypk(self.box,
                                                                                batch_size,
                                                                                self.pose_estimation_model)
        self.Ypk = Ypk
        self.prediction_runtime = time() - preds_time
        self.total_runtime = time() - t0
        print("Predicted [%.1fs]" % self.prediction_runtime)
        print("Prediction performance: %.3f FPS" % (self.num_frames * self.num_cams / self.prediction_runtime))
        self.save_predictions_to_h5()


if __name__ == "__main__":
    # model_type = PER_WING
    model_type = BODY_POINTS
    # box_path_no_masks = r"movie_1_1701_2200_500_frames_3tc_7tj _no_masks.h5"
    box_path_no_masks = r"movie_dataset_1_3_600_frames_no_masks.h5"
    pose_estimation_model_path = 'train_3_cams_weights.h5'
    wings_detection_model_path = "wings_detection_yolov8_weights.pt"

    out_path = "predictions_movie_1_3_600_frames_yolov8_body.h5"

    predictions = Predictions(box_path_no_masks,
                              model_type,
                              out_path,
                              pose_estimation_model_path,
                              wings_detection_model_path)
    predictions.run_predict_box()
