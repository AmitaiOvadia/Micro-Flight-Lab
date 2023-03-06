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
    rows = tf.math.floordiv(tf.cast(idx,tf.int32), in_shape[1])
    cols = tf.math.floormod(tf.cast(idx,tf.int32), in_shape[1])

    # Dumb way to get actual values without indexing
    vals = tf.math.reduce_max(flattened, axis=1)

    # Return N x 3 x C tensor
    pred = tf.stack([
        tf.cast(cols, tf.float32),
        tf.cast(rows, tf.float32),
        vals
    ], axis=1)
    return pred


def convert_to_peak_outputs(model, include_confmaps=False):
    """ Creates a new Keras model with a wrapper to yield channel peaks from rank-4 tensors. """
    if type(model.output) == list:
        confmaps = model.output[-1]
    else:
        confmaps = model.output

    if include_confmaps:
        return keras.Model(model.input, [Lambda(tf_find_peaks)(confmaps), confmaps])
    else:
        return keras.Model(model.input, Lambda(tf_find_peaks)(confmaps))
        # return keras.Model(model.input, Maxima2D()(confmaps))


def get_left_right_points(X, batch_size, model_peaks, save_confmaps):
    input_left = X[:, :, :, [0, 1, 2, 3]]
    input_right = X[:, :, :, [0, 1, 2, 4]]
    Ypk_left, confmaps_left, confmaps_max_l, confmaps_min_l = predict_Ypk(input_left, batch_size, model_peaks,
                                                                          save_confmaps)
    Ypk_right, confmaps_right, confmaps_max_r, confmaps_min_r = predict_Ypk(input_right, batch_size, model_peaks,
                                                                            save_confmaps)
    Ypk = np.concatenate((Ypk_left, Ypk_right), axis=2)
    return Ypk


def save_predictions_to_h5(box, Ypk, cropzone, box_dset, box_path, confmaps, confmaps_max, confmaps_min, model_path, num_samples,
                           out_path, prediction_runtime, save_confmaps, t0_all, weights_path):
    with h5py.File(out_path, "w") as f:
        f.attrs["num_samples"] = box.shape[0]
        f.attrs["img_size"] = box.shape[1:]
        f.attrs["box_path"] = box_path
        f.attrs["box_dset"] = box_dset
        f.attrs["model_path"] = model_path
        f.attrs["weights_path"] = weights_path
        # f.attrs["model_name"] = model_name
        positions = Ypk[:, :2, :]
        confidence_val = Ypk[:, 2, :]
        # positions = np.transpose(positions, (0,2,1))
        ds_pos = f.create_dataset("positions_pred", data=positions.astype("int32"), compression="gzip",
                                  compression_opts=1)
        ds_pos.attrs["description"] = "coordinate of peak at each sample"
        ds_pos.attrs["dims"] = "(sample, [x, y], joint) === (sample, [column, row], joint)"

        ds_conf = f.create_dataset("conf_pred", data=confidence_val.squeeze(), compression="gzip", compression_opts=1)
        ds_conf.attrs["description"] = "confidence map value in [0, 1.0] at peak"
        ds_conf.attrs["dims"] = "(sample, joint)"

        ds_conf = f.create_dataset("box", data=box, compression="gzip", compression_opts=1)
        ds_conf.attrs["description"] = "The predicted box"
        ds_conf.attrs["dims"] = f"{box.shape}"

        ds_conf = f.create_dataset("cropzone", data=cropzone, compression="gzip", compression_opts=1)
        ds_conf.attrs["description"] = "cropzone of every image for 2D to 3D projection"
        ds_conf.attrs["dims"] = f"{cropzone.shape}"

        if save_confmaps:
            ds_confmaps = f.create_dataset("confmaps", data=confmaps, compression="gzip", compression_opts=1)
            ds_confmaps.attrs["description"] = "confidence maps"
            ds_confmaps.attrs["dims"] = "(sample, width, height, channel)"
            ds_confmaps.attrs["range_min"] = confmaps_min
            ds_confmaps.attrs["range_max"] = confmaps_max

        total_runtime = time() - t0_all
        f.attrs["total_runtime_secs"] = total_runtime
        f.attrs["prediction_runtime_secs"] = prediction_runtime
    return total_runtime


def predict_Ypk(X, batch_size, model_peaks, save_confmaps):
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


def get_ensemble_results(sum_of_confmaps, num_samples):
    num_points = sum_of_confmaps.shape[-1]
    Ypk = np.zeros((num_samples, 3, num_points))
    for image in range(num_samples):
        for point in range(num_points):
            confmap = sum_of_confmaps[image, :, :, point]
            peak = np.unravel_index(np.argmax(confmap, axis=None), confmap.shape)
            Ypk[image, [0, 1], point] = peak  # get peak index
            Ypk[image, 2, point] = confmap[peak[0], peak[1]]  # get peak value
    return Ypk

def preprocess(X):
    """ Normalizes input data. """

    # Add singleton dim for single images
    if X.ndim == 3:
        X = X[None, ...]
        # Normalize
    if X.dtype == "uint8" or np.max(X) > 1:
        X = X.astype("float32") / 255
    return X


def adjust_dimensions(X, permute=(0, 3, 2, 1)):
    #  preprocess X
    #  if there are 12-20 channels and we want to reduce to 3 or 5

    # Adjust dimensions

    X = np.transpose(X, permute)
    X = np.transpose(X, (1, 2, 3, 0))

    if X.shape[2] == 12:  # if number of channels is 3
        x1 = X[:, :, 0:3, :]
        x2 = X[:, :, 3:6, :]
        x3 = X[:, :, 6:9, :]
        x4 = X[:, :, 9:12, :]
        X = np.concatenate((x1, x2, x3, x4), axis=3)

    if X.shape[2] == 15:
        x1 = X[:, :, 0:5, :]
        x2 = X[:, :, 5:10, :]
        x3 = X[:, :, 10:15, :]
        X = np.concatenate((x1, x2, x3), axis=3)

    if X.shape[2] == 20:  # if number of channels is 5

        x1 = X[:, :, 0:5, :]
        x2 = X[:, :, 5:10, :]
        x3 = X[:, :, 10:15, :]
        x4 = X[:, :, 15:20, :]
        X = np.concatenate((x1, x2, x3, x4), axis=3)
    X = np.transpose(X, (3, 0, 1, 2))
    return X


def fix_masks(X):
    """
    goes throw each frame, if there is no mask for a specific wing, unite masks of the closest times before and after
    this frame.
    :param X: a box of size (num_frames, 20, 192, 192)
    :return: same box
    """
    search_range = 5
    num_channels = 5
    num_frames = int(X.shape[0])
    num_cams = int(X.shape[1]/num_channels)
    problematic_masks = []
    for frame in range(num_frames):
        for cam in range(num_cams):
            for mask_num in range(2):
                mask = X[frame, 3 + mask_num + num_channels * cam, :, :]
                if np.all(mask == 0):  # check if all 0:
                    problematic_masks.append((frame, cam, mask_num))
                    # find previous matching mask
                    prev_mask = np.zeros(mask.shape)
                    next_mask = np.zeros(mask.shape)
                    for prev_frame in range(frame - 1, max(0, frame - search_range - 1), -1):
                        prev_mask_i = X[prev_frame, 3 + mask_num + num_channels * cam, :, :]
                        if not np.all(prev_mask_i == 0):  # there is a good mask
                            prev_mask = prev_mask_i
                            break
                    # find next matching mask
                    for next_frame in range(frame + 1, min(num_frames, frame + search_range)):
                        next_mask_i = X[next_frame, 3 + mask_num + num_channels * cam, :, :]
                        if not np.all(next_mask_i == 0):  # there is a good mask
                            next_mask = next_mask_i
                            break
                    # combine the 2 masks
                    new_mask = prev_mask + next_mask
                    new_mask[new_mask >= 1] = 1
                    # replace empty mask with new mask
                    X[frame, 3 + mask_num + num_channels * cam, :, :] = new_mask

    # do dilation to all masks
    X = adjust_masks_size(X, "PREDICT")
    return X


def do_dilation_do_all_masks(X, radius=5):
    num_channels = 5
    num_frames = int(X.shape[0])
    for frame in range(num_frames):
        for cam in range(4):
            for mask_num in range(2):
                mask = X[frame, 3 + mask_num + num_channels * cam, :, :]
                dilated_mask = binary_dilation(mask, iterations=radius).astype(int)
                X[frame, 3 + mask_num + num_channels * cam, :, :] = dilated_mask
    return X


def add_masks(box):
    """ Add masks detected using a YOLOV8 model """
    wings_model = "wings_detection_yolov8_weights_4_3.pt"
    model = YOLO(wings_model)  # load a pretrained YOLOv8n model
    model.fuse()
    im_size = box.shape[-1]
    new_num_channels = 5
    num_cams = 4
    num_times_channels = 3
    num_frames = box.shape[0]
    cur_channels = box.shape[1]
    new_box = np.zeros((box.shape[0], num_cams * new_num_channels, im_size, im_size))
    for frame in range(num_frames):
        for cam in range(num_cams):
            print(f"frame number {frame}, cam number {cam}")
            img_3_ch = box[frame, np.array([0,1,2]) + num_times_channels * cam, :, :]
            img_3_ch = np.transpose(img_3_ch, [1, 2, 0])
            net_input = np.round(255*img_3_ch)
            results = model(net_input)[0]
            num_masks = results.masks.shape[0]
            for mask_num in range(min(num_masks, 2)):
                mask = results.masks[mask_num, :, :].masks.numpy()
                # mask = cv2.resize(mask, (im_size, im_size), interpolation=cv2.INTER_AREA)
                new_box[frame, num_times_channels + mask_num + new_num_channels * cam, :, :] = mask
            new_box[frame, np.array([0, 1, 2]) + new_num_channels * cam, :, :] = np.transpose(img_3_ch, [2, 0, 1])

            # show the image
            # imtoshow = new_box[frame, np.array([0,1,2]) + new_num_channels * cam, :, :]
            # imtoshow = np.transpose(imtoshow, [1, 2, 0])
            # mask_1 = new_box[frame, 3 + new_num_channels * cam, :, :]
            # mask_2 = new_box[frame, 4 + new_num_channels * cam, :, :]
            # imtoshow[:, :, 1] += mask_1
            # imtoshow[:, :, 1] += mask_2
            # matplotlib.use('TkAgg')
            # plt.imshow(imtoshow)
            # plt.show()
            # a=0
    return new_box


def predict_box(box_path,
                model_path_wings,
                out_path, *,
                add_head_tail=False,
                model_type=False,
                box_dset="/box",
                cropzone="/cropzone",
                test_dset="/testing/box",
                segmentation_scores = "/segmentation_scores",
                epoch=None,
                video=True,
                verbose=True,
                overwrite=False,
                save_confmaps=False,
                batch_size=32):
    """
    Predict and save peak coordinates for a box.

    :param box_path: path to HDF5 file with box dataset
    :param model_path_wings: path to Keras weights file or run folder with weights subfolder
    :param out_path: path to HDF5 file to save results to
    :param box_dset: name of HDF5 dataset containing box images
    :param epoch: epoch to use if run folder provided instead of Keras weights file
    :param verbose: if True, prints some info and statistics during procesing
    :param overwrite: if True and out_path exists, file will be overwritten
    :param save_confmaps: if True, saves the full confidence maps as additional datasets in the output file (very slow)
    :param batch_size: number of samples to evaluate at once per batch (see keras.Model API)
    """

    if verbose:
        print("model_path:", model_path_wings)

    # Find model weights
    model_name = None
    weights_path = model_path_wings

    box = h5py.File(box_path,"r")[box_dset]
    cropzone = h5py.File(box_path,"r")[cropzone]

    num_samples = box.shape[0]
    if verbose:
        print("Input:", box_path)
        print("box.shape:", box.shape)

    if verbose:
        print("Output:", out_path)

    t0_all = time()
    if os.path.exists(out_path):
        if overwrite:
            os.remove(out_path)
            print("Deleted existing output.")
        else:
            print("Error: Output path already exists.")
            return

    # Load and prepare wings model
    if model_type == PER_WING_SPLIT_TO_3 or model_type == PER_WING_PER_POINT or model_type == ENSEMBLE:
        models_list = []
        if ENSEMBLE:
            save_confmaps = True
        for path in model_path_wings:
            model = keras.models.load_model(path)
            model_peaks = convert_to_peak_outputs(model, include_confmaps=save_confmaps)
            if verbose:
                print("weights_path:", weights_path)
                print("Loaded model: %d layers, %d params" % (len(model.layers), model.count_params()))
            models_list.append(model_peaks)

    else:
        model = keras.models.load_model(weights_path)
        model_peaks = convert_to_peak_outputs(model, include_confmaps=save_confmaps)
        if verbose:
            print("weights_path:", weights_path)
            print("Loaded model: %d layers, %d params" % (len(model.layers), model.count_params()))

    # Load data and preprocess (normalize)
    t0 = time()
    box = preprocess(box[:])

    # box = box[:5, :,:,:]
    # add masks
    box = add_masks(box)

    # fix the masks
    if video:
        box = fix_masks(box)
    else:
        box = do_dilation_do_all_masks(box)
    X = adjust_dimensions(box)

    # visualize
    # visualize_box_confmaps(X, None, 'ALL_POINTS_MODEL')

    if verbose:
        print("Loaded [%.1fs]" % (time() - t0))

    t0 = time()
    confmaps, confmaps_max, confmaps_min, = None, None, None
    if model_type == NO_MASKS or model_type == HEAD_TAIL:
        input = X[:, :, :, [0, 1, 2]]
        Ypk, confmaps, confmaps_max, confmaps_min = predict_Ypk(input, batch_size, model_peaks, save_confmaps)

    elif model_type == ALL_POINTS or model_type == BODY_POINTS:
        Ypk, confmaps, confmaps_max, confmaps_min = predict_Ypk(X, batch_size, model_peaks, save_confmaps)

    elif model_type == PER_WING:
        input_left = X[:, :, :, [0, 1, 2, 3]]
        input_right = X[:, :, :, [0, 1, 2, 4]]
        print("predict wing 1 points")
        Ypk_left, confmaps_left, confmaps_max_l, confmaps_min_l = predict_Ypk(input_left, batch_size, model_peaks,
                                                                              save_confmaps)
        print("predict wing 2 points")
        Ypk_right, confmaps_right, confmaps_max_r, confmaps_min_r = predict_Ypk(input_right, batch_size, model_peaks,
                                                                                save_confmaps)
        Ypk = np.concatenate((Ypk_left, Ypk_right), axis=2)
        if save_confmaps:
            confmaps = np.transpose(np.concatenate((confmaps_left, confmaps_right), axis=1), (0, 2, 3, 1))
            confmaps_min, confmaps_max = np.min([confmaps_min_l, confmaps_min_r]), np.max([confmaps_max_l, confmaps_max_r])
        # visualize_box_confmaps(X, confmaps, model_type)
        # X = np.concatenate((input_left, input_right), axis=0)

    elif model_type == ENSEMBLE:
        num_samples = X.shape[0]
        sum_of_confmaps = np.zeros((num_samples, 192, 192, 14)).astype(np.float32)
        for model_peak in models_list:
            input_left = X[:, :, :, [0, 1, 2, 3]]
            input_right = X[:, :, :, [0, 1, 2, 4]]
            Ypk_left, confmaps_left, confmaps_max_l, confmaps_min_l = predict_Ypk(input_left, batch_size,
                                                                                  model_peak, save_confmaps=True)
            Ypk_right, confmaps_right, confmaps_max_r, confmaps_min_r = predict_Ypk(input_right, batch_size,
                                                                                    model_peak, save_confmaps=True)
            Ypk_i = np.concatenate((Ypk_left, Ypk_right), axis=2)
            confmaps = np.transpose(np.concatenate((confmaps_left, confmaps_right), axis=1), (0, 2, 3, 1))
            # summing all confidence maps of same points
            sum_of_confmaps = np.add(sum_of_confmaps, confmaps)
        confmaps = sum_of_confmaps / len(models_list)
        Ypk = get_ensemble_results(sum_of_confmaps, num_samples)
        confmaps_max, confmaps_min = np.max(confmaps), np.min(confmaps)

    elif model_type == PER_WING_SPLIT_TO_3 or model_type == PER_WING_PER_POINT:
        Ypk_left_list, Ypk_right_list = [], []
        confmaps_left_list, confmaps_right_list = [], []
        for model_peak in models_list:
            input_left = X[:, :, :, [0, 1, 2, 3]]
            input_right = X[:, :, :, [0, 1, 2, 4]]
            Ypk_left, confmaps_left, confmaps_max_l, confmaps_min_l = predict_Ypk(input_left, batch_size,
                                                                                  model_peak, save_confmaps)
            Ypk_right, confmaps_right, confmaps_max_r, confmaps_min_r = predict_Ypk(input_right, batch_size,
                                                                                    model_peak, save_confmaps)
            Ypk_left_list.append(Ypk_left)
            Ypk_right_list.append(Ypk_right)
            # confmaps_left_list.append(confmaps_left)
            # confmaps_right_list.append(confmaps_right)
        Ypk_left_list, Ypk_right_list = np.concatenate(Ypk_left_list, axis=2), np.concatenate(Ypk_right_list, axis=2)
        # confmaps_left_list, confmaps_right_list = np.concatenate(confmaps_left_list, axis=2), np.concatenate(confmaps_right_list, axis=2)
        Ypk = np.concatenate((Ypk_left_list, Ypk_right_list), axis=2)
        # confmaps = np.concatenate((confmaps_left_list, confmaps_right_list), axis=2)


    prediction_runtime = time() - t0
    if verbose:
        print("Predicted [%.1fs]" % prediction_runtime)
        print("Prediction performance: %.3f FPS" % (num_samples / prediction_runtime))

    # Save
    t0 = time()
    save_confmaps = False
    save_predictions_to_h5(box, Ypk, cropzone, box_dset, box_path, confmaps, confmaps_max, confmaps_min, model_path_wings,
                           num_samples, out_path, prediction_runtime, save_confmaps, t0_all,
                           weights_path)
    total_runtime = time() - t0_all
    if verbose:
        print("Saved [%.1fs]" % (time() - t0))

        print("Total runtime: %.1f mins" % (total_runtime / 60))
        print("Total performance: %.3f FPS" % (num_samples / total_runtime))





if __name__ == "__main__":

    # model_type = PER_WING
    # model_type = ENSEMBLE
    # model_type = NO_MASKS
    # model_type = HEAD_TAIL
    # model_type = ALL_POINTS
    # model_type = BODY_POINTS
    # box_path_no_masks = r"movie_dataset_1_3_600_frames_no_masks.h5"
    # model_type = BODY_POINTS
    # model_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\body parts model\body_parts_model_dilation_2\best_model.h5"
    # out_path = f"predictions_{box_path_no_masks}_{model_type}_xx.h5"
    # predict_box(box_path_no_masks, model_path, out_path, model_type=model_type)

    box_path_no_masks = r"movie_1_1701_2200_500_frames_3tc_7tj_no_masks.h5"
    model_type = PER_WING
    model_path = r"train_3_cams_weights.h5"
    out_path = f"predictions_{box_path_no_masks}_{model_type}_xx.h5"
    predict_box(box_path_no_masks, model_path, out_path, model_type=model_type)



    # model_type = BODY_POINTS
    # model_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\body parts model\body_parts_model_dilation_2\best_model.h5"
    # out_path = fr"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\magnet movie\predict_over_movie_body.h5"
    # predict_box(magnet_movie_path, model_path, model_path_head_tail, out_path, model_type=model_type)