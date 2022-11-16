import h5py
import numpy as np
import os
from time import time
import keras
import keras.models
from keras.layers import Lambda
import tensorflow as tf
from training import preprocess
from training_with_masks import visualize_box_confmaps
# from leap.utils import find_weights, find_best_weights, preprocess
# from leap.layers import Maxima2D

PER_WING = "PER WING"
ALL_POINTS = "ALL POINTS"
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


def save_predictions_to_h5(X, Ypk, box_dset, box_path, confmaps, confmaps_max, confmaps_min, model_path, num_samples,
                           out_path, prediction_runtime, save_confmaps, t0_all, weights_path):
    with h5py.File(out_path, "w") as f:
        f.attrs["num_samples"] = X.shape[0]
        f.attrs["img_size"] = X.shape[1:]
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
        # confmaps = (confmaps - confmaps_min) / (confmaps_max - confmaps_min)
        # confmaps = (confmaps * 255).astype('uint8')

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


def predict_box(test_box, box_path, model_path, out_path, *,
                model_type=False,
                box_dset="/box",
                test_dset="/testing/box",
                epoch=None,
                verbose=True,
                overwrite=False,
                save_confmaps=False,
                batch_size=32):
    """
    Predict and save peak coordinates for a box.

    :param box_path: path to HDF5 file with box dataset
    :param model_path: path to Keras weights file or run folder with weights subfolder
    :param out_path: path to HDF5 file to save results to
    :param box_dset: name of HDF5 dataset containing box images
    :param epoch: epoch to use if run folder provided instead of Keras weights file
    :param verbose: if True, prints some info and statistics during procesing
    :param overwrite: if True and out_path exists, file will be overwritten
    :param save_confmaps: if True, saves the full confidence maps as additional datasets in the output file (very slow)
    :param batch_size: number of samples to evaluate at once per batch (see keras.Model API)
    """

    if verbose:
        print("model_path:", model_path)

    # Find model weights
    model_name = None
    weights_path = model_path

    # Input data
    if test_box:
        box = h5py.File(box_path,"r")[test_dset]
        box = np.reshape(box, (box.shape[0], box.shape[1] * box.shape[2], box.shape[3], box.shape[4])) # todo
    else:
        box = h5py.File(box_path,"r")[box_dset]
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

    # Load and prepare model

    if model_type == PER_WING_SPLIT_TO_3 or model_type == PER_WING_PER_POINT or model_type == ENSEMBLE:
        models_list = []
        if ENSEMBLE:
            save_confmaps = True
        for path in model_path:
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
    X = preprocess(box[:])

    #  preprocess X
    #  if there are 12-20 channels and we want to reduce to 3 or 5
    X = np.transpose(X, (1, 2, 3, 0))

    if X.shape[2] == 12:  # if number of channels is 3
        x1 = X[:, :, 0:3, :]
        x2 = X[:, :, 3:6, :]
        x3 = X[:, :, 6:9, :]
        x4 = X[:, :, 9:12,:]
        X = np.concatenate((x1, x2, x3, x4), axis=3)

    if X.shape[2] == 20:  # if number of channels is 5
        x1 = X[:, :, 0:5, :]
        x2 = X[:, :, 5:10, :]
        x3 = X[:, :, 10:15, :]
        x4 = X[:, :, 15:20, :]
        X = np.concatenate((x1, x2, x3, x4), axis=3)
    X = np.transpose(X, (3, 0, 1, 2))
    if verbose:
        print("Loaded [%.1fs]" % (time() - t0))

    # Evaluate
    t0 = time()
    confmaps, confmaps_max, confmaps_min, = None, None, None
    if model_type == NO_MASKS or model_type == HEAD_TAIL:
        input = X[:, :, :, [0, 1, 2]]
        Ypk, confmaps, confmaps_max, confmaps_min = predict_Ypk(input, batch_size, model_peaks, save_confmaps)

    elif model_type == ALL_POINTS:
        Ypk, confmaps, confmaps_max, confmaps_min = predict_Ypk(X, batch_size, model_peaks, save_confmaps)

    elif model_type == PER_WING:  ## todo extract confidence maps
        input_left = X[:, :, :, [0, 1, 2, 3]]
        input_right = X[:, :, :, [0, 1, 2, 4]]
        Ypk_left, confmaps_left, confmaps_max_l, confmaps_min_l = predict_Ypk(input_left, batch_size, model_peaks,
                                                                              save_confmaps)
        Ypk_right, confmaps_right, confmaps_max_r, confmaps_min_r = predict_Ypk(input_right, batch_size, model_peaks,
                                                                                save_confmaps)
        Ypk = np.concatenate((Ypk_left, Ypk_right), axis=2)
        if save_confmaps:
            confmaps = np.transpose(np.concatenate((confmaps_left, confmaps_right), axis=1), (0, 2, 3, 1))
            confmaps_min, confmaps_max = np.min([confmaps_min_l, confmaps_min_r]), np.max([confmaps_max_l, confmaps_max_r])
        # visualize_box_confmaps(X, np.transpose(confmaps, (0,2,3,1)) , model_type)
        X = np.concatenate((input_left, input_right), axis=0)

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
        Ypk = get_ensemble_results(sum_of_confmaps, num_samples)
        confmaps = sum_of_confmaps / len(models_list)
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
    save_confmaps=False
    save_predictions_to_h5(X, Ypk, box_dset, box_path, confmaps, confmaps_max, confmaps_min, model_path,
                                           num_samples, out_path, prediction_runtime, save_confmaps, t0_all,
                                           weights_path)
    total_runtime = time() - t0_all
    if verbose:
        print("Saved [%.1fs]" % (time() - t0))

        print("Total runtime: %.1f mins" % (total_runtime / 60))
        print("Total performance: %.3f FPS" % (num_samples / total_runtime))





if __name__ == "__main__":
    box_path = r"C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\movie_test_set\movie_dataset_900_frames_5_channels_ds_3tc_7tj.h5"
    # out_path = r"C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\main_dataset_1000_frames_5_channels\head_tail_dataset\HEAD_TAIL_800_frames_7_11\predict_over_train_set.h5"
    # model_path = r"C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\main_dataset_1000_frames_5_channels\head_tail_dataset\HEAD_TAIL_800_frames_7_11\final_model.h5"

    # model_type = PER_WING_SPLIT_TO_3
    # model_type = PER_WING_PER_POINT
    model_type = PER_WING
    # model_type = ENSEMBLE
    # model_type = NO_MASKS
    # model_type = HEAD_TAIL

    test_box = False
    model_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\per_wing_model_trained_by_800_images_segmented_masks_15_10\final_model.h5"
    out_path = r"/train_nn_project/models/segmented masks/per_wing_model_trained_by_800_images_segmented_masks_15_10/roni_masks_predictions_over_movie.h5"
    predict_box(test_box, box_path, model_path, out_path, model_type=model_type, box_dset="/box", verbose=True,
                overwrite=False, save_confmaps=False, batch_size=8)


    # model_paths = []
    # for i in range(1, 10):
    #     path = rf"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\ensamble\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_{i}\final_model.h5"
    #     model_paths.append(path)
    # model_path = model_paths
    # predict_box(test_box, box_path, model_path, out_path, model_type=model_type, box_dset="/box", verbose=True,
    #             overwrite=False, save_confmaps=False, batch_size=8)

