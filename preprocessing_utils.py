from scipy.ndimage import binary_dilation, binary_closing
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

PER_WING_MODEL = 'PER_WING_MODEL'
ALL_POINTS_MODEL = 'ALL_POINTS_MODEL'
PER_POINT_PER_WING_MODEL = 'PER_POINT_PER_WING_MODEL'
SPLIT_2_2_3_MODEL = 'SPLIT_2_2_3_MODEL'
TRAIN_ON_2_GOOD_CAMERAS_MODEL = "TRAIN_ON_2_GOOD_CAMERAS_MODEL"
MEAN_SQUARE_ERROR = "MEAN_SQUARE_ERROR"
EUCLIDIAN_DISTANCE = "EUCLIDIAN_DISTANCE"
TWO_CLOSE_POINTS_TOGATHER_NO_MASKS = "TWO_CLOSE_POINTS_TOGATHER_NO_MASKS"
MOVIE_TRAIN_SET = "MOVIE_TRAIN_SET"
RANDOM_TRAIN_SET = "RANDOM_TRAIN_SET"
HEAD_TAIL = "HEAD_TAIL"
LEFT_INDEXES = np.arange(0,7)
RIGHT_INDEXES = np.arange(7,14)


def preprocess(X, permute=(0, 3, 2, 1)):
    """ Normalizes input data. """

    # Add singleton dim for single images
    if X.ndim == 3:
        X = X[None, ...]

    # Adjust dimensions
    if permute != None:
        X = np.transpose(X, permute)

    # Normalize
    if X.dtype == "uint8" or np.max(X) > 1:
        X = X.astype("float32") / 255

    return X


def train_val_split(X, Y, val_size=0.15, shuffle=True):
    """ Splits datasets into vision and validation sets. """

    if val_size < 1:
        val_size = int(np.round(len(X) * val_size))

    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)

    val_idx = idx[:val_size]
    idx = idx[val_size:]

    return X[idx], Y[idx], X[val_idx], Y[val_idx], idx, val_idx

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


def visualize_box_confmaps(box, confmaps, model_type):
    """ visualize the input to the network """
    matplotlib.use('TkAgg')
    w = 10
    h = 10
    columns = 2
    rows = 2
    num_images = box.shape[0]
    for image in range(num_images):
        print(image)
        fig = plt.figure(figsize=(8, 8))
        fly = box[image, :, :, 1]
        masks = np.zeros((192, 192))
        try:
            confmap = np.sum(confmaps[image, :, :, :], axis=2)
        except: a=0
        if model_type == ALL_POINTS_MODEL:
            masks = np.sum(box[image, :, :, [3, 4]], axis=0)
        else:
            try:
                masks = box[image, :, :, 3]
            except:
                a = 0

        img = np.zeros((192, 192, 3))
        img[:, :, 1] = fly
        try:
            img[:, :, 2] = confmap
        except: a=0
        img[:, :, 0] = masks
        plt.imshow(img)
        plt.show()



def create_run_folders(run_name, base_path="models", clean=False):
    """ Creates subfolders necessary for outputs of vision. """

    def is_empty_run(run_path):
        weights_path = os.path.join(run_path, "weights")
        has_weights_folder = os.path.exists(weights_path)
        return not has_weights_folder or len(os.listdir(weights_path)) == 0

    run_path = os.path.join(base_path, run_name)

    if not clean:
        initial_run_path = run_path
        i = 1
        while os.path.exists(run_path):  # and not is_empty_run(run_path):
            run_path = "%s_%02d" % (initial_run_path, i)
            i += 1

    if os.path.exists(run_path):
        shutil.rmtree(run_path)

    os.makedirs(run_path)
    os.makedirs(os.path.join(run_path, "weights"))
    os.makedirs(os.path.join(run_path, "viz_pred"))
    os.makedirs(os.path.join(run_path, "viz_confmaps"))
    print("Created folder:", run_path)

    return run_path


def split_per_wing(box, confmaps, model_type, trainset_type):
    """ make sure the confmaps fits the wings """
    min_in_mask = 3
    num_joints = confmaps.shape[-1]
    num_joints_per_wing = int(num_joints/2)
    LEFT_INDEXES = np.arange(0,num_joints_per_wing)
    RIGHT_INDEXES = np.arange(num_joints_per_wing, 2*num_joints_per_wing)

    left_wing_box = box[:, :, :, :, [0, 1, 2, 3]]
    right_wing_box = box[:, :, :, :, [0, 1, 2, 4]]
    right_wing_confmaps = confmaps[:, :, :, :, LEFT_INDEXES]
    left_wing_confmaps = confmaps[:, :, :, :, RIGHT_INDEXES]

    num_frames = box.shape[0]
    num_cams = box.shape[1]
    num_pts_per_wing = right_wing_confmaps.shape[-1]
    left_peaks = np.zeros((num_frames, num_cams, 2, num_pts_per_wing))
    right_peaks = np.zeros((num_frames, num_cams, 2, num_pts_per_wing))
    for cam in range(num_cams):
        l_p = tf_find_peaks(left_wing_confmaps[:, cam, :, :, :])[:, :2, :].numpy()
        r_p = tf_find_peaks(right_wing_confmaps[:, cam, :, :, :])[:, :2, :].numpy()
        left_peaks[:, cam, :, :] = l_p
        right_peaks[:, cam, :, :] = r_p

    left_peaks = left_peaks.astype(int)
    right_peaks = right_peaks.astype(int)

    new_left_wing_box = np.zeros(left_wing_box.shape)
    new_right_wing_box = np.zeros(right_wing_box.shape)
    new_right_wing_confmaps = np.zeros(right_wing_confmaps.shape)
    new_left_wing_confmaps = np.zeros(left_wing_confmaps.shape)

    num_of_bad_masks = 0
    # fit confmaps to wings
    num_frames = box.shape[0]
    for frame in range(num_frames):
        for cam in range(num_cams):
            append = True
            fly_image = left_wing_box[frame, cam, :, :, [0, 1, 2]]

            left_confmap = left_wing_confmaps[frame, cam, :, :, :]
            right_confmap = right_wing_confmaps[frame, cam, :, :, :]

            left_mask = left_wing_box[frame, cam, :, :, 3]
            right_mask = right_wing_box[frame, cam, :, :, 3]

            left_peaks_i = left_peaks[frame, cam, :, :]
            right_peaks_i = right_peaks[frame, cam, :, :]

            # check peaks
            left_values = 0
            right_values = 0
            for i in range(left_peaks_i.shape[-1]):
                left_values += left_mask[left_peaks_i[1, i], left_peaks_i[0, i]]
                right_values += right_mask[right_peaks_i[1, i], right_peaks_i[0, i]]

            # switch masks if peaks are completely missed
            if left_values < min_in_mask and right_values < min_in_mask:
                temp = left_mask
                left_mask = right_mask
                right_mask = temp

            # check peaks again
            left_values = 0
            right_values = 0
            for i in range(left_peaks_i.shape[-1]):
                left_values += left_mask[left_peaks_i[1, i], left_peaks_i[0, i]]
                right_values += right_mask[right_peaks_i[1, i], right_peaks_i[0, i]]

            # don't append if one mask is missing
            mask_exist = True
            if left_values < min_in_mask or right_values < min_in_mask:
                mask_exist = False
                num_of_bad_masks += 1

            if trainset_type == MOVIE_TRAIN_SET or (trainset_type == RANDOM_TRAIN_SET and mask_exist):
                # copy fly image
                new_left_wing_box[frame, cam, :, :, [0, 1, 2]] = fly_image
                new_left_wing_box[frame, cam, :, :, 3] = left_mask
                # copy mask
                new_right_wing_box[frame, cam, :, :, [0, 1, 2]] = fly_image
                new_right_wing_box[frame, cam, :, :, 3] = right_mask
                # copy confmaps
                new_right_wing_confmaps[frame, cam, :, :, :] = right_confmap
                new_left_wing_confmaps[frame, cam, :, :, :] = left_confmap

    if model_type == PER_WING_MODEL:
        box = np.concatenate((new_left_wing_box, new_right_wing_box), axis=0)
        confmaps = np.concatenate((new_left_wing_confmaps, new_right_wing_confmaps), axis=0)

    elif model_type == ALL_POINTS_MODEL:
        # copy fly
        box[:, :, :, :, [0, 1, 2]] = new_left_wing_box[:, :, :, :, [0, 1, 2]]
        # copy left mask
        box[:, :, :, :, 3] = new_left_wing_box[:, :, :, :, 3]
        box[:, :, :, :, 4] = new_right_wing_box[:, :, :, :, 3]
        confmaps[:, :, :, :, LEFT_INDEXES] = new_left_wing_confmaps
        confmaps[:, :, :, :, RIGHT_INDEXES] = new_right_wing_confmaps

    print(f"finish preprocess. number of bad masks = {num_of_bad_masks}")
    return box, confmaps


def fix_movie_masks(box):
    """
    goes throw each frame, if there is no mask for a specific wing, unite masks of the closest times before and after
    this frame.
    :param box: a box of size (num_frames, 20, 192, 192)
    :return: same box
    """
    search_range = 5
    num_channels = 5
    num_frames = int(box.shape[0])
    problematic_masks = []
    for frame in range(num_frames):
        for cam in range(4):
            for mask_num in range(2):
                mask = box[frame, cam, :, :, 3 + mask_num]
                if np.all(mask == 0):  # check if all 0:
                    problematic_masks.append((frame, cam, mask_num))
                    # find previous matching mask
                    prev_mask = np.zeros(mask.shape)
                    next_mask = np.zeros(mask.shape)
                    for prev_frame in range(frame - 1, max(0, frame - search_range - 1), -1):
                        prev_mask_i = box[prev_frame, cam, :, :, 3 + mask_num]
                        if not np.all(prev_mask_i == 0):  # there is a good mask
                            prev_mask = prev_mask_i
                            break
                    # find next matching mask
                    for next_frame in range(frame + 1, min(num_frames, frame + search_range)):
                        next_mask_i = box[next_frame, cam, :, :, 3 + mask_num]
                        if not np.all(next_mask_i == 0):  # there is a good mask
                            next_mask = next_mask_i
                            break
                    # combine the 2 masks
                    new_mask = prev_mask + next_mask
                    new_mask[new_mask >= 1] = 1
                    # replace empty mask with new mask
                    box[frame, cam, :, :, 3 + mask_num] = new_mask
                    # matplotlib.use('TkAgg')
                    # plt.imshow(new_mask)
                    # plt.show()

    return box, problematic_masks


def adjust_mask(mask, radious=5):
    # mask = binary_dilation(mask, iterations=radius).astype(int)
    mask = binary_closing(mask).astype(int)
    return mask


def adjust_masks_size(box, train_or_predict, radius=5):
    """ adjust the size of the wings masks """
    if train_or_predict == "TRAIN":
        num_training_samples = box.shape[0]
        for image_num in range(num_training_samples):
            mask = box[image_num, :, :, 3]
            non_0_1 = np.count_nonzero(mask)
            adjusted_mask = adjust_mask(mask)
            non_0_2 = np.count_nonzero(adjusted_mask)
            box[image_num, :, :, 3] = adjusted_mask
            # matplotlib.use('TkAgg')
            # plt.imshow(adjusted_mask - mask)
            # plt.show()

    elif train_or_predict == "PREDICT":
        num_channels = 5
        num_frames = int(box.shape[0])
        for frame in range(num_frames):
            for cam in range(4):
                for mask_num in range(2):
                    mask = box[frame, 3 + mask_num + num_channels * cam, :, :]
                    adjusted_mask = adjust_mask(mask)
                    box[frame, 3 + mask_num + num_channels * cam, :, :] = adjusted_mask
    return box


def reshape_to_cnn_input(box, confmaps):
    """ reshape the  input from """
    confmaps = np.transpose(confmaps, (5,4,3,2,1,0))
    confmaps = np.reshape(confmaps, [-1, confmaps.shape[-3], confmaps.shape[-2], confmaps.shape[-1]])
    box = np.transpose(box, (5, 4, 3, 2, 1, 0))
    box = np.reshape(box, [-1, box.shape[-3], box.shape[-2], box.shape[-1]])
    return box, confmaps


def get_mix_with_test(box, confmaps, test_path):
    test_box, test_confmaps = load_dataset(test_path, X_dset="box", Y_dset="confmaps")
    trainset_type = MOVIE_TRAIN_SET
    test_box[0], test_confmaps[0] = split_per_wing(test_box[0], test_confmaps[0], ALL_POINTS_MODEL, trainset_type)
    test_box[1], test_confmaps[1] = split_per_wing(test_box[1], test_confmaps[1], ALL_POINTS_MODEL, trainset_type)
    test_box[0], problematic_masks_inds = fix_movie_masks(test_box[0])
    test_box[1], problematic_masks_inds = fix_movie_masks(test_box[1])
    problematic_masks_inds = np.array(problematic_masks_inds)
    box = np.concatenate((box, test_box), axis=1)
    confmaps = np.concatenate((confmaps, test_confmaps), axis=1)
    return box, confmaps

def take_2_good_cameras(box, confmaps):
    num_frames = box.shape[0]
    num_cams = box.shape[1]
    new_num_cams = 2
    image_shape = box.shape[2]
    num_channels_box = box.shape[-1]
    num_channels_confmap = confmaps.shape[-1]
    new_box = np.zeros((num_frames, new_num_cams, image_shape, image_shape, num_channels_box))
    new_confmap = np.zeros((num_frames, new_num_cams, image_shape, image_shape, num_channels_confmap))
    for frame in range(num_frames):
        wings_size = np.zeros(4)
        for cam in range(num_cams):
            wing_mask = box[frame, cam, :, :, -1]
            wings_size[cam] = np.count_nonzero(wing_mask)
        wings_size_argsort = np.argsort(wings_size)[::-1]
        best_2_cameras = wings_size_argsort[:2]
        new_box[frame, ...] = box[frame, best_2_cameras, ...]
        new_confmap[frame, ...] = confmaps[frame, best_2_cameras, ...]
    return new_box, new_confmap



def do_reshape_per_wing(box, confmaps, model_type=PER_WING_MODEL):
    """ reshape input to a per wing model input """
    box_0, confmaps_0 = split_per_wing(box[0], confmaps[0], PER_WING_MODEL, RANDOM_TRAIN_SET)
    box_1, confmaps_1 = split_per_wing(box[1], confmaps[1], PER_WING_MODEL, RANDOM_TRAIN_SET)
    box = np.concatenate((box_0, box_1), axis=0)
    confmaps = np.concatenate((confmaps_0, confmaps_1), axis=0)
    if model_type == TRAIN_ON_2_GOOD_CAMERAS_MODEL:
        box, confmaps = take_2_good_cameras(box, confmaps)
    box = np.reshape(box, newshape=[box.shape[0] * box.shape[1], box.shape[2], box.shape[3], box.shape[4]])
    confmaps = np.reshape(confmaps,
                          newshape=[confmaps.shape[0] * confmaps.shape[1], confmaps.shape[2], confmaps.shape[3],
                                    confmaps.shape[4]])
    box = adjust_masks_size(box, "TRAIN")
    return box, confmaps


def load_dataset(data_path, X_dset="box", Y_dset="confmaps", permute=(0, 3, 2, 1)):
    """ Loads and normalizes datasets. """
    # Load
    with h5py.File(data_path, "r") as f:
        X = f[X_dset][:]
        Y = f[Y_dset][:]

    # Adjust dimensions
    X = preprocess(X, permute=None)
    Y = preprocess(Y, permute=None)
    return X, Y


def test_generators(data_path):
    box, confmap = load_dataset(data_path)
    image_size = confmap.shape[-2]
    num_channels_img = box.shape[-1]
    num_channels_confmap = confmap.shape[-1]
    box = box.reshape([-1, image_size, image_size, num_channels_img])
    confmap = confmap.reshape([-1, image_size, image_size, num_channels_confmap])
    seed = 0
    batch_size = 8
    data_gen_args = dict(rotation_range=180,
                         zoom_range=[0.8, 1.2],
                         horizontal_flip=True,
                         vertical_flip=True,)
    # data generator
    datagen_x = ImageDataGenerator(**data_gen_args)
    datagen_y = ImageDataGenerator(**data_gen_args)
    # prepare iterator
    datagen_x.fit(box, augment=True, seed=seed)
    datagen_y.fit(confmap, augment=True, seed=seed)
    flow_box = datagen_x.flow(box, batch_size=batch_size, seed=seed)
    flow_conf = datagen_y.flow(confmap, batch_size=batch_size, seed=seed)
    matplotlib.use('TkAgg')
    for j in range(5):
        for i in range(9):
            # define subplot
            plt.subplot(330 + 1 + i)
            # generate batch of images
            batch_x = flow_box.next()
            batch_y = flow_conf.next()
            # convert to unsigned integers for viewing
            image = batch_x[0][:, :, 1]
            conf = np.sum(np.squeeze(batch_y[0]), axis=-1)
            # plot raw pixel data
            plt.imshow(image + conf)
        # show the figure
        plt.show()

