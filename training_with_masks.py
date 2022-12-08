import os
from time import time
import shutil

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from training import preprocess

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LambdaCallback, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from scipy.io import loadmat, savemat
# import skimage
# from skimage.morphology import binary_dilation, disk
from scipy.ndimage import binary_dilation

from viz import show_pred, show_confmap_grid, plot_history
# from pose_estimation_prediction import fix_masks
from pose_estimation_models import basic_nn
PER_WING_MODEL = 'PER_WING_MODEL'
ALL_POINTS_MODEL = 'ALL_POINTS_MODEL'
PER_POINT_PER_WING_MODEL = 'PER_POINT_PER_WING_MODEL'
SPLIT_2_2_3_MODEL = 'SPLIT_2_2_3_MODEL'
MEAN_SQUARE_ERROR = "MEAN_SQUARE_ERROR"
EUCLIDIAN_DISTANCE = "EUCLIDIAN_DISTANCE"
TWO_CLOSE_POINTS_TOGATHER_NO_MASKS = "TWO_CLOSE_POINTS_TOGATHER_NO_MASKS"
MOVIE_TRAIN_SET = "MOVIE_TRAIN_SET"
RANDOM_TRAIN_SET = "RANDOM_TRAIN_SET"
HEAD_TAIL = "HEAD_TAIL"
LEFT_INDEXES = np.arange(0,7)
RIGHT_INDEXES = np.arange(7,14)
"""
things to try:

mirroring the images during generation of training set

color jittering during augmentation
try histogram eqaulization

0 mean the image: subtract the mean of each channel
make loss euclidian distance between predicted point and original point
cross entropy loss between two confidence maps
add batch normalization layers  
Xavier normal initializer
do enseble over different sigmas
manipulate image histogram to make it more sharp
do random or grid search for hyper-parameters (start by course then to fine):
hyper-parameters: 
sigma of output, number of filters, number of batches, dilation rate, kernel size, learning rate, number of convolution blocks,
activations, dropouts, 

modify the loss to by on the x,y coordinates, without confmaps.
"""
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


def euclidian_distance_loss(y_true, y_pred):
    """
    costume made loss to compute euclidian distance between two points
    :param y_true: the labeled confmaps
    :param y_pred: the predicted confmaps
    :return: the mean square loss
    """
    true_x_y = tf_find_peaks(y_true)
    pred_x_y = tf_find_peaks(y_pred)
    return tf.norm(true_x_y - pred_x_y, ord='euclidean')


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


class LossHistory(Callback):
    def __init__(self, run_path):
        super().__init__()
        self.run_path = run_path

    def on_train_begin(self, logs={}):
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        # Append to log list
        self.history.append(logs.copy())

        # Save history so far to MAT file
        savemat(os.path.join(self.run_path, "history.mat"),
                {k: [x[k] for x in self.history] for k in self.history[0].keys()})

        # Plot graph
        plot_history(self.history, save_path=os.path.join(self.run_path, "history.png"))


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


def augmented_data_generator(batch_size, box, confmap, seed=0, rotation_range=180):
    # we create two instances with the same arguments
    data_gen_args = dict(featurewise_center=True,
                         rotation_range=rotation_range,
                         zoom_range=[0.8, 1.2])
    # data generator
    datagen_x = ImageDataGenerator(**data_gen_args)
    datagen_y = ImageDataGenerator(**data_gen_args)
    # prepare iterator
    datagen_x.fit(box, augment=True, seed=seed)
    datagen_y.fit(confmap, augment=True, seed=seed)
    flow_box = datagen_x.flow(box, batch_size=batch_size, seed=seed)
    flow_conf = datagen_y.flow(confmap, batch_size=batch_size, seed=seed)
    train_generator = zip(flow_box, flow_conf)
    return train_generator


def train(box, confmaps,  *, data_path='',
          base_output_path="models",
          run_name=None,
          data_name=None,
          net_name="leap_cnn",
          clean=False,
          box_dset="box",
          confmap_dset="confmaps",
          loss_function=MEAN_SQUARE_ERROR,
          val_size=0.15,
          preshuffle=True,
          filters=64,
          rotate_angle=15,
          epochs=50,
          batch_size=32,
          batches_per_epoch=50,
          validation_steps=10,
          viz_idx=5,
          reduce_lr_factor=0.1,
          reduce_lr_patience=3,
          reduce_lr_min_delta=1e-5,
          reduce_lr_cooldown=0,
          reduce_lr_min_lr=1e-10,
          save_every_epoch=False,
          seed=0,
          dilation_rate=2,
          amsgrad=False,
          upsampling_layers=False,
          ):

    # box, confmaps = load_dataset(data_path, X_dset=box_dset, Y_dset=confmap_dset,  model_type=model_type)
    train_box, train_confmap, val_box, val_confmap, train_idx, val_idx = train_val_split(box, confmaps,
                                                                                         val_size=val_size,
                                                                                         shuffle=preshuffle)
    viz_sample = (val_box[viz_idx], val_confmap[viz_idx])

    # Pull out metadata
    img_size = box.shape[1:]
    num_output_channels = confmaps.shape[-1]
    print("img_size:", img_size)
    print("num_output_channels:", num_output_channels)

    run_path = create_run_folders(run_name, base_path=base_output_path, clean=clean)

    # Initialize vision callbacks
    history_callback = LossHistory(run_path=run_path)
    reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=reduce_lr_factor,
                                           patience=reduce_lr_patience, verbose=1, mode="auto",
                                           epsilon=reduce_lr_min_delta, cooldown=reduce_lr_cooldown,
                                           min_lr=reduce_lr_min_lr)
    if save_every_epoch:
        checkpointer = ModelCheckpoint(filepath=os.path.join(run_path, "weights/weights.{epoch:03d}-{val_loss:.9f}.h5"),
                                       verbose=1, save_best_only=False)
    else:
        checkpointer = ModelCheckpoint(filepath=os.path.join(run_path, "best_model.h5"), verbose=1, save_best_only=True)

    # get model
    # filters=64, num_blocks=2 was good
    if loss_function == MEAN_SQUARE_ERROR:
        loss_function = "mean_squared_error"
    elif loss_function == EUCLIDIAN_DISTANCE:
        loss_function = euclidian_distance_loss

    model = basic_nn(img_size, num_output_channels, filters=filters, num_blocks=2, kernel_size=3, loss_function=loss_function, dilation_rate=dilation_rate)

    # Save initial network
    model.save(os.path.join(run_path, "initial_model.h5"))

    viz_grid_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: show_confmap_grid(model, *viz_sample, plot=True,
                                                                                          save_path=os.path.join(
                                                                                              run_path,
                                                                                              "viz_confmaps/confmaps_%03d.png" % epoch),
                                                                                          show_figure=False))
    viz_pred_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: show_pred(model, *viz_sample,
                                                                                  save_path=os.path.join(run_path,
                                                                                                         "viz_pred/pred_%03d.png" % epoch),
                                                                                  show_figure=False))

    # Train!
    epoch0 = 0
    t0_train = time()

    print("creating generators")

    train_datagen = augmented_data_generator(batch_size, train_box, train_confmap, seed)
    val_datagen = augmented_data_generator(batch_size, val_box, val_confmap, seed)

    print("creating generators - done!")

    training = model.fit(
        train_datagen,
        initial_epoch=epoch0,
        epochs=epochs,
        verbose=1,
        # use_multiprocessing=True,
        # workers=8,
        steps_per_epoch=batches_per_epoch,
        max_queue_size=512,
        shuffle=False,
        validation_data=val_datagen,
        callbacks=[
            reduce_lr_callback,
            checkpointer,
            history_callback,
            # viz_grid_callback,
            # viz_pred_callback
        ],
        validation_steps=validation_steps
    )

    # Compute total elapsed time for vision
    elapsed_train = time() - t0_train
    print("Total runtime: %.1f mins" % (elapsed_train / 60))

    # Save final model
    model.history = history_callback.history
    model.save(os.path.join(run_path, "final_model.h5"))


def test_generators(data_path):
    box, confmap = load_dataset(data_path)
    seed = 0
    batch_size = 8
    data_gen_args = dict(featurewise_center=True,
                         rotation_range=30,
                         zoom_range=[0.8, 1.2])
    # data generator
    datagen_x = ImageDataGenerator(**data_gen_args)
    datagen_y = ImageDataGenerator(**data_gen_args)
    # prepare iterator
    datagen_x.fit(box, augment=True, seed=seed)
    datagen_y.fit(confmap, augment=True, seed=seed)
    flow_box = datagen_x.flow(box, batch_size=batch_size, seed=seed)
    flow_conf = datagen_y.flow(confmap, batch_size=batch_size, seed=seed)
    matplotlib.use('TkAgg')

    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)
        # generate batch of images
        batch_x = flow_box.next()
        batch_y = flow_conf.next()
        # convert to unsigned integers for viewing
        image = batch_x[0][:, :, 1]
        conf = np.squeeze(batch_y[0])
        # plot raw pixel data
        plt.imshow(image + conf)
    # show the figure
    plt.show()


def visualize_box_confmaps(box, confmaps, model_type):
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
        confmap = np.sum(confmaps[image, :, :, :], axis=2)
        if model_type == ALL_POINTS_MODEL:
            masks = np.sum(box[image, :, :, [3, 4]], axis=0)
        else:
            masks = box[image, :, :, 3]
        img = np.zeros((192, 192, 3))
        img[:, :, 1] = fly
        img[:, :, 2] = confmap
        img[:, :, 0] = masks
        plt.imshow(img)
        plt.show()
        a=0
    # num_mirrors = box.shape[0]
    # num_frames = box.shape[1]
    # num_cams = box.shape[2]
    # for side in range(num_mirrors):
    #     for frame in [15, 16, 17, 18, 19, 20, 21, 45, 46, 47, 48, 49, 50, 51]:
    #         print(frame)
    #         fig = plt.figure(figsize=(8, 8))
    #         for cam in range(num_cams):
    #             # fly = box[side, frame, cam, :, :, 1]
    #             # fly =
    #             confmap = np.sum(confmaps[side, frame, cam, :, :, :], axis=2)
    #             if model_type == ALL_POINTS_MODEL:
    #                 masks = np.sum(box[side, frame, cam, :, :, [3,4]], axis=0)
    #             else:
    #                 masks = box[side, frame, cam, :, :, 3]
    #             img = np.zeros((192, 192, 3))
    #             img[:, :, 1] = fly
    #             img[:, :, 2] = confmap
    #             img[:, :, 0] = masks
    #             fig.add_subplot(rows, columns, cam + 1)
    #             plt.imshow(img)
    #         plt.show()
    #     a=0


def split_per_wing(box, confmaps, model_type, trainset_type):
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
            if left_values < 3 and right_values < 3:
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
            if left_values < 3 or right_values < 3:
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
                    a=0

    return box, problematic_masks


def increase_masks_size(box, radius=5):
    num_training_samples = box.shape[0]
    for image_num in range(num_training_samples):
        mask = box[image_num, :, :, 3]
        non_0_1 = np.count_nonzero(mask)
        dilated_mask = binary_dilation(mask, iterations=radius).astype(int)
        non_0_2 = np.count_nonzero(dilated_mask)
        box[image_num, :, :, 3] = dilated_mask
        # matplotlib.use('TkAgg')
        # plt.imshow(dilated_mask - mask)
        # plt.show()
        a=0
    return box


# def reshape_for_train(box, confmaps, model_type):




def train_model(model_type, data_path, sigma='3',
                masks=True,
                test_path='',
                loss_function=MEAN_SQUARE_ERROR,
                mix_with_test=False,
                val_fraction=0.25,
                filters=64,
                batch_size=100,
                batches_per_epoch=100,
                epochs=30,
                validation_steps=200,
                seed=0,
                dilation_rate=2):
    """
    train a model or models, depending on the specified model type
    :param model_type: the type of the model on which to train:
    PER_WING_MODEL:
    ALL_POINTS_MODEL:
    PER_POINT_PER_WING_MODEL:
    SPLIT_2_2_3_MODEL:
    :param data_path: path of h5 file of training data
    :return: trained model or models
    """

    box, confmaps = load_dataset(data_path)
    if mix_with_test == True:
        test_box, test_confmaps = load_dataset(test_path, X_dset="box", Y_dset="confmaps")
        trainset_type = MOVIE_TRAIN_SET
        test_box[0], test_confmaps[0] = split_per_wing(test_box[0], test_confmaps[0], ALL_POINTS_MODEL, trainset_type)
        test_box[1], test_confmaps[1] = split_per_wing(test_box[1], test_confmaps[1], ALL_POINTS_MODEL, trainset_type)
        test_box[0], problematic_masks_inds = fix_movie_masks(test_box[0])
        test_box[1], problematic_masks_inds = fix_movie_masks(test_box[1])
        problematic_masks_inds = np.array(problematic_masks_inds)
        box = np.concatenate((box, test_box), axis=1)
        confmaps = np.concatenate((confmaps, test_confmaps), axis=1)


    num_frames = box.shape[0]/2
    if model_type == ALL_POINTS_MODEL or model_type == HEAD_TAIL:
        # visualize_box_confmaps(box, confmaps, model_type)
        if not masks or model_type == HEAD_TAIL:
            box = box[:,:,:,[0,1,2]]
        train(box, confmaps,
              run_name=f"{model_type}_{filters}",
              val_size=val_fraction,
              loss_function=loss_function,
              epochs=epochs,
              batch_size=batch_size,
              batches_per_epoch=batches_per_epoch,
              validation_steps=validation_steps,
              filters=filters,
              dilation_rate=dilation_rate)

    elif model_type == PER_WING_MODEL:
        box_0, confmaps_0 = split_per_wing(box[0], confmaps[0], PER_WING_MODEL, RANDOM_TRAIN_SET)
        box_1, confmaps_1 = split_per_wing(box[1], confmaps[1], PER_WING_MODEL, RANDOM_TRAIN_SET)
        box = np.concatenate((box_0, box_1), axis=0)
        confmaps = np.concatenate((confmaps_0, confmaps_1), axis=0)
        box = np.reshape(box, newshape=[box.shape[0]*box.shape[1], box.shape[2], box.shape[3], box.shape[4]])
        confmaps = np.reshape(confmaps, newshape=[confmaps.shape[0]*confmaps.shape[1], confmaps.shape[2], confmaps.shape[3], confmaps.shape[4]])
        box = increase_masks_size(box)
        # visualize_box_confmaps(box, confmaps, model_type)
        train(box, confmaps,
              run_name=f"per_wing_model_trained_by_{num_frames}_images_segmented_masks",
              val_size=val_fraction,
              loss_function=loss_function,
              epochs=epochs,
              batch_size=batch_size,
              batches_per_epoch=batches_per_epoch,
              validation_steps=validation_steps,
              filters=filters,
              dilation_rate=dilation_rate,
              seed=seed)

    # elif model_type == PER_POINT_PER_WING_MODEL:
    #     box, confmaps = split_per_wing(box, confmaps)
    #     points_per_wing = confmaps.shape[-1]
    #     tensor_confmaps = np.transpose(np.array([confmaps]), (4, 1, 2, 3, 0))
    #     for point in range(points_per_wing):
    #         confmaps_i = tensor_confmaps[point, :, :, :, :]
    #         # visualize_box_confmaps(box, confmaps_i, model_type)
    #         train(box, confmaps_i,
    #           run_name=f"per_point_per_wing_point_num_{point + 1}_filters_{filters}_trained_by_{num_frames}_images",
    #           val_size=val_fraction,
    #           loss_function=loss_function,
    #           epochs=epochs,
    #           batch_size=batch_size,
    #           batches_per_epoch=batches_per_epoch,
    #           validation_steps=validation_steps,
    #           filters=filters,
    #           dilation_rate=dilation_rate)
    #
    # elif model_type == TWO_CLOSE_POINTS_TOGATHER_NO_MASKS:
    #     box = box[:, :, :, [0, 1, 2]]
    #     confmaps = confmaps[:, :, :, [2, 6, 9, 13]]
    #     train(box, confmaps,
    #           run_name=f"two_points_same_time",
    #           val_size=val_fraction,
    #           loss_function=loss_function,
    #           epochs=epochs,
    #           batch_size=batch_size,
    #           batches_per_epoch=batches_per_epoch,
    #           validation_steps=validation_steps,
    #           filters=filters,
    #           dilation_rate=dilation_rate)
    #
    # elif model_type == SPLIT_2_2_3_MODEL:
    #     box, confmaps = split_per_wing(box, confmaps)
    #     conf_points_1_2 = confmaps[:, :, :, [0, 1]]
    #     conf_points_3_4 = confmaps[:, :, :, [2, 3]]
    #     conf_points_5_6_7 = confmaps[:, :, :, [4, 5, 6]]
    #     # visualize_box_confmaps(box, conf_points_1_2, data_path)
    #     model_confs = [conf_points_1_2, conf_points_3_4, conf_points_5_6_7]
    #     points = ['1-2', '3-4', '5-6-7']
    #     for i, conf in enumerate(model_confs):
    #         train(box, conf,
    #               run_name=f"model_2_2_3_points_{points[i]}_filters_{filters}_trained_by_{num_frames}_images",
    #               val_size=val_fraction,
    #               loss_function=loss_function,
    #               epochs=epochs,
    #               batch_size=batch_size,
    #               batches_per_epoch=batches_per_epoch,
    #               validation_steps=validation_steps,
    #               filters=filters,
    #               dilation_rate=dilation_rate,)

if __name__ == '__main__':
    model_type = PER_WING_MODEL
    data_path = "pre_train_100_frames_segmented_masks_reshaped.h5"
    test_path = "movie_dataset_71_frames_segmented_masks_reshaped.h5"
    train_model(model_type=PER_WING_MODEL,
                mix_with_test=True,
                data_path=data_path,
                test_path=test_path,
                val_fraction=0.1,
                filters=64,
                batch_size=100,
                batches_per_epoch=40,
                epochs=30,
                validation_steps=50,)
