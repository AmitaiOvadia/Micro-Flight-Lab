from time import time
from PIL import Image
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LambdaCallback, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.io import loadmat, savemat
# import skimage
# from skimage.morphology import binary_dilation, disk
# from scipy.ndimage import binary_dilation, binary_closing
from scipy.ndimage import gaussian_filter, shift
from viz import show_pred, show_confmap_grid, plot_history
from pose_estimation_models import basic_nn, ed3d, two_wings_net, pretrained_per_wing_vgg
from preprocessing_utils import *

TWO_WINGS_TOGATHER = "TWO_WINGS_TOGATHER"

PER_WING_MODEL = 'PER_WING_MODEL'
ALL_POINTS_MODEL = 'ALL_POINTS_MODEL'
PER_POINT_PER_WING_MODEL = 'PER_POINT_PER_WING_MODEL'
SPLIT_2_2_3_MODEL = 'SPLIT_2_2_3_MODEL'
TRAIN_ON_2_GOOD_CAMERAS_MODEL = "TRAIN_ON_2_GOOD_CAMERAS_MODEL"
TRAIN_ON_3_GOOD_CAMERAS_MODEL = "TRAIN_ON_3_GOOD_CAMERAS_MODEL"
BODY_PARTS_MODEL = "BODY_PART_MODEL"
ALL_CAMS = "ALL_CAMS"
VGG_PER_WING = "VGG_PER_WING"
MEAN_SQUARE_ERROR = "MEAN_SQUARE_ERROR"
EUCLIDIAN_DISTANCE = "EUCLIDIAN_DISTANCE"
TWO_CLOSE_POINTS_TOGATHER_NO_MASKS = "TWO_CLOSE_POINTS_TOGATHER_NO_MASKS"
MOVIE_TRAIN_SET = "MOVIE_TRAIN_SET"
RANDOM_TRAIN_SET = "RANDOM_TRAIN_SET"
HEAD_TAIL = "HEAD_TAIL"
INPUT_OF_12_CHANNELS_3_CAMERAS = "INPUT_OF_12_CHANNELS_3_CAMERAS"
LEFT_INDEXES = np.arange(0, 7)
RIGHT_INDEXES = np.arange(7, 14)
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


def augment(img, h_fl, v_fl, rotation_angle, shift_y_x):
    if np.max(img) <= 1:
        img = np.uint8(img * 255)
    if h_fl:
        img = np.fliplr(img)
    if v_fl:
        img = np.flipud(img)
    img = shift(img, shift_y_x)
    img_pil = Image.fromarray(img)
    img_pil = img_pil.rotate(rotation_angle, 3)
    img = np.asarray(img_pil)
    if np.max(img) > 1:
        img = img/255
    return img



def blur_channel(img_channel, sigma):
    """ a (imsize, imsize) numpy array """
    return gaussian_filter(img_channel, sigma=sigma)


def custom_augmentations(img):
    """get an image of shape (height, width, num_channels) and return augmented image"""
    do_horizontal_flip = np.random.randint(2)
    do_vertical_flip = np.random.randint(2)
    rotation_angle = np.random.randint(-180, 180)
    shift_y_x = np.random.randint(-10, 10, 2)
    num_channels = img.shape[-1]
    for channel in range(num_channels):
        img[:, :, channel] = augment(img[:, :, channel], do_horizontal_flip,
                                     do_vertical_flip, rotation_angle, shift_y_x)
    return img



def augmented_data_generator(batch_size, box, confmap, seed=0, rotation_range=180):
    # we create two instances with the same arguments

    # data generator

    data_gen_args = dict(preprocessing_function=custom_augmentations,)
    #
    # data_gen_args = dict(rotation_range=90,
    #                      zoom_range=[0.8, 1.2],
    #                      horizontal_flip=True,
    #                      vertical_flip=True,
    #                      interpolation_order=2,)

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
          epochs=30,
          batch_size=32,
          batches_per_epoch=50,
          validation_steps=10,
          viz_idx=1,
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

    net_input = dict(img_size=img_size, num_output_channels=num_output_channels,
                     filters=filters, num_blocks=2, kernel_size=3)

    if model_type == ALL_CAMS:
        model = ed3d(**net_input)
    elif model_type == TWO_WINGS_TOGATHER:
        model = two_wings_net(**net_input)
    elif model_type == VGG_PER_WING:
        model = pretrained_per_wing_vgg(**net_input)
    else:
        model = basic_nn(**net_input)
    # Save initial network
    model.save(os.path.join(run_path, "initial_model.h5"))

    viz_grid_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: show_confmap_grid(model, *viz_sample, plot=True,
                                                                                          save_path=os.path.join(
                                                                                              run_path,
                                                                                              "viz_confmaps\confmaps_%03d.png" % epoch),
                                                                                          show_figure=False))
    viz_pred_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: show_pred(model, *viz_sample,
                                                                                  save_path=os.path.join(run_path,
                                                                                                         "viz_pred\pred_%03d.png" % epoch),
                                                                                  show_figure=False))
    # Train!
    epoch0 = 0
    t0_train = time()

    print("creating generators")

    train_datagen = augmented_data_generator(batch_size, train_box, train_confmap, seed)
    if validation_steps == None:
        val_datagen = (val_box, val_confmap)
    else:
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
            viz_pred_callback
        ],
        validation_steps=validation_steps
    )

    # Compute total elapsed time for vision
    elapsed_train = time() - t0_train
    print("Total runtime: %.1f mins" % (elapsed_train / 60))

    # Save final model
    model.history = history_callback.history
    model.save(os.path.join(run_path, "final_model.h5"))


def train_model(model_type, data_path,
                run_name='model',
                sigma='3',
                masks=True,
                test_path='',
                loss_function=MEAN_SQUARE_ERROR,
                mix_with_test=False,
                val_fraction=0.1,
                filters=64,
                batch_size=100,
                batches_per_epoch=100,
                epochs=30,
                validation_steps=None,
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
    # box = box[:, :10, :, :, :, :]
    # confmaps = confmaps[:, :10, :, :, :, :]
    if mix_with_test == True and model_type != HEAD_TAIL and model_type != BODY_PARTS_MODEL:
        box, confmaps = get_mix_with_test(box, confmaps, test_path)
    if model_type == ALL_POINTS_MODEL or model_type == HEAD_TAIL or model_type == TWO_WINGS_TOGATHER:
        box, confmaps = reshape_to_cnn_input(box, confmaps)
    elif model_type == PER_WING_MODEL or model_type == VGG_PER_WING:
        box, confmaps = do_reshape_per_wing(box, confmaps)
    elif model_type == TRAIN_ON_2_GOOD_CAMERAS_MODEL or model_type == TRAIN_ON_3_GOOD_CAMERAS_MODEL:
        box, confmaps = do_reshape_per_wing(box, confmaps, model_type)
    elif model_type == BODY_PARTS_MODEL:
        box, confmaps = reshape_to_body_parts(box, confmaps)
    elif model_type == ALL_CAMS:
        box, confmaps = do_reshape_per_wing(box, confmaps, model_type)
    # visualize_box_confmaps(box, confmaps, model_type)
    print(f"number of training samples is {box.shape[0]}")
    train(box, confmaps,
          run_name=run_name,
          val_size=val_fraction,
          loss_function=loss_function,
          epochs=epochs,
          batch_size=batch_size,
          batches_per_epoch=batches_per_epoch,
          validation_steps=validation_steps,
          filters=filters,
          dilation_rate=dilation_rate,
          seed=seed)


if __name__ == '__main__':
    # model_type = PER_WING_MODEL
    # model_type = TRAIN_ON_2_GOOD_CAMERAS_MODEL
    # model_type = TRAIN_ON_3_GOOD_CAMERAS_MODEL
    # model_type = BODY_PARTS_MODEL
    # data_path = "pre_train_100_frames_segmented_masks_reshaped.h5"
    # test_path = r"train_set_movie_14_pts_sigma_3.h5"
    # data_path = r"trainset_random_14_pts_sigma_3.h5"
    # model_type = TRAIN_ON_3_GOOD_CAMERAS_MODEL
    # model_type = ALL_CAMS
    model_type = TWO_WINGS_TOGATHER
    # model_type = VGG_PER_WING

    batch_size = 70
    batches_per_epoch = 200
    filters = 64
    print(f"{model_type}")
    test_path = "train_set_movie_14_pts_yolo_masks.h5"
    data_path = "trainset_random_14_pts_200_frames.h5"
    train_model(model_type=model_type,
                mix_with_test=True,
                test_path=test_path,
                data_path=data_path,
                run_name=f"{model_type}_merged_right_order_22_3_bs_{batch_size}_bpe_{batches_per_epoch}_fil_{filters}",
                filters=filters,
                val_fraction=0.1,
                batch_size=batch_size,
                batches_per_epoch=batches_per_epoch,
                dilation_rate=2,
                epochs=30)
