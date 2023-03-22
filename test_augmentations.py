import os
from PIL import Image, ImageFilter
import h5py
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate, gaussian_filter, shift

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


def load_dataset(data_path, X_dset="box", Y_dset="confmaps", permute=(0, 3, 2, 1)):
    """ Loads and normalizes datasets. """
    # Load
    with h5py.File(data_path, "r") as f:
        X = f[X_dset][:]
        Y = f[Y_dset][:]

    # Adjust dimensions
    X = preprocess(X, permute=None)
    Y = preprocess(Y, permute=None)
    if X.shape[0] != 2:
        X = np.transpose(X, [5, 4, 3, 2, 1, 0])
    if Y.shape[0] != 2:
        Y = np.transpose(Y, [5, 4, 3, 2, 1, 0])
    return X, Y


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


def test_generators(data_path):
    box, confmaps = load_dataset(data_path)
    image_size = confmaps.shape[-2]
    num_channels_img = box.shape[-1]
    num_channels_confmap = confmaps.shape[-1]
    box = box.reshape([-1, image_size, image_size, num_channels_img])
    confmaps = confmaps.reshape([-1, image_size, image_size, num_channels_confmap])

    # box = box.reshape([-1, 4, image_size, image_size, num_channels_img])
    # confmaps = confmaps.reshape([-1, 4, image_size, image_size, num_channels_confmap])
    #
    # box = box[:, :, :, :, :-1]
    # confmaps = confmaps[:, :, :, :, :7]
    #
    # box_1 = box[:, 0, :, :, :]
    # box_2 = box[:, 1, :, :, :]
    # box_3 = box[:, 2, :, :, :]
    # box_4 = box[:, 3, :, :, :]
    # box = np.concatenate((box_1, box_2, box_3, box_4), axis=-1)
    #
    # confmaps_1 = confmaps[:, 0, :, :, :]
    # confmaps_2 = confmaps[:, 1, :, :, :]
    # confmaps_3 = confmaps[:, 2, :, :, :]
    # confmaps_4 = confmaps[:, 3, :, :, :]
    # confmaps = np.concatenate((confmaps_1, confmaps_2, confmaps_3, confmaps_4), axis=-1)

    # matplotlib.use('TkAgg')
    # example_img = np.transpose(box[0, :, :, [0, 1, 2]], [2, 1, 0])
    # plt.imshow(example_img, cmap='gray')
    # plt.show()

    seed = 0
    batch_size = 8

    data_gen_args = dict(preprocessing_function=custom_augmentations,)

    # data_gen_args = dict(rotation_range=45,
    #                      zoom_range=[0.8, 1.2],
    #                      horizontal_flip=True,
    #                      vertical_flip=True,
    #                      width_shift_range=10,
    #                      height_shift_range=10,
    #                      interpolation_order=2,)

    # data generator
    datagen_x = ImageDataGenerator(**data_gen_args)
    datagen_y = ImageDataGenerator(**data_gen_args)
    # prepare iterator
    datagen_x.fit(box[:10, :, :, :], augment=True, seed=seed)
    datagen_y.fit(confmaps[:10, :, :, :], augment=True, seed=seed)
    flow_box = datagen_x.flow(box, batch_size=batch_size, seed=seed)
    flow_conf = datagen_y.flow(confmaps, batch_size=batch_size, seed=seed)
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
            conf = np.sum(np.squeeze(batch_y[0][:, :, :7]), axis=-1)
            # plot raw pixel data
            plt.imshow(image + conf, cmap='gray')
        # show the figure
        plt.show()
        pass


if __name__ == '__main__':
    data_path = "trainset_random_14_pts_yolo_masks.h5"
    test_generators(data_path)