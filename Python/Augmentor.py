import numpy as np
from scipy.ndimage import shift
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Augmentor:
    def __init__(self,
                 config):
        self.use_custom_function = bool(config["custom"])
        self.xy_shifts = config["augmentation shift x y"]
        self.rotation_range = config["rotation range"]
        self.seed = config["seed"]
        self.batch_size = config["batch_size"]
        self.zoom_range = config["zoom range"]
        self.interpolation_order = config["interpolation order"]
        self.do_horizontal_flip = bool(config["horizontal flip"])
        self.do_vertical_flip = bool(config["vertical flip"])
        self.custom_augmentation_function = self.get_custom_augmentation_function()
        self.debug_mode = bool(config["debug mode"])

        if self.debug_mode:
            self.batch_size = 1

    def get_data_generator(self, box, confmaps):
        datagen = self.config_data_generator(box, confmaps)
        return datagen

    def config_data_generator(self, box, confmaps):
        if self.use_custom_function:
            data_gen_args = dict(preprocessing_function=self.custom_augmentation_function)
        else:
            data_gen_args = dict(rotation_range=self.rotation_range,
                                 zoom_range=self.zoom_range,
                                 horizontal_flip=self.do_horizontal_flip,
                                 vertical_flip=self.do_vertical_flip,
                                 width_shift_range=self.xy_shifts,
                                 height_shift_range=self.xy_shifts,
                                 interpolation_order=self.interpolation_order,)

        datagen_x = ImageDataGenerator(**data_gen_args)
        datagen_y = ImageDataGenerator(**data_gen_args)
        # prepare iterator
        datagen_x.fit(box, augment=True, seed=self.seed)
        datagen_y.fit(confmaps, augment=True, seed=self.seed)
        flow_box = datagen_x.flow(box, batch_size=self.batch_size, seed=self.seed, shuffle=False)
        flow_conf = datagen_y.flow(confmaps, batch_size=self.batch_size, seed=self.seed, shuffle=False)
        train_generator = zip(flow_box, flow_conf)
        return train_generator

    @staticmethod
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
            img = img / 255
        return img

    def get_custom_augmentation_function(self):
        rotation_range = self.rotation_range
        xy_shift = self.xy_shifts
        can_horizontal_flip = self.do_horizontal_flip
        can_vertical_flip = self.do_vertical_flip

        def custom_augmentations(img):
            """get an image of shape (height, width, num_channels) and return augmented image"""
            # if img.shape[-1] == 4:
            #     import matplotlib.pyplot as plt
            #     import matplotlib
            #     matplotlib.use('TkAgg')
            #     plt.imshow(img[:,:,1] + img[:,:,3])
            #     plt.show()
            do_horizontal_flip = bool(np.random.randint(2)) and can_horizontal_flip
            do_vertical_flip = bool(np.random.randint(2)) and can_vertical_flip
            rotation_angle = np.random.randint(-rotation_range, rotation_range)
            shift_y_x = np.random.randint(-xy_shift, xy_shift, 2)
            num_channels = img.shape[-1]
            for channel in range(num_channels):
                img[:, :, channel] = Augmentor.augment(img[:, :, channel], do_horizontal_flip,
                                                       do_vertical_flip, rotation_angle, shift_y_x)
            return img
        return custom_augmentations

