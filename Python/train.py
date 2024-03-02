from constants import *
import json
import preprocessor
import Network
import Augmentor
import os
import shutil
import CallBacks
from datetime import date
from time import time
import sys
from tensorflow.keras.optimizers import Adam
from Network import PointWiseLoss

# train_config.json file of parameters to save all the parameters: in jason format


class Trainer:
    def __init__(self, configuration_path):
        with open(configuration_path) as C:
            config = json.load(C)
            self.config = config
            self.batch_size = config['batch_size']
            self.num_epochs = config['epochs']
            self.batches_per_epoch = config['batches per epoch']
            self.val_fraction = config['val_fraction']
            self.base_output_path = config["base output path"]
            self.viz_idx = 1
            self.model_type = config["model type"]
            self.clean = bool(config["clean"])
            self.debug_mode = bool(config["debug mode"])
            self.wing_rank = config["rank wing size"]
            self.epochs_pointwise_loss = config["epochs pointwise loss"]
            self.preprocessor = preprocessor.Preprocessor(config)

        if self.debug_mode:
            self.batches_per_epoch = 1

        # create the running folders
        custom_per_wing = f"_trained_on_wing_rank_{self.wing_rank}" if self.model_type == PER_WING_1_SIZE_RANK else ""
        self.run_name = f"{self.model_type}_{date.today().strftime('%b %d')}{custom_per_wing}"
        self.run_path = self.create_run_folders()
        self.save_configuration()

        # do preprocessing according to the model type
        self.preprocessor.do_preprocess()
        self.box, self.confmaps = self.preprocessor.get_box(), self.preprocessor.get_confmaps()

        # self.visualize_box_confmaps()

        # get the right cnn architecture
        self.img_size = self.box.shape[1:]
        self.num_output_channels = self.confmaps.shape[-1]
        self.network = Network.Network(config, image_size=self.img_size,
                                       number_of_output_channels=self.num_output_channels)
        self.model = self.network.get_model()

        # split to train and validation
        self.viz_idx = 1
        self.train_box, self.train_confmap, self.val_box, self.val_confmap, _, _ = self.train_val_split()
        self.validation = (self.val_box, self.val_confmap)
        self.viz_sample = (self.val_box[self.viz_idx], self.val_confmap[self.viz_idx])
        print("img_size:", self.img_size)
        print("num_output_channels:", self.num_output_channels)

        # create callback functions
        self.callbacker = CallBacks.CallBacks(config, self.run_path, self.model, self.viz_sample)
        self.callbacks_list = self.callbacker.get_callbacks()
        self.history_callback = self.callbacker.get_history_callback()

        # get augmentations generator
        self.augmentor = Augmentor.Augmentor(config)
        self.train_data_generator = self.augmentor.get_data_generator(self.box, self.confmaps)
        print("creating generators - done!")

    def train(self):

        self.model.save(os.path.join(self.run_path, "initial_model.h5"))
        epoch0 = 0
        t0_train = time()
        training = self.model.fit(
            self.train_data_generator,
            initial_epoch=epoch0,
            epochs=self.num_epochs,
            verbose=1,
            steps_per_epoch=self.batches_per_epoch,
            max_queue_size=512,
            shuffle=False,
            validation_data=self.validation,
            callbacks=self.callbacks_list,
            validation_steps=None
        )
        # Save final model
        self.model.history = self.history_callback.history
        self.model.save(os.path.join(self.run_path, "final_confmaps_model.h5"))
        elapsed_train = time() - t0_train
        print("Total runtime first loss: %.1f mins" % (elapsed_train / 60))

    def save_configuration(self):
        with open(f"{self.run_path}/configuration.json", 'w') as file:
            json.dump(self.config, file, indent=4)

    def train_val_split(self, shuffle=True):
        """ Splits datasets into train and validation sets. """

        val_size = int(np.round(len(self.box) * self.val_fraction))
        idx = np.arange(len(self.box))
        if shuffle:
            np.random.shuffle(idx)

        val_idx = idx[:val_size]
        idx = idx[val_size:]

        return self.box[idx], self.confmaps[idx], self.box[val_idx], self.confmaps[val_idx], idx, val_idx

    def create_run_folders(self):
        """ Creates subfolders necessary for outputs of vision. """
        run_path = os.path.join(self.base_output_path, self.run_name)

        if not self.clean:
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

    def visualize_box_confmaps(self):
        """ visualize the input to the network """
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('TkAgg')

        num_images = self.box.shape[0]
        for image in range(num_images):
            print(image)
            if self.model_type == HEAD_TAIL_ALL_CAMS:
                fig, axs = plt.subplots(2, 2)
                axs = axs.flatten()
                flies = self.box[image, ...]
                confmaps = self.confmaps[image, ...]
                for i, ax in enumerate(axs):
                    fly_inds = np.array([0, 1, 2]) + i*3
                    confmaps_inds = np.array([0, 1]) + i*2
                    image = flies[..., fly_inds]
                    image[..., 1] += np.sum(confmaps[..., confmaps_inds], axis=-1)
                    ax.imshow(image)
                plt.show()
                continue
            fig = plt.figure(figsize=(8, 8))
            fly = self.box[image, :, :, 1]
            masks = np.zeros((192, 192))
            try:
                confmap = np.sum(self.confmaps[image, :, :, :], axis=2)
            except:
                a = 0
            if self.model_type == ALL_POINTS_MODEL:
                masks = np.sum(self.box[image, :, :, [-2, -1]], axis=0)
            else:
                try:
                    masks = self.box[image, :, :, 3]
                except:
                    a = 0

            img = np.zeros((192, 192, 3))
            img[:, :, 1] = fly
            try:
                img[:, :, 2] = confmap
            except:
                a = 0
            img[:, :, 0] = masks

            plt.imshow(img)
            plt.show()


if __name__ == '__main__':
    config_path = sys.argv[1]  # get the first argument
    print(config_path)
    trainer = Trainer(config_path)
    trainer.train()


