import os
from time import time
import shutil

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LambdaCallback, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from scipy.io import loadmat, savemat

from viz import show_pred, show_confmap_grid, plot_history

from pose_estimation_models import basic_nn



def preprocess(X, permute=(0, 3, 2, 1)):
    """ Normalizes input data. """

    # Add singleton dim for single images
    if X.ndim == 3:
        X = X[None, ...]

    # Adjust dimensions
    X = np.transpose(X, permute)

    # Normalize
    if X.dtype == "uint8":
        X = X.astype("float32") / 255

    return X


def load_dataset(data_path, X_dset="box", Y_dset="confmaps", permute=(0, 3, 2, 1)):
    """ Loads and normalizes datasets. """

    # Load
    with h5py.File(data_path, "r") as f:
        X = f[X_dset][:]
        Y = f[Y_dset][:]

    # Adjust dimensions
    # x_permute = (0, 3, 2, 1)
    # y_permute = (0, 3, 4, 1, 2)

    X = preprocess(X, permute)
    Y = preprocess(Y, permute)
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


def augmented_data_generator(batch_size, box, confmap, seed):
    # we create two instances with the same arguments
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
    train_generator = zip(flow_box, flow_conf)
    return train_generator



def train(data_path, *,
          base_output_path="models",
          run_name=None,
          data_name=None,
          net_name="leap_cnn",
          clean=False,
          box_dset="box",
          confmap_dset="confmaps",
          val_size=0.15,
          preshuffle=True,
          filters=64,
          rotate_angle=15,
          epochs=50,
          batch_size=32,
          batches_per_epoch=50,
          validation_steps=10,
          viz_idx=0,
          reduce_lr_factor=0.1,
          reduce_lr_patience=3,
          reduce_lr_min_delta=1e-5,
          reduce_lr_cooldown=0,
          reduce_lr_min_lr=1e-10,
          save_every_epoch=False,
          seed=0,
          amsgrad=False,
          upsampling_layers=False,
          ):
    # why 184?
    box, confmap = load_dataset(data_path, X_dset=box_dset, Y_dset=confmap_dset)
    train_box, train_confmap, val_box, val_confmap, train_idx, val_idx = train_val_split(box, confmap,
                                                                                         val_size=val_size,
                                                                                         shuffle=preshuffle)
    viz_sample = (val_box[viz_idx], val_confmap[viz_idx])

    # Pull out metadata
    img_size = box.shape[1:]
    num_output_channels = confmap.shape[-1]
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

    viz_grid_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: show_confmap_grid(model, *viz_sample, plot=True,
                                                                                          save_path=os.path.join(
                                                                                              run_path,
                                                                                              "viz_confmaps/confmaps_%03d.png" % epoch),
                                                                                          show_figure=False))
    viz_pred_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: show_pred(model, *viz_sample,
                                                                                  save_path=os.path.join(run_path,
                                                                                                         "viz_pred/pred_%03d.png" % epoch),
                                                                                  show_figure=False))

    # get model
    # filters=64, num_blocks=2 was good
    model = basic_nn(img_size, num_output_channels, filters=64, num_blocks=2, kernel_size=3)

    # Save initial network
    model.save(os.path.join(run_path, "initial_model.h5"))

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
        #use_multiprocessing=True,
        #workers=8,
        steps_per_epoch=batches_per_epoch,
        max_queue_size=512,
        shuffle=False,
        validation_data=val_datagen,
        callbacks=[
            reduce_lr_callback,
            checkpointer,
            history_callback,
            viz_grid_callback,
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


if __name__ == '__main__':
    data_path = r"C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\predictions\experiment\test_5_channels_3_times_2_masks\training_set_5_channels_3times_2masks.h5"
    # data_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\datasets\amitai_dataset3_ds_3tc_7tj.h5"
    train(data_path, run_name="train_model3",
          epochs=30,
          batch_size=3,
          batches_per_epoch=3,
          validation_steps=10, )
