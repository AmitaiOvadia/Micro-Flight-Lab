from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LambdaCallback, Callback
from scipy.io import loadmat, savemat
from viz import show_pred, show_confmap_grid, plot_history
import os


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


class CallBacks:
    def __init__(self, config, run_path, model, viz_sample):
        self.model = model
        self.history_callback = LossHistory(run_path=run_path)
        self.reduce_lr_factor = config["reduce_lr_factor"],
        self.reduce_lr_patience = config["reduce_lr_patience"],
        self.reduce_lr_min_delta = config["reduce_lr_min_delta"],
        self.reduce_lr_cooldown = config["reduce_lr_cooldown"],
        self.reduce_lr_min_lr = config["reduce_lr_min_lr"],
        self.save_every_epoch = bool(config["save_every_epoch"])

        self.reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=self.reduce_lr_factor[0],
                                                    patience=self.reduce_lr_patience[0], verbose=1, mode="auto",
                                                    epsilon=self.reduce_lr_min_delta[0], cooldown=self.reduce_lr_cooldown[0],
                                                    min_lr=self.reduce_lr_min_lr[0])
        if self.save_every_epoch:
            self.checkpointer = ModelCheckpoint(
                filepath=os.path.join(run_path, "weights/weights.{epoch:03d}-{val_loss:.9f}.h5"),
                verbose=1, save_best_only=False)
        else:
            self.checkpointer = ModelCheckpoint(filepath=os.path.join(run_path, "best_model.h5"), verbose=1,
                                                save_best_only=True)

        self.viz_grid_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: show_confmap_grid(self.model, *viz_sample, plot=True,
                                                               save_path=os.path.join(
                                                                   run_path,
                                                                   "viz_confmaps\confmaps_%03d.png" % epoch),
                                                               show_figure=False))
        self.viz_pred_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: show_pred(self.model, *viz_sample,
                                                                                           save_path=os.path.join(
                                                                                               run_path,
                                                                                               "viz_pred\pred_%03d.png" % epoch),
                                                                                           show_figure=False))

    def get_history_callback(self):
        return self.history_callback

    def get_callbacks(self):
        return [self.reduce_lr_callback,
                self.checkpointer,
                self.history_callback,
                self.viz_pred_callback]
