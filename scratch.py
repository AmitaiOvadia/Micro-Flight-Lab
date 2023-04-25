import h5py
import numpy as np

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


def create_wings_syncronized_dataset():
    global path, data, f
    import scipy.io
    box_path_movie_1 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 1\movie_1_1701_2200_500_frames_3tc_7tj_no_masks.h5"
    path = r"movie_1_box.mat"
    data = scipy.io.loadmat(f"{path}")
    new_box = data["box"].T
    prev_box = h5py.File(box_path_movie_1, "r")["box"][:]
    cam1 = np.transpose(prev_box[:, 0:3, :, :], [0, 2, 3, 1])
    cam2 = np.transpose(prev_box[:, 3:6, :, :], [0, 2, 3, 1])
    cam3 = np.transpose(prev_box[:, 6:9, :, :], [0, 2, 3, 1])
    cam4 = np.transpose(prev_box[:, 9:12, :, :], [0, 2, 3, 1])
    new_cam1 = np.zeros((500, 192, 192, 5))
    new_cam1[:, :, :, 0:3] = cam1
    new_cam2 = np.zeros((500, 192, 192, 5))
    new_cam2[:, :, :, 0:3] = cam2
    new_cam3 = np.zeros((500, 192, 192, 5))
    new_cam3[:, :, :, 0:3] = cam3
    new_cam4 = np.zeros((500, 192, 192, 5))
    new_cam4[:, :, :, 0:3] = cam4
    new_cam1[:, :, :, 3:5] = np.transpose(new_box[:, 0, :, :, :], [0, 2, 3, 1])[:, :, :, 1:]
    new_cam2[:, :, :, 3:5] = np.transpose(new_box[:, 1, :, :, :], [0, 2, 3, 1])[:, :, :, 1:]
    new_cam3[:, :, :, 3:5] = np.transpose(new_box[:, 2, :, :, :], [0, 2, 3, 1])[:, :, :, 1:]
    new_cam4[:, :, :, 3:5] = np.transpose(new_box[:, 3, :, :, :], [0, 2, 3, 1])[:, :, :, 1:]
    box_to_save = np.concatenate([new_cam1, new_cam2, new_cam3, new_cam4], axis=-1)
    with h5py.File("box_to_save_movie_1.h5", "w") as f:
        ds_pos = f.create_dataset("box", data=box_to_save, compression="gzip",
                                  compression_opts=1)
def get_masks(img_3_ch, model):
    net_input = img_3_ch
    masks_2 = np.zeros((2, 192, 192))
    if np.max(img_3_ch) <= 1:
        net_input = np.round(255 * img_3_ch)
    results = model(net_input)[0]

    # find if the masks detected are overlapping
    boxes = results.boxes.boxes.numpy()

    masks_found = results.masks.masks.numpy()[[0, 1], :, :]
    # add masks
    for wing in range(2):
        mask = masks_found[wing, :, :]
        score = results.boxes.boxes[wing, 4]
        masks_2[wing, :, :] = mask
        # else:
        # print(f"score = {score}")
        # matplotlib.use('TkAgg')
        # img_3_ch[:, :, 2] += mask
        # plt.imshow(img_3_ch)
        # plt.show()
    return masks_2

def use_tracker_to_add_wings():
    from ultralytics import YOLO
    import h5py
    wings_detection_model_path = "wings_detection_yolov8_weights_13_3.pt"
    box_path_no_masks = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 17\movie_17_1401_2000_ds_5tc_14tj.h5"
    box = h5py.File(box_path_no_masks, "r")["/box"][:]
    model = YOLO(wings_detection_model_path)
    model.fuse()
    # Run object detection on saved video
    results = model.predict(source='video.gif', save=True, save_txt=True, tracker='botsort.yaml')
    pass

def save_as_video():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.cm as cm
    import h5py
    matplotlib.use('TkAgg')
    wings_detection_model_path = "wings_detection_yolov8_weights_13_3.pt"
    box_path_no_masks = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 14\dataset_movie_14_frames_1301_2300_ds_3tc_7tj.h5"
    box = h5py.File(box_path_no_masks, "r")["/box"][:]

    img = []  # some array of images
    frames = []  # for storing the generated images
    fig = plt.figure()
    for frame in range(box.shape[0]):
        print(f"saving frame {frame + 1}")
        frames.append([plt.imshow(box[frame, 3 + np.array([0, 1, 2]), :, :].T, cmap=cm.Greys_r, animated=True)])
    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    ani.save('video.gif')
    plt.show()




if __name__ == "__main__":
    use_tracker_to_add_wings()
    # save_as_video()

