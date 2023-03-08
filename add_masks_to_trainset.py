import h5py
import numpy as np
from ultralytics import YOLO




def load_dataset(data_path, X_dset="box", Y_dset="confmaps", permute=(0, 3, 2, 1)):
    """ Loads and normalizes datasets. """
    # Load
    with h5py.File(data_path, "r") as f:
        X = f[X_dset][:]
        Y = f[Y_dset][:]
    if X.shape[0] != 2:
        X = np.transpose(X, [5, 4, 3, 2, 1, 0])
        Y = np.transpose(Y, [5, 4, 3, 2, 1, 0])
    return X, Y


def save_new_box(box, path):
    with h5py.File(path, "a") as f:
        del f["box"]
        ds_conf = f.create_dataset("box", data=box, compression="gzip", compression_opts=1)
        ds_conf.attrs["description"] = "box"
        ds_conf.attrs["dims"] = f"{box.shape}"


def get_masks(wings_detection_model, img_3_ch):
    net_input = img_3_ch
    if np.max(img_3_ch) <= 1:
        net_input = np.round(255 * img_3_ch)
    results = wings_detection_model(net_input)[0]
    masks = results.masks.masks.numpy()[:2, :, :]
    return masks


def add_masks(box, wings_detection_model_path):
    model = YOLO(wings_detection_model_path)
    model.fuse()
    box_masks = np.zeros(box.shape[:-1] + (box.shape[-1] + 2,))
    mir_num = box.shape[0]
    num_frames = box.shape[1]
    num_cams = box.shape[2]
    for mir in range(mir_num):
        for frame in range(num_frames):
            print(frame)
            for cam in range(num_cams):
                img_3_ch = box[mir, frame, cam, :, :, :]
                masks = get_masks(model, img_3_ch)
                masks = np.transpose(masks, [1, 2, 0])
                img_5_ch = np.zeros(img_3_ch.shape[:-1] + (img_3_ch.shape[-1] + 2,))
                img_5_ch[:, :, [0, 1, 2]] = img_3_ch
                img_5_ch[:, :, [3, 4]] = masks
                box_masks[mir, frame, cam, :, :, :] = img_5_ch
    return box_masks


def add_masks_to_trainset(trainset_path, yolo_model_path):
    box, confmaps = load_dataset(trainset_path)
    box = add_masks(box, yolo_model_path)
    save_new_box(box, trainset_path)


if __name__ == '__main__':

    movie_trainset_path = r"train_set_movie_14_pts_yolo_masks.h5"

    random_trainset_path = r"trainset_random_14_pts_200_frames.h5"

    yolo_model_path = "wings_detection_yolov8_weights_4_3.pt"
    add_masks_to_trainset(random_trainset_path, yolo_model_path)

