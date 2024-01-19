import h5py
from ultralytics import YOLO
import numpy as np
from scipy.spatial.distance import cdist

NUM_CAMS = 4


class Add_Masks:
    def __init__(self, wings_detection_model_path,
                 box_path):
        self.wings_detection_model_path = wings_detection_model_path
        self.box_path = box_path
        self.box = self.get_box()
        self.wings_detection_model = self.get_wings_detection_model()
        self.cropzone = self.get_cropzone()
        self.num_frames, self.num_times_channels, self.im_size, _ = self.box.shape
        self.num_cams = NUM_CAMS
        self.num_times_channels //= self.num_cams
        self.masks = np.zeros((self.num_frames, self.num_cams, self.im_size, self.im_size, 2))
        self.scores = np.zeros((self.num_frames, self.num_cams, 2))
        self.boxes = np.zeros((self.num_frames, self.num_cams, 4))

    def detect_masks_and_save(self):
        self.run_masks_detection()
        self.save_masks_to_h5()

    def get_box(self):
        return h5py.File(self.box_path, "a")["/box"][:]

    def get_cropzone(self):
        return h5py.File(self.box_path, "a")["/cropzone"]

    def get_wings_detection_model(self):
        """ load a pretrained YOLOv8 segmentation model"""
        model = YOLO(self.wings_detection_model_path)
        model.fuse()
        return model

    def save_masks_to_h5(self):
        f = h5py.File(self.box_path, "a")

        masks_dset = f.create_dataset("train_masks", data=self.masks.astype("int32"),  compression="gzip",
                                      compression_opts=1)
        masks_dset.attrs["description"] = "train_masks for each wing in each camera, order (left right) still unfixed, " \
                                          "image of zeros for no mask"
        masks_dset.attrs["dims"] = "(num_frames, num_cams, im_size, im_size, 2)"

        masks_dset = f.create_dataset("scores", data=self.scores, compression="gzip",
                                      compression_opts=1)
        masks_dset.attrs["description"] = "a score from 0->1 for each mask (left right according to train_masks order)"
        masks_dset.attrs["dims"] = f"{self.scores.shape}"

        masks_dset = f.create_dataset("boxes", data=self.boxes, compression="gzip",
                                      compression_opts=1)
        masks_dset.attrs["description"] = "The bounding box that around each mask"
        masks_dset.attrs["dims"] = f"{self.boxes.shape}"

        f.close()

    def run_masks_detection(self):
        """ add train_masks to the self.train_masks array """
        if self.num_times_channels == 3:
            for cam in range(self.num_cams):
                print(f"finds wings for camera number {cam + 1}")
                img_3_ch_all = self.box[:, np.array([0, 1, 2]) + self.num_times_channels * cam, :, :]
                img_3_ch_all = np.transpose(img_3_ch_all, [0, 2, 3, 1])
                img_3_ch_input = np.round(img_3_ch_all * 255)
                img_3_ch_input = [img_3_ch_input[i] for i in range(self.num_frames)]
                results = self.wings_detection_model(img_3_ch_input)
                for frame in range(self.num_frames):
                    masks_2 = np.zeros((self.im_size, self.im_size, 2))
                    result = results[frame]
                    boxes = result.boxes.data.numpy()
                    inds_to_keep = self.eliminate_close_vectors(boxes, 10)
                    num_wings_found = np.count_nonzero(inds_to_keep)
                    if num_wings_found > 0:
                        masks_found = result.masks.data.numpy()[inds_to_keep, :, :]
                    for wing in range(min(num_wings_found, 2)):
                        box = boxes[wing, :4]
                        self.boxes[frame, cam, :] = box
                        mask = masks_found[wing, :, :]
                        score = boxes[wing, 4]
                        self.scores[frame, cam, wing] = score
                        masks_2[:, :, wing] = mask
                    self.masks[frame, cam, :, :, :] = masks_2

    @staticmethod
    def eliminate_close_vectors(matrix, threshold):
        # calculate pairwise Euclidean distances
        distances = cdist(matrix, matrix, 'euclidean')
        # create a mask to identify which vectors to keep
        inds_to_del = np.ones(len(matrix), dtype=bool)
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                if distances[i, j] < threshold:
                    # eliminate one of the vectors
                    inds_to_del[j] = False

        # return the new matrix with close vectors eliminated
        return inds_to_del


if __name__ == "__main__":
    box_path = r"C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\segmentation 3D\example\movie_8_2001_2500_ds_3tc_7tj.h5"
    model_path = "wings_segmentation/YOLO models/wings_detection_yolov8_weights_13_3.pt"
    add_masks_obj = Add_Masks(model_path, box_path).detect_masks_and_save()
