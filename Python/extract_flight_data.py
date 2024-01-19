import numpy as np


class FlightAnalysis:
    def __init__(self, points_3D_path):
        self.points_3D_path = points_3D_path
        self.points_3D = self.load_points()
        self.num_joints = self.points_3D.shape[1]
        self.num_frames = self.points_3D.shape[0]
        self.num_wings_points = self.num_joints - 2
        self.num_points_per_wing = self.num_wings_points // 2
        self.left_inds = np.arange(0, self.num_points_per_wing)
        self.right_inds = np.arange(self.num_points_per_wing, self.num_wings_points)
        self.wings_pnts_inds = np.array([self.left_inds, self.right_inds])
        self.head_tail_inds = [self.num_wings_points, self.num_wings_points + 1]

    def load_points(self):
        return np.load(self.points_3D_path)

    def extract_wings_planes(self):
        all_4_planes = np.zeros((self.num_frames, 4, 4))
        all_2_planes = np.zeros((self.num_frames, 2, 4))
        all_planes_errors

    @staticmethod
    def fit_plane(points):
        pass





if __name__ == '__main__':
    point_numpy_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\example datasets\movie_14_800_1799_ds_3tc_7tj_WINGS_AND_BODY_SAME_MODEL_Jan 18_06\points_3D.npy"  # get the first argument
    a = FlightAnalysis(point_numpy_path)




