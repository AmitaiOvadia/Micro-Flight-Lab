import scipy.io
import numpy as np
from traingulate import Triangulate
import json
import h5py
from matplotlib import pyplot as plt
from visualize import Visualizer
from extract_flight_data import FlightAnalysis
from predict_2D_sparse_box import Predictor2D
import os

reprojected_points_2D_path = (r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled "
                              r"dataset\estimated_positions.mat")
ground_truth_2D_path = (r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled "
                        r"dataset\ground_truth_labels.mat")
configuration_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled dataset\2D_to_3D_config.json"
h5 = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled dataset\trainset_movie_1_370_520_ds_3tc_7tj.h5"
save_directory = (r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D "
                  r"code\validation results")

CALIBRATION_ERROR = "CALIBRATION_ERROR"
DETECTION_ERROR = "DETECTION_ERROR"
CALIBRATION_REPROJECTION_ERROR = 0.5


class EstimateAnalysisErrors:
    def __init__(self, task=DETECTION_ERROR, num_samples=10):
        self.all_analysis_objects_smoothed = None
        self.all_analysis_objects = None
        self.task_name = task
        self.num_samples = num_samples
        # Load the .mat file
        self.ground_truth_2D = EstimateAnalysisErrors.load_from_mat(ground_truth_2D_path, name='ground_truth')
        self.num_frames = len(self.ground_truth_2D)
        self.num_joints = self.ground_truth_2D.shape[2]
        self.reprojected_points_2D = EstimateAnalysisErrors.load_from_mat(reprojected_points_2D_path, name='positions',
                                                                          end=self.num_frames)
        self.cropzone = h5py.File(h5, 'r')['/cropzone'][:self.num_frames]
        with open(configuration_path) as C:
            config = json.load(C)
            self.triangulate = Triangulate(config)
        self.ground_truth_3D = self.get_3D_points(self.ground_truth_2D, self.cropzone)
        self.ground_truth_3D_smoothed = Predictor2D.smooth_3D_points(self.ground_truth_3D)

        self.ground_truth_analysis = FlightAnalysis(validation=True, points_3D=self.ground_truth_3D)
        self.ground_truth_analysis_smoothed = FlightAnalysis(validation=True, points_3D=self.ground_truth_3D_smoothed)

        self.first_frame = self.ground_truth_analysis_smoothed.first_y_body_frame
        self.last_frame = self.ground_truth_analysis_smoothed.end_frame

        if task == DETECTION_ERROR:
            self.detection_error_analysis_dist_distribution()
            # self.detection_error_analysis_gaussian()
        else:  # task == CALIBRATION_ERROR
            self.calibration_error_analysis()

    def detection_error_analysis_dist_distribution(self):
        estimated_3D = self.get_3D_points(self.reprojected_points_2D, self.cropzone)
        print("started sampling")
        all_flies = self.sample_n_flies_from_dist_distribution(self.ground_truth_3D_smoothed,
                                                               estimated_3D,
                                                               n=self.num_samples)
        print("started smoothing")
        all_flies_smoothed = [Predictor2D.smooth_3D_points(fly) for fly in all_flies]
        # all_flies_smoothed = []
        print("analyzing not smoothed")
        self.all_analysis_objects = EstimateAnalysisErrors.get_all_analysis(all_flies)
        print("analyzing smoothed")
        self.all_analysis_objects_smoothed = EstimateAnalysisErrors.get_all_analysis(all_flies_smoothed)
        print("saving")
        self.save_attributes_validations()

    def sample_n_flies_from_dist_distribution(self, ground_truth_3D, estimated_3D, n, nbins=20):
        distances = np.linalg.norm(ground_truth_3D - estimated_3D, axis=-1)
        head_hist, head_edges = np.histogram(distances[:, -1], bins=nbins)
        tail_hist, tail_edges = np.histogram(distances[:, -2], bins=nbins)
        histograms = self.num_joints * [0]
        histograms_bins = self.num_joints * [0]
        original_dists = self.num_joints * [0]
        histograms[-1], histograms[-2] = head_hist, tail_hist
        histograms_bins[-1], histograms_bins[-2] = head_edges, tail_edges
        original_dists[-1], original_dists[-2] = (np.concatenate((distances[:, -1], distances[:, -1])) ,
                                                  np.concatenate((distances[:, -2], distances[:, -2])))
        num_joints_per_wing = (distances.shape[1] - 2)//2
        for i in range(num_joints_per_wing):
            joint_dists = np.concatenate((distances[:, i], distances[:, i + num_joints_per_wing]))
            hist, edges = np.histogram(joint_dists, bins=nbins)
            histograms[i], histograms[i + num_joints_per_wing] = hist, hist
            histograms_bins[i], histograms_bins[i + num_joints_per_wing] = edges, edges
            original_dists[i], original_dists[i + num_joints_per_wing] = joint_dists, joint_dists

        all_flies = EstimateAnalysisErrors.sample_points_around_base(base_points=ground_truth_3D,
                                                                     histograms=histograms,
                                                                     bins=histograms_bins,
                                                                     N=n,
                                                                     original_dists=np.array(original_dists))
        return all_flies

    def calibration_error_analysis(self):
        # sample multiple flies in 2D
        sigma = CALIBRATION_REPROJECTION_ERROR * np.ones((self.num_joints,))
        all_flies_2D, _ = self.sample_n_flies_gaussian(self.ground_truth_2D, sigma=sigma, n=self.num_samples,
                                                       smooth=False)
        # triangulate to 3D
        all_flies = [self.get_3D_points(fly, self.cropzone) for fly in all_flies_2D]
        # smooth
        all_flies_smoothed = [Predictor2D.smooth_3D_points(fly) for fly in all_flies]
        # run the same tests
        self.all_analysis_objects = EstimateAnalysisErrors.get_all_analysis(all_flies)
        print("started analysing smoothed")
        self.all_analysis_objects_smoothed = EstimateAnalysisErrors.get_all_analysis(all_flies_smoothed)
        self.save_attributes_validations()

    @staticmethod
    def sample_points_around_base(base_points, histograms, bins, N, original_dists):
        """
        Sample N sets of points for each base point in base_points using D histograms.
        base_points: (M, D, 3) array of base points.
        histograms: List of D histograms for each dimension.
        bins: List of D arrays, each containing bin edges for the corresponding histogram.
        N: Number of sets of points to sample.

        Returns: A list of N arrays, each of shape (M, D, 3) containing sampled points.
        """
        M, D, _ = base_points.shape
        sampled_sets = []

        for _ in range(N):
            # Step 1: Sample distances using histograms for each dimension
            distances = EstimateAnalysisErrors.sample_from_histogram_multi(histograms, bins, size=(M, D))

            # Step 2: Sample directions on a unit sphere
            directions = EstimateAnalysisErrors.sample_uniform_sphere((M, D))

            # Step 3: Scale the directions by the sampled distances
            sampled_noise = directions * distances[..., np.newaxis]  # Shape: (M, D, 3)

            # Step 4: Translate the sampled points to the base points
            sampled_noise += base_points

            # Append the sampled set to the list
            sampled_sets.append(sampled_noise)

        return sampled_sets

    @staticmethod
    def sample_from_histogram_multi(histograms, bins, size):
        """
        Sample distances based on D histograms.
        histograms: List of length D, each element is a histogram array.
        bins: List of length D, each element is a bin edge array corresponding to the histograms.
        size: Tuple (M, D) indicating how many samples to take for each dimension.
        D: Number of dimensions.
        """
        M, D = size
        distances = np.zeros((M, D))

        # For each dimension, sample distances according to its histogram
        for d in range(D):
            histogram = histograms[d]

            # Normalize the histogram so the sum equals 100
            normalized_histogram = (histogram / np.sum(histogram)) * M

            # Create a list of samples for this dimension
            samples_d = []

            # For each bin, sample points uniformly between its edges
            for bin_idx in range(len(histogram)):
                bin_count = int(np.round(normalized_histogram[bin_idx]))  # Get the number of samples for this bin
                lower_edge = bins[d][bin_idx]
                upper_edge = bins[d][bin_idx + 1]
                if bin_count > 0:
                    # Sample uniformly within the bin's range
                    samples_bin = np.random.uniform(lower_edge, upper_edge, bin_count)
                    samples_d.extend(samples_bin)

            # Ensure that we have exactly M samples for this dimension
            samples_d = np.array(samples_d)
            if len(samples_d) > M:
                samples_d = np.random.choice(samples_d, M, replace=False)  # Randomly reduce samples to M
            elif len(samples_d) < M:
                additional_samples = np.random.choice(samples_d, M - len(samples_d),
                                                      replace=True)  # Add more samples if needed
                samples_d = np.concatenate([samples_d, additional_samples])
            np.random.shuffle(samples_d)
            distances[:, d] = samples_d
        return distances

    @staticmethod
    def sample_uniform_sphere(size):
        """
        Sample 'size' points uniformly on a 3D sphere.
        size: (M, D) where M is the number of base points, D is the number of dimensions.
        Returns: (M, D, 3) array of unit vectors.
        """
        M, D = size
        phi = np.random.uniform(0, 2 * np.pi, size=(M, D))
        cos_theta = np.random.uniform(-1, 1, size=(M, D))
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        x = sin_theta * np.cos(phi)
        y = sin_theta * np.sin(phi)
        z = cos_theta
        points_on_sphere = np.stack([x, y, z], axis=-1)
        # fig = plt.figure(figsize=(20, 20))  # Adjust the figure size here
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(points_on_sphere[..., 0], points_on_sphere[..., 1], points_on_sphere[..., 2])
        # ax.set_aspect('equal')
        # plt.show()
        return points_on_sphere  # Shape: (M, D, 3)

    def detection_error_analysis_gaussian(self):
        estimated_3D = self.get_3D_points(self.reprojected_points_2D, self.cropzone)
        distance_per_joint = self.get_distance_per_joint(self.ground_truth_3D, estimated_3D)
        print("started sampling")
        # self.all_flies, self.all_flies_smoothed = self.sample_n_flies(n=500, smooth=True)
        all_flies, all_flies_smoothed = self.sample_n_flies_gaussian(mean_points=self.ground_truth_3D,
                                                            sigma=distance_per_joint,
                                                            n=self.num_samples,
                                                            smooth=True)
        print("finished sampling")
        print("started analysing not smoothed")
        self.all_analysis_objects = EstimateAnalysisErrors.get_all_analysis(all_flies)
        print("started analysing smoothed")
        self.all_analysis_objects_smoothed = EstimateAnalysisErrors.get_all_analysis(all_flies_smoothed)
        self.save_attributes_validations()

    def save_attributes_validations(self):
        self.get_uncertainty(attribute=['yaw_angle'])
        self.get_uncertainty(attribute=['pitch_angle'])
        self.get_uncertainty(attribute=['roll_angle'])
        self.get_uncertainty(attribute=['wings_phi_left', 'wings_phi_right'])
        self.get_uncertainty(attribute=['wings_psi_left', 'wings_psi_right'])
        self.get_uncertainty(attribute=['wings_theta_left', 'wings_theta_right'])
        self.get_uncertainty(attribute=['omega_body'])

    def get_uncertainty(self, attribute=None):
        if attribute is None:
            attribute = []

        # Initialize variables for smoothed and non-smoothed cases
        data = {"smoothed": [], "not_smoothed": []}
        stats = {}

        for smoothed in [True, False]:
            if smoothed:
                ground_truth_analysis = self.ground_truth_analysis_smoothed
                all_analysis_objects = self.all_analysis_objects_smoothed
                label = "smoothed"
            else:
                ground_truth_analysis = self.ground_truth_analysis
                all_analysis_objects = self.all_analysis_objects
                label = "not_smoothed"

            all_variables = np.empty((0,))
            for attr in attribute:
                ground_truth_attribute = getattr(ground_truth_analysis, attr)
                ground_truth_attribute = ground_truth_attribute[self.first_frame:self.last_frame]  # take only inside wingbit
                for analysis in all_analysis_objects:
                    variable = getattr(analysis, attr)
                    variable = variable[self.first_frame:self.last_frame]
                    if len(variable) != len(ground_truth_attribute):
                        continue
                    delta = ground_truth_attribute - variable
                    not_nan_delta = delta[~np.isnan(delta)]
                    all_variables = np.append(all_variables, not_nan_delta)

            # Store results for both smoothed and non-smoothed
            all_variables = np.array(all_variables).ravel()
            # all_variables = EstimateAnalysisErrors.remove_outliers_mad(all_variables, threshold=16)
            data[label] = all_variables
            stats[label] = {"std": np.std(all_variables), "mean": np.mean(all_variables)}

            # Find the common bin edges for both histograms
        all_data = np.concatenate([data["smoothed"], data["not_smoothed"]])
        bins = np.histogram_bin_edges(all_data, bins=100)

        # Plot both histograms on the same graph with the same bins and normalized
        plt.figure(dpi=600)
        plt.hist(data["not_smoothed"], bins=bins, alpha=0.5, density=True,
                 label=f"Not Smoothed",
                 color='red')

        plt.hist(data["smoothed"], bins=bins, alpha=0.5, density=True,
                 label=f"Smoothed",
                 color='blue')

        # Add title with stats for both cases
        title = (f"Task: {self.task_name}\nAttribute: {attribute}\n"
                 f"Smoothed: std {stats['smoothed']['std']:.3f}, mean {stats['smoothed']['mean']:.3f} "
                 f"\nNot Smoothed: std {stats['not_smoothed']['std']:.3f}, mean {stats['not_smoothed']['mean']:.3f}\n"
                 f"Number of sampled movies: {len(self.all_analysis_objects_smoothed)}")

        plt.title(title)
        plt.legend(loc='upper right')
        plt.tight_layout()

        # Save figure
        save_path = os.path.join(save_directory, f"{attribute} smoothed vs not smoothed {self.task_name}.png")
        plt.savefig(save_path)
        plt.close()
        # plt.show()

    @staticmethod
    def get_all_analysis(all_flies):
        all_analysis = []
        for fly in all_flies:
            analysis = FlightAnalysis(validation=True, points_3D=fly)
            all_analysis.append(analysis)
        return all_analysis

    @staticmethod
    def remove_outliers_mad(data, threshold=5):
        # Calculate the median of the data
        median = np.median(data)
        abs_deviation = np.abs(data - median)
        mad = np.median(abs_deviation)
        if mad == 0:
            return data
        upper_limit = median + threshold * mad
        lower_limit = median - threshold * mad
        filtered_data = data[(data >= lower_limit) & (data <= upper_limit)]
        if len(filtered_data) < len(data):
            print(f"outliers detected, {len(data)} to {len(filtered_data)}")
        return filtered_data

    def sample_n_flies_gaussian(self, mean_points, sigma, n=100, smooth=True):
        all_flies = []
        all_flies_smoothed = []
        mean_shape = mean_points.shape  # This could be (N, 18, 3) or (N, 4, 18, 2)
        sigma_shape = [1] * (len(mean_shape) - 2) + [sigma.shape[0], 1]
        sigma_matrix_nd = sigma.reshape(sigma_shape)  # Broadcasting shape for sigma
        sigma_matrix_nd = np.repeat(sigma_matrix_nd, mean_shape[-1], axis=-1)  # Match the last dimension of mean_points
        for i in range(n):
            random_samples = np.random.randn(*mean_shape)  # Match the shape of mean_points
            new_fly = mean_points + random_samples * sigma_matrix_nd  # Broadcasting should work automatically
            all_flies.append(new_fly)
        if smooth and self.task_name == DETECTION_ERROR:
            all_flies_smoothed = [Predictor2D.smooth_3D_points(fly) for fly in all_flies]
        return all_flies, all_flies_smoothed

    def get_3D_points(self, points_2D, cropzone):
        points_3D_all_multiviews, _ = self.triangulate.triangulate_points_all_possible_views(points_2D, cropzone)
        points_3D_4_cams = points_3D_all_multiviews[:, :, -1, :]  # take the 4 views triangulation
        return points_3D_4_cams

    @staticmethod
    def get_distance_per_joint(ground_truth, estimated_points):
        distances = np.linalg.norm(ground_truth - estimated_points, axis=-1)
        distances_per_joints = distances.mean(axis=0)
        return distances_per_joints

    @staticmethod
    def load_from_mat(path, name, end=100):
        mat = scipy.io.loadmat(path)
        numpy_array = mat[name]
        numpy_array = np.transpose(numpy_array, (3, 2, 0, 1))
        return numpy_array[:end]


if __name__ == '__main__':
    EstimateAnalysisErrors(task=DETECTION_ERROR, num_samples=2000)
    # EstimateAnalysisErrors(task=CALIBRATION_ERROR, num_samples=2000)
