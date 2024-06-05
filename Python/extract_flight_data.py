import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import h5py
import scipy
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from scipy.interpolate import make_smoothing_spline
from scipy.signal import medfilt
from sklearn.linear_model import LinearRegression
from visualize import Visualizer
from scipy.signal import savgol_filter, find_peaks
from numpy.polynomial.polynomial import Polynomial
from utils import get_start_frame
import pandas as pd
import plotly.graph_objects as go

# matplotlib.use('TkAgg')


sampling_rate = 16000
dt = 1 / 16000
LEFT = 0
RIGHT = 1
NUM_TIPS_FOR_PLANE = 15
WINGS_JOINTS_INDS = [7, 15]
WING_TIP_IND = 2
UPPER_PLANE_POINTS = [0, 1, 2, 3]
LOWER_PLANE_POINTS = [3, 4, 5, 6]


class HalfWingbit:
    def __init__(self, start, end, phi_vals, frames, start_peak_val, end_peak_val, avarage_value, strock_direction):
        self.start = start
        self.end = end
        self.middle_frame = (end - start) / 2
        self.frames = frames
        self.phi_vals = phi_vals
        self.start_peak_val = start_peak_val
        self.end_peak_val = end_peak_val
        self.avarage_value = avarage_value
        self.strock_direction = strock_direction
        self.hign_peak_val = max(self.start_peak_val, self.end_peak_val)
        self.low_peak_val = min(self.start_peak_val, self.end_peak_val)
        self.amplitude = self.hign_peak_val - self.low_peak_val

class FullWingBit:
    def __init__(self, start, end, frames, phi_vals, theta_vals, psi_vals):
        self.start = start
        self.end = end
        self.middle_frame = (end - start) / 2
        self.frames = frames
        self.phi_vals = phi_vals
        self.theta_vals = theta_vals,
        self.psi_vals = psi_vals


class FlightAnalysis:
    def __init__(self, points_3D_path, find_auto_correlation=False,
                       create_html=False, show_phi=False):

        # set attributes
        self.points_3D_path = points_3D_path
        self.points_3D = self.load_points()
        self.num_joints = self.points_3D.shape[1]
        self.num_frames = self.points_3D.shape[0]
        self.num_wings_points = self.num_joints - 2
        self.num_points_per_wing = self.num_wings_points // 2
        self.left_inds = np.arange(0, self.num_points_per_wing)
        self.right_inds = np.arange(self.num_points_per_wing, self.num_wings_points)
        self.left_wing_inds = self.left_inds[:-1]
        self.right_wing_inds = self.right_inds[:-1]
        self.all_wing_inds = np.array([self.left_wing_inds, self.right_wing_inds])
        self.all_non_body_points = np.array([self.left_inds, self.right_inds])
        self.head_tail_inds = [self.num_wings_points, self.num_wings_points + 1]
        self.wings_tips_inds = [WING_TIP_IND, WING_TIP_IND + self.num_points_per_wing]

        # calculate things
        self.points_3D = FlightAnalysis.enforce_3D_consistency(self.points_3D, self.right_inds, self.left_inds)
        self.first_analysed_frame = self.get_first_analysed_frame()
        self.head_tail_points = self.get_head_tail_points(smooth=True)
        self.points_3D[:, self.head_tail_inds, :] = self.head_tail_points
        self.x_body = self.get_head_tail_vec()
        self.set_right_left()
        self.wings_tips_left, self.wings_tips_right = self.get_wing_tips()
        self.wings_joints_points = self.get_wings_joints(smooth=True)
        self.wings_joints_vec = self.get_wings_joints_vec()
        self.all_2_planes = self.get_planes(points_inds=self.all_wing_inds)
        self.all_upper_planes = self.get_planes(points_inds=self.all_wing_inds[:, UPPER_PLANE_POINTS])
        self.all_lower_planes = self.get_planes(points_inds=self.all_wing_inds[:, LOWER_PLANE_POINTS])

        # visualize planes
        # Visualizer.show_points_and_wing_planes_3D(self.points_3D[:100], self.all_upper_planes)

        self.left_wing_CM, self.right_wing_CM = self.get_wings_CM()
        self.left_wing_span, self.right_wing_span = self.get_wings_spans()
        self.left_wing_chord, self.right_wing_chord = self.get_wings_cords()

        # check = FlightAnalysis.row_wize_dot(self.left_wing_chord, self.left_wing_chord)

        self.center_of_mass = self.get_center_of_mass()
        self.CM_speed, self.CM_dot = self.get_body_speed()

        self.y_body, self.first_y_body_frame, self.end_frame = self.get_roni_y_body()
        self.z_body = self.get_z_body()

        self.yaw_angle = self.get_body_yaw()
        self.pitch_angle = self.get_body_pitch()
        self.roll_angle = self.get_body_roll()

        self.yaw_dot = self.get_dot(self.yaw_angle)
        self.pitch_dot = self.get_dot(self.pitch_angle)
        self.roll_dot = self.get_roll_dot()

        self.average_roll_angle = np.mean(self.roll_angle[self.first_y_body_frame:self.end_frame])
        self.average_roll_speed = np.nanmean(np.abs(self.roll_dot))
        self.average_roll_velocity = np.nanmean(self.roll_dot)
        self.stroke_planes = self.get_stroke_planes()
        self.wing_tips_speed = self.get_wing_tips_speed()

        self.wings_phi_left, self.wings_phi_right = self.get_wings_phi()
        self.wings_theta_left, self.wings_theta_right = self.get_wings_theta()
        self.wings_psi_left, self.wings_psi_right = self.get_wings_psi()

        self.left_amplitudes, self.right_amplitudes, self.left_half_wingbits, self.right_half_wingbits = self.get_half_wingbits_objects()

        self.left_full_wingbits, self.right_full_wingbits = self.get_full_wingbits_objects()
        self.left_wing_peaks, self.wingbit_frequencies_left, self.wingbit_average_frequency_left = self.get_waving_frequency(self.wings_phi_left)
        self.left_wing_right, self.wingbit_frequencies_right, self.wingbit_average_frequency_right = self.get_waving_frequency(self.wings_phi_right)

        self.omega_lab, self.omega_body, self.angular_speed_lab, self.angular_speed_body = self.get_angular_velocities(self.x_body, self.y_body,
                                                                                                   self.z_body, self.first_y_body_frame,
                                                                                                   self.end_frame)
        # plt.plot(self.omega_body[:, :])
        # plt.plot(self.roll_angle * 100)
        # plt.show()
        if find_auto_correlation:
            self.auto_correlation_axis_angle = self.get_auto_correlation_axis_angle(self.x_body, self.y_body,
                                                                                    self.z_body, self.first_y_body_frame,
                                                                                    self.end_frame)
            self.auto_correlation_x_body = self.get_auto_correlation_x_body(self.x_body)

        if create_html:
            dir = os.path.dirname(self.points_3D_path)
            save_path_html = os.path.join(dir, 'movie_html.html')
            Visualizer.create_movie_plot(com=self.center_of_mass, x_body=self.x_body, y_body=self.y_body,
                                         points_3D=self.points_3D,
                                         start_frame=self.first_y_body_frame, save_path=save_path_html)
        self.adjust_starting_frame()
        pass

    def get_roll_dot(self):
        roll_dot_final = np.zeros_like(self.roll_angle)
        roll_dot = self.get_dot(self.roll_angle[self.first_y_body_frame:self.end_frame])
        roll_dot_final[self.first_y_body_frame:self.end_frame] = roll_dot
        return roll_dot_final

    def get_full_wingbits_objects(self):
        all_half_wingbits_objects = [self.left_half_wingbits, self.right_half_wingbits]
        phi_wings = [self.wings_phi_left, self.wings_phi_right]
        theta_wings = [self.wings_theta_left, self.wings_theta_right]
        psi_wings = [self.wings_psi_left, self.wings_psi_right]
        full_wingbits_objects = [[], []]
        for wing in range(2):
            half_wingbits = all_half_wingbits_objects[wing]
            for i in range(0, len(half_wingbits), 2):
                first_half = half_wingbits[i]
                second_half = half_wingbits[i + 1]
                start = first_half.start
                end = second_half.end
                frames = np.unique(np.concatenate((first_half.frames, second_half.frames)))
                wingbit_phi = phi_wings[wing][frames - self.first_analysed_frame]
                wingbit_theta = theta_wings[wing][frames - self.first_analysed_frame]
                wingbit_psi = psi_wings[wing][frames - self.first_analysed_frame]
                full_wingbit = FullWingBit(start=start,
                                           end=end,
                                           frames=frames,
                                           phi_vals=wingbit_phi,
                                           theta_vals=wingbit_theta,
                                           psi_vals=wingbit_psi)
                full_wingbits_objects[wing].append(full_wingbit)
        left_full_wingbits, right_full_wingbits = full_wingbits_objects
        return left_full_wingbits, right_full_wingbits
    def get_half_wingbits_objects(self):
        phi_wings = [self.wings_phi_left, self.wings_phi_right]
        half_wingbits = [[],[]]
        phi_amplitudes = [[],[]]
        for wing in range(2):
            phi = phi_wings[wing]
            max_peak_values, max_peaks_frames, min_peak_values, min_peaks_frames = self.get_phi_peaks(phi)
            max_peaks_frames = np.array(max_peaks_frames) + self.first_analysed_frame
            min_peaks_frames = np.array(min_peaks_frames) + self.first_analysed_frame
            max_peaks = np.column_stack((max_peaks_frames, max_peak_values))
            min_peaks = np.column_stack((min_peaks_frames, min_peak_values))
            all_peaks = np.concatenate((max_peaks, min_peaks), axis=0)
            all_peaks = all_peaks[all_peaks[:, 0].argsort()]
            all_amplitudes = np.abs(np.diff(all_peaks[:, 1]))
            if np.min(all_amplitudes) < 100:
                assert f"need to check minimum amplitude in phi wing {wing}"
            for cur_peak_ind in range(len(all_peaks) - 1):
                next_peak_ind = cur_peak_ind + 1
                cur_peak_val = all_peaks[cur_peak_ind, 1]
                next_peak_val = all_peaks[next_peak_ind, 1]
                avarage_val = (cur_peak_val + next_peak_val)/2
                half_bit_start = np.round(all_peaks[cur_peak_ind, 0]).astype(int)
                half_bit_end = np.round(all_peaks[next_peak_ind, 0]).astype(int)
                halfbit_frames = np.arange(half_bit_start, half_bit_end)
                strock_direction = "back_strock" if next_peak_val > cur_peak_val else "up_strock"
                amplitude = np.abs(cur_peak_val - next_peak_val)
                mid_index = (all_peaks[cur_peak_ind, 0] + all_peaks[next_peak_ind, 0])/2
                half_wingbit_obj = HalfWingbit(start=all_peaks[cur_peak_ind, 0],
                                          end=all_peaks[next_peak_ind, 0],
                                          frames=halfbit_frames,
                                          phi_vals=phi[halfbit_frames],
                                          start_peak_val=cur_peak_val,
                                          end_peak_val=next_peak_val,
                                          avarage_value=avarage_val,
                                          strock_direction=strock_direction)
                half_wingbits[wing].append(half_wingbit_obj)
                phi_amplitudes[wing].append((mid_index, amplitude))
        left_half_wingbits, right_half_wingbits = half_wingbits
        left_wing_amplitudes, right_wing_amplitudes = np.array(phi_amplitudes[0]), np.array(phi_amplitudes[1])
        return left_wing_amplitudes, right_wing_amplitudes, left_half_wingbits, right_half_wingbits

    @staticmethod
    def add_nan_frames(original_array, N):
        nan_frames = np.full((N,) + original_array.shape[1:], np.nan)
        new_array = np.concatenate((nan_frames, original_array), axis=0)
        return new_array

    @staticmethod
    def fill_with_nans(original_array, indices):
        nan_frames = np.full((len(indices),) + original_array.shape[1:], np.nan)
        original_array[indices, ...] = nan_frames
        return original_array

    def adjust_starting_frame(self):
        # fill all the not analysed frames with nans
        # fill with nans all arrays that rely on y_body
        indices = np.concatenate((np.arange(0, self.first_y_body_frame), np.arange(self.end_frame, self.num_frames - 1)))
        self.y_body = FlightAnalysis.fill_with_nans(self.y_body, indices)
        self.z_body = FlightAnalysis.fill_with_nans(self.z_body, indices)
        self.roll_angle = FlightAnalysis.fill_with_nans(self.roll_angle, indices)
        self.roll_dot = FlightAnalysis.fill_with_nans(self.roll_dot, indices)
        self.omega_lab = FlightAnalysis.fill_with_nans(self.omega_lab, indices)
        self.stroke_planes = FlightAnalysis.fill_with_nans(self.stroke_planes, indices)
        self.wings_phi_left = FlightAnalysis.fill_with_nans(self.wings_phi_left, indices)
        self.wings_phi_right = FlightAnalysis.fill_with_nans(self.wings_phi_right, indices)
        self.wings_theta_right = FlightAnalysis.fill_with_nans(self.wings_theta_right, indices)
        self.wings_theta_left = FlightAnalysis.fill_with_nans(self.wings_theta_left, indices)
        self.wings_psi_right = FlightAnalysis.fill_with_nans(self.wings_psi_right, indices)
        self.wings_psi_left = FlightAnalysis.fill_with_nans(self.wings_psi_left, indices)

        # add nan frames before the starting frame of the analysis
        attributes = [
            "points_3D", "head_tail_points", "x_body", "y_body", "z_body",
            "wings_tips_left", "wings_tips_right", "left_wing_CM", "right_wing_CM", "wings_joints_points",
            "CM_dot", "CM_speed",
            "left_wing_span", "right_wing_span", "left_wing_chord", "right_wing_chord",
            "all_2_planes", "all_upper_planes", "all_lower_planes", "wings_span_vecs",
            "wings_joints_vec", "wings_joints_vec_smoothed", "yaw_angle", "pitch_angle", "roll_angle",
            "roll_dot", "pitch_dot", "yaw_dot", "stroke_planes", "center_of_mass", "body_speed",
            "wing_tips_speed", "wings_phi_left", "wings_phi_right", "wings_theta_left", "wings_theta_right",
            "wings_psi_left", "wings_psi_right",
             "omega_lab", "omega_body", "angular_speed_lab", "angular_speed_body"
        ]

        for attr in attributes:
            if hasattr(self, attr):
                if attr not in ["left_half_wingbits", "self.right_half_wingbits",
                                "left_amplitudes", "right_amplitudes"]:
                    data = getattr(self, attr)
                    # Call the add_nan_frames method and update the attribute
                    updated_data = FlightAnalysis.add_nan_frames(data, self.first_analysed_frame)
                    setattr(self, attr, updated_data)

    def get_first_analysed_frame(self):
        directory_path = os.path.dirname(os.path.realpath(self.points_3D_path))
        start_frame = get_start_frame(directory_path)
        return int(start_frame)

    @staticmethod
    def low_pass_filter(data, cutoff, fs):
        """
        Apply a low-pass filter to the data array.

        Parameters:
            data (np.array): Input data array.
            cutoff (float): Cutoff frequency in Hz.
            fs (int): Sampling frequency of the data.

        Returns:
            np.array: Filtered data.
        """
        n = len(data)  # length of the data
        freq = np.fft.fftfreq(n, d=1 / fs)  # frequency array

        # Fourier transform of the data
        data_fft = np.fft.fft(data)

        # Create a mask for frequencies higher than the cutoff
        mask = abs(freq) > cutoff

        # Apply mask to the Fourier spectrum
        data_fft[mask] = 0

        # Inverse Fourier Transform to get back to time domain
        filtered_data = np.fft.ifft(data_fft)

        return filtered_data.real  # Return the real part of the complex output



    @staticmethod
    def savgol_smoothing(input_points, lam, polyorder, window_length, median_kernel=1, plot=False):
        points_smoothed = np.zeros_like(input_points)
        for pnt in range(input_points.shape[1]):
            points_orig = input_points[:, pnt, :]
            points = np.apply_along_axis(medfilt, axis=0, arr=points_orig, kernel_size=median_kernel)
            A = np.arange(len(points))
            for axis in range(3):
                points[:, axis] = savgol_filter(np.copy(points[:, axis]), window_length, polyorder)
                spline = make_smoothing_spline(A, points[:, axis], lam=lam * len(A))
                points[:, axis] = spline(A)
            if plot:
                plt.plot(points_orig)
                plt.plot(points)
                plt.show()
            points_smoothed[:, pnt, :] = points
        return points_smoothed

    def get_wings_cords(self):
        wings_spans = np.concatenate((self.left_wing_span[:, np.newaxis, :],
                                      self.right_wing_span[:, np.newaxis, :]), axis=1)
        wing_reference_points = [[4, 1], [12, 9]]
        estimated_chords_left = normalize(self.points_3D[:, wing_reference_points[LEFT][1]] -
                                          self.points_3D[:, wing_reference_points[LEFT][0]])
        estimated_chords_right = normalize(self.points_3D[:, wing_reference_points[RIGHT][1]] -
                                           self.points_3D[:, wing_reference_points[RIGHT][0]])
        estimated_chords = [estimated_chords_left, estimated_chords_right]
        wings_chords = np.zeros_like(wings_spans)
        for frame in range(self.num_frames):
            for wing in range(2):
                span = wings_spans[frame, wing, :]
                # wing_plane_normal = self.all_2_planes[frame, wing, :-1]
                wing_plane_normal = self.all_upper_planes[frame, wing, :-1]
                chord = np.cross(span, wing_plane_normal)
                chord /= np.linalg.norm(chord)
                # check direction
                estimated_chord = estimated_chords[wing][frame]
                if np.dot(chord, estimated_chord) < 0:
                    chord = -chord
                # make sure chord is
                wings_chords[frame, wing, :] = chord
        # for wing in range(2):
        #     chords = wings_chords[:, wing]
        #     projections = FlightAnalysis.row_wize_dot(chords, self.x_body)
        #     mean_projection = projections.mean()
        #     if mean_projection < 0:
        #         wings_chords[:, wing] = -wings_chords[:, wing]
        return wings_chords[:, LEFT], wings_chords[:, RIGHT]

    def get_wings_psi(self):
        all_psi = np.zeros((2, self.num_frames))
        all_psi_rad = np.zeros((2, self.num_frames))  # Store radians here for unwrapping

        for frame in range(self.first_y_body_frame, self.num_frames):
            for wing in range(2):
                strkpln = self.stroke_planes[frame, :-1]
                if wing == RIGHT:
                    spn = self.right_wing_span[frame]
                    chord = self.right_wing_chord[frame]
                    signy = 1
                else:
                    spn = self.left_wing_span[frame]
                    chord = self.left_wing_chord[frame]
                    signy = -1

                Surf_sp = np.cross(strkpln, spn)
                Surf_sp /= np.linalg.norm(Surf_sp)
                yax = np.cross(spn, Surf_sp)
                yax /= np.linalg.norm(yax)
                ypsi = signy * np.dot(chord, yax)
                xpsi = np.dot(chord, Surf_sp)
                angtmp = np.arctan2(ypsi, xpsi)
                all_psi_rad[wing, frame] = angtmp  # Store radians

        # Unwrap radians to prevent jumps
        all_psi_rad = np.unwrap(all_psi_rad, axis=1)

        # Convert radians to degrees
        all_psi_deg = np.rad2deg(all_psi_rad)

        # Split into left and right wing psi
        left_wing_psi, right_wing_psi = all_psi_deg
        return left_wing_psi, right_wing_psi

    def get_head_tail_points(self, smooth=True):
        head_tail_points = self.points_3D[:, self.head_tail_inds, :]
        if smooth:
            head_tail_smoothed = FlightAnalysis.savgol_smoothing(head_tail_points, lam=100, polyorder=1,
                                                                 window_length=73*1, median_kernel=41)
            head_tail_points = head_tail_smoothed
        return head_tail_points

    def get_wings_joints(self, smooth=True):
        wings_joints = self.points_3D[:, WINGS_JOINTS_INDS, :]
        if smooth:
            wings_joints_smoothed = FlightAnalysis.savgol_smoothing(wings_joints, lam=100, polyorder=1,
                                                                    window_length=3*73, median_kernel=41)
            wings_joints = wings_joints_smoothed
        return wings_joints

    def get_wings_CM(self):
        CM_left = np.mean(self.points_3D[:, self.left_wing_inds, :], axis=1)
        CM_right = np.mean(self.points_3D[:, self.right_wing_inds, :], axis=1)
        return CM_left, CM_right

    def get_wing_tips(self):
        wing_tips = self.points_3D[:, self.wings_tips_inds, :]
        wings_tips_left, wings_tips_right = wing_tips[:, LEFT, :], wing_tips[:, RIGHT, :]
        return wings_tips_left, wings_tips_right

    def get_center_of_mass(self):
        CM = np.mean(self.head_tail_points, axis=1)
        return CM

    def get_body_speed(self):
        return self.get_speed(self.center_of_mass)

    def get_wing_tips_speed(self):
        left_tip_speed, _ = self.get_speed(self.wings_tips_left[:, :])
        right_tip_speed, _ = self.get_speed(self.wings_tips_right[:, :])
        wing_tips_speed = np.concatenate((right_tip_speed[:, np.newaxis], left_tip_speed[:, np.newaxis]), axis=1)
        return wing_tips_speed

    @staticmethod
    def get_speed(points_3d):
        T = np.arange(len(points_3d))
        derivative_3D = np.zeros((len(points_3d), 3))
        for axis in range(3):
            derivative_3D[:, axis] = FlightAnalysis.get_dot(points_3d[:, axis])
        speed = np.linalg.norm(derivative_3D, axis=1)
        return speed, derivative_3D

    def get_wings_spans(self):
        """
        calculates the wing spans as the normalized vector from the wing center of mass to the wing tip
        Returns: an array of size (num_frames, 2, 3), axis 1 is left and right
        """
        wing_spans = np.zeros((self.num_frames, 2, 3))
        wings_CM = np.array([self.left_wing_CM, self.right_wing_CM])
        wings_tips = [self.wings_tips_left, self.wings_tips_right]
        for wing in range(2):
            CMs = wings_CM[wing]
            tip = wings_tips[wing]
            wing_span = tip - CMs
            wing_span = wing_span / np.linalg.norm(wing_span, axis=1)[:, np.newaxis]
            wing_spans[:, wing, :] = wing_span
        left_wing_span = wing_spans[:, LEFT, :]
        right_wing_span = wing_spans[:, RIGHT, :]
        return left_wing_span, right_wing_span

    def get_z_body(self):
        z_body = np.cross(self.x_body, self.y_body, axis=-1)
        z_body = normalize(z_body, 'l2')
        return z_body

    def set_right_left(self):
        wing_CM_left = np.mean(self.points_3D[0, self.left_inds[:-1], :], axis=0)
        wing_CM_right = np.mean(self.points_3D[0, self.right_inds[:-1], :], axis=0)
        wings_vec = wing_CM_right - wing_CM_left
        wings_vec = wings_vec / np.linalg.norm(wings_vec)
        cross = np.cross(wings_vec, self.x_body[0])
        z = 2
        need_flip = False
        if cross[z] < 0:
            need_flip = True
        if need_flip:
            left = self.points_3D[:, self.left_inds, :]
            right = self.points_3D[:, self.right_inds, :]
            self.points_3D[:, self.left_inds, :] = right
            self.points_3D[:, self.right_inds, :] = left

    def load_points(self):
        return np.load(self.points_3D_path)

    def get_head_tail_vec(self, smooth=False):
        head_tail_vec = self.head_tail_points[:, 1, :] - self.head_tail_points[:, 0, :]
        head_tail_vec = normalize(head_tail_vec, axis=1, norm='l2')

        return head_tail_vec

    def get_wings_joints_vec(self, smooth=False):
        wings_joints_vec = self.wings_joints_points[:, LEFT] - self.wings_joints_points[:, RIGHT]
        wings_joints_vec = normalize(wings_joints_vec, axis=1, norm='l2')
        return wings_joints_vec

    def get_body_pitch(self):
        pitch = np.rad2deg(np.arcsin(self.x_body[:, 2]))
        return pitch

    def get_body_yaw(self):
        only_xy = normalize(self.x_body[:, :-1], axis=1, norm='l2')
        yaw = np.rad2deg(np.arctan2(only_xy[:, 1], only_xy[:, 0]))
        yaw = np.unwrap(yaw + 180, period=360) - 180
        return yaw

    def get_Rzy(self, phi, theta):
        num_frames = phi.shape[0]
        Rzy_all = []
        for i in range(num_frames):
            x_body_i = self.x_body[i]
            Rzy = FlightAnalysis.euler_rotation_matrix(np.deg2rad(self.yaw_angle[i]), np.deg2rad(self.pitch_angle[i]), psi_rad=0)
            Rzy_all.append(Rzy)
        Rzy_all = np.array(Rzy_all)
        return Rzy_all


    def get_body_roll(self):
        phi = self.yaw_angle
        theta = self.pitch_angle
        Rzy_all = self.get_Rzy(phi, theta)
        all_roll_angles = np.zeros(self.num_frames,)
        for frame in range(self.first_y_body_frame, self.num_frames):
            Rzy = Rzy_all[frame]
            yb_frame = self.y_body[frame]
            rotated_yb_frame = Rzy @ yb_frame
            roll_frame = np.arctan(rotated_yb_frame[2]/rotated_yb_frame[1])
            roll_angle = np.rad2deg(roll_frame)
            all_roll_angles[frame] = roll_angle
        all_roll_angles = np.array(all_roll_angles)
        # roll_angle = np.rad2deg(np.arcsin(self.y_body[:, 2]))
        # roll_wings_joints = np.rad2deg(np.arcsin(self.wings_joints_vec[:, 2]))
        return all_roll_angles

    @staticmethod
    def get_dot(data, window_length=5, polyorder=2):
        data_dot = savgol_filter(data, window_length, polyorder, deriv=1, delta=(1 / sampling_rate))
        # plt.plot(data_dot / 100)
        # plt.plot(data)
        # plt.axhline(y=0, color='gray', linestyle='--')
        # plt.show()
        return data_dot

    def get_body_angles(self):
        yaw_angle = self.get_body_yaw()
        pitch_angle = self.get_body_pitch()
        roll_angle = self.get_body_roll()
        return yaw_angle, pitch_angle, roll_angle

    @staticmethod
    def calculate_angle(arr1, arr2):

        arr1[np.abs(arr1) < 1e-8] = 0
        arr2[np.abs(arr2) < 1e-8] = 0
        dot_product = FlightAnalysis.row_wize_dot(arr1, arr2)
        angles_rad = np.arccos(dot_product)
        angle_degrees = np.rad2deg(angles_rad)
        return angle_degrees


    @staticmethod
    def consistent_angle(arr1, arr2, stroke_normal):
        angles = []
        for i, (a, b, n) in enumerate(zip(arr1, arr2, stroke_normal)):
            cos_theta = np.dot(a, b)
            cos_theta = np.clip(cos_theta, -1, 1)  # Clipping to avoid numerical issues outside arccos range
            theta_rad = np.arccos(cos_theta)

            cross_prod = np.cross(a, b)
            if np.dot(cross_prod, n) < 0:  # Assuming the Z-axis as reference; change depending on your reference axis
                theta_rad = 2 * np.pi - theta_rad
            theta_deg = np.rad2deg(theta_rad)
            angles.append(theta_deg)  # Convert radians to degrees

        return np.array(angles)


    def get_wings_theta(self):
        wings_spans_left = self.left_wing_span
        wings_spans_right = self.right_wing_span
        stroke_normals = self.stroke_planes[:, :-1]
        theta_left = 90 - np.rad2deg(np.arccos(FlightAnalysis.row_wize_dot(wings_spans_left, stroke_normals)))
        theta_right = 90 - np.rad2deg(np.arccos(FlightAnalysis.row_wize_dot(wings_spans_right, stroke_normals)))
        return theta_left, theta_right

    def get_wings_phi(self):
        proj_left = FlightAnalysis.project_vector_to_plane(self.left_wing_span, self.stroke_planes)
        proj_right = FlightAnalysis.project_vector_to_plane(self.right_wing_span, self.stroke_planes)
        proj_xbody = FlightAnalysis.project_vector_to_plane(self.x_body, self.stroke_planes)

        projected_wings = np.array([proj_left, proj_right])
        thetas = []
        for wing_num in range(2):
            proj = projected_wings[wing_num]
            stroke_plane_normal = self.stroke_planes[:, :-1]
            if wing_num == 1:
                stroke_plane_normal = -stroke_plane_normal
            theta = FlightAnalysis.consistent_angle(proj, proj_xbody, stroke_plane_normal)
            theta = 360 - theta
            theta[:self.first_y_body_frame] = 0
            theta[self.end_frame:] = 0
            thetas.append(theta)

        theta_left, theta_right = thetas

        # plt.plot(theta_left[self.start_frame:self.end_frame])
        # plt.plot(theta_right[self.start_frame:self.end_frame])
        # plt.show()

        return theta_left, theta_right


    def get_waving_frequency(self, data):
        refined_peaks, peak_values = self.get_peaks(data)
        distances = np.diff(refined_peaks)
        frequencies = (1 / distances) * sampling_rate
        average_distance = np.mean(distances)
        average_frequency = (1 / average_distance) * sampling_rate
        return refined_peaks, frequencies, average_frequency

    def get_phi_peaks(self, phi):
        max_peaks_inds, max_peak_values = self.get_peaks(phi, show=False, prominence=90)
        all_max_peaks = np.stack([max_peaks_inds, max_peak_values]).T
        min_peaks_inds, min_peak_values = self.get_peaks(-phi, show=False, prominence=90)
        min_peak_values = [-min_peak for min_peak in min_peak_values]
        return max_peak_values, max_peaks_inds, min_peak_values, min_peaks_inds

    def get_peaks(self, data, distance=50, prominence=100, show=False, window_size=7, height=None):
        peaks, _ = find_peaks(data, height=height, prominence=prominence, distance=distance )
        if show:
            plt.figure(figsize=(10, 6))
            x_values = np.arange(self.first_y_body_frame, self.end_frame)
            plt.plot(x_values, data[self.first_y_body_frame: self.end_frame], label='Data', marker='o')
        refined_peaks = []
        peak_values = []
        for peak in peaks:
            # Define the window around the peak
            start = max(0, peak - window_size // 2)
            end = min(len(data), peak + window_size // 2 + 1)

            # Fit a second-degree polynomial
            x = np.arange(start, end)
            y = data[start:end]
            coef = np.polyfit(x, y, 2)

            # Create a polynomial object
            p = Polynomial(coef[::-1])  # np.polyfit returns highest power first

            # Find the vertex of the parabola (the maximum point for a concave down parabola)
            vertex = -coef[1] / (2 * coef[0])

            # Check if vertex is within the window
            if start <= vertex <= end:
                refined_peak = vertex
            else:
                refined_peak = peak
            value = p(refined_peak)
            refined_peaks.append(refined_peak)
            peak_values.append(value)

            # Plotting the polynomial over the window
            if show:
                x_fit = np.linspace(start, end, 100)
                y_fit = p(x_fit)
                plt.plot(x_fit, y_fit, label=f'Fit around peak at {peak}', linestyle='--')
        if show:
            plt.scatter(peaks, data[peaks], color='red', label='Initial Peaks')
            plt.scatter(refined_peaks, data[np.round(refined_peaks).astype(int)], color='green', marker='x',
                        label='Refined Peaks')
            # plt.legend()
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.title('Peak Detection and Polynomial Fit')
            plt.show()
        return refined_peaks, peak_values

    @staticmethod
    def project_vector_to_plane(Vs, Ps):
        stroke_normals = Ps[:, :-1]
        proj_N_V = FlightAnalysis.row_wize_dot(Vs, stroke_normals)[:, np.newaxis] * stroke_normals
        V_on_plane = Vs - proj_N_V
        # check if indeed on the plane:
        # check = FlightAnalysis.row_wize_dot(V_on_plane, stroke_normals)
        V_on_plane = normalize(V_on_plane, norm='l2')
        return V_on_plane

    def get_stroke_planes(self):
        theta = np.pi / 4
        stroke_normal = self.rodrigues_rot(self.x_body, self.y_body, -theta)
        stroke_normal = normalize(stroke_normal, norm='l2')
        body_center = np.mean(self.head_tail_points, axis=1)
        d = - np.sum(np.multiply(stroke_normal, body_center), axis=1)
        stroke_planes = np.column_stack((stroke_normal, d))
        return stroke_planes

    @staticmethod
    def euler_rotation_matrix(phi_rad, theta_rad, psi_rad):
        """
        Returns a 3D Euler rotation matrix given angles in radians.
        :param phi_rad: Rotation about the x-axis
        :param theta_rad: Rotation about the y-axis (negated as in the original function)
        :param psi_rad: Rotation about the z-axis
        :return: 3x3 rotation matrix
        """
        # Adjust theta angle (negate as in the original function)
        theta_rad = -theta_rad

        # Calculate trigonometric values for the angles
        cph = np.cos(phi_rad)
        sph = np.sin(phi_rad)
        cth = np.cos(theta_rad)
        sth = np.sin(theta_rad)
        cps = np.cos(psi_rad)
        sps = np.sin(psi_rad)

        # Define the matrix rows according to the MATLAB function's logic
        M1 = np.array([cth * cph, cth * sph, -sth])
        M2 = np.array([sps * sth * cph - cps * sph, sps * sth * sph + cps * cph, cth * sps])
        M3 = np.array([cps * sth * cph + sps * sph, cps * sth * sph - sps * cph, cth * cps])

        # Concatenate rows into a 3x3 matrix
        TrdRotmat = np.array([M1, M2, M3])

        return TrdRotmat

    @staticmethod
    def rodrigues_rot(V, K, theta):
        """
        Args:
            V:  the vector to rotate
            K: the axis of rotation
            theta: angle in radians
        Returns:

        """
        num_frames, ndims = V.shape[0], V.shape[1]
        V_rot = np.zeros_like(V)
        for frame in range(num_frames):
            vi = V[frame, :]
            ki = K[frame, :]
            vi_rot = np.cos(theta) * vi + np.cross(ki, vi) * np.sin(theta) + ki * np.dot(ki, vi) * (1 - np.cos(theta))
            V_rot[frame, :] = vi_rot
        return V_rot

    def get_planes(self, points_inds):
        planes = np.zeros((self.num_frames, 2, 4))
        for frame in range(self.num_frames):
            for wing in range(2):
                points = self.points_3D[frame, points_inds[wing], :]
                plane_P, error = self.fit_plane(points)
                if frame > 0:
                    prev_plane_P = planes[frame - 1, 0, :]
                    if np.dot(prev_plane_P, plane_P) < 0:
                        plane_P = -plane_P
                planes[frame, wing, :] = plane_P
        return planes


    @staticmethod
    def get_auto_correlation_axis_angle(x_body, y_body, z_body, start_frame, end_frame):
        # first_nonzero_index = np.argmax((y_body != 0).any(axis=1))
        # reversed_mat = np.flip(y_body, axis=0)
        # last_index_reversed = np.argmax((reversed_mat != 0).any(axis=1))
        # last_nonzero_index = y_body.shape[0] - 1 - last_index_reversed

        T = (end_frame - start_frame) // 2
        x_body, y_body, z_body = (x_body[start_frame:end_frame],
                                  y_body[start_frame:end_frame],
                                  z_body[start_frame:end_frame])
        AC = np.zeros(T)
        AC[0] = 1
        for df in range(1, T):
            xb, yb, zb = x_body[:-df], y_body[:-df], z_body[:-df]
            Rs = np.stack([xb, yb, zb], axis=-1)

            xb_pair, yb_pair, zb_pair = x_body[df:], y_body[df:], z_body[df:]
            Rs_pair = np.stack([xb_pair, yb_pair, zb_pair], axis=-1)

            angels_radiance = np.array(
                [FlightAnalysis.get_rotation_axis_angle(Rs[i], Rs_pair[i]) for i in range(Rs.shape[0])])
            cosines = np.cos(angels_radiance)
            AC[df] = np.mean(cosines)
        return AC

    @staticmethod
    def get_angular_velocities(x_body, y_body, z_body, start_frame, end_frame):
        num_frames = len(x_body)
        x_body, y_body, z_body = (x_body[start_frame:end_frame],
                                  y_body[start_frame:end_frame],
                                  z_body[start_frame:end_frame])
        Rs = np.stack([x_body, y_body, z_body], axis=-1)
        N = len(Rs)
        T = np.arange(N)
        dRdt = np.zeros_like(Rs)
        # find dRdt
        for i in range(3):
            for j in range(3):
                entry_ij = Rs[:, i, j]
                spline = make_smoothing_spline(y=entry_ij, x=T)
                vals = spline(T)
                derivative = spline.derivative()(T)
                dRdt[:, i, j] = derivative

        w_x, w_y, w_z = np.zeros((3, N))
        for frame in range(N):
            A_dot = dRdt[frame]
            A = Rs[frame]
            omega = A_dot @ A.T
            wx_frame = (omega[2, 1] - omega[1, 2])/2
            wy_frame = (omega[0, 2] - omega[2, 0])/2
            wz_frame = (omega[1, 0] - omega[0, 1])/2
            w_x[frame] = np.rad2deg(wx_frame) * sampling_rate
            w_y[frame] = np.rad2deg(wy_frame) * sampling_rate
            w_z[frame] = np.rad2deg(wz_frame) * sampling_rate

        omega_lab = np.column_stack((w_x, w_y, w_z))
        omega_body = np.zeros_like(omega_lab)
        for frame in range(N):
            omega_lab_i = omega_lab[frame, :]
            R = Rs[frame]
            omega_body_i = R.T.dot(omega_lab_i)
            omega_body[frame, :] = omega_body_i

        nan_frames = np.full((start_frame, 3), np.nan)
        omega_lab = np.concatenate((nan_frames, omega_lab), axis=0)
        omega_body = np.concatenate((nan_frames, omega_body), axis=0)

        nan_frames = np.full((num_frames - end_frame, 3), np.nan)
        omega_lab = np.concatenate((omega_lab, nan_frames), axis=0)
        omega_body = np.concatenate((omega_body, nan_frames), axis=0)

        angular_speed_body = np.linalg.norm(omega_body, axis=-1)
        angular_speed_lab = np.linalg.norm(omega_lab, axis=-1)

        return omega_lab, omega_body, angular_speed_lab, angular_speed_body
    @staticmethod
    def get_auto_correlation_x_body(x_body):
        T = len(x_body) // 2
        AC = np.zeros(T)
        AC[0] = 1
        for df in range(1, T):
            x_bodies = x_body[:-df]
            x_bodies_pair = x_body[df:]
            cosines = FlightAnalysis.row_wize_dot(x_bodies, x_bodies_pair)
            AC[df] = np.mean(cosines)
        return AC

    @staticmethod
    def get_rotation_axis_angle(vectors1, vectors2):
        # vectors1, vectors2 (3, 3) matrices of perpendicular unit vectors (each row is a vector)
        # Create rotation matrix
        R = np.dot(vectors2, vectors1.T)

        # Calculate rotation axis
        # _, v = np.linalg.eig(R)
        # axis = np.real(v[:, np.isclose(np.linalg.eigvals(R), 1)])

        # Calculate rotation angle
        angle_rad = np.arccos((np.trace(R) - 1) / 2)
        return angle_rad

    @staticmethod
    def fit_plane(points):

        # fit a plane
        pca = PCA(n_components=3)
        pca.fit(points)
        normal = pca.components_[-1]  # the z component of the data
        mean = pca.mean_
        d = - normal.T @ mean  # ax + by + cz + d = 0 -> ax + by + cz = -d
        plane_P = np.append(normal, d)

        # find the fit error
        error = np.mean(np.abs(points @ normal + d))
        return plane_P, error

    @staticmethod
    def auto_correlation_1D(angles, is_radians=False):
        """
        computes the auto-correlation between angles:
        C(dt) =  <cos[Theta(t) - theta(t + dt)]>
        """
        T = len(angles) // 2
        AC = np.zeros(T)
        AC[0] = 1
        if not is_radians:
            angles = np.deg2rad(angles)
        for df in range(1, T):
            d_theta = angles[df:] - angles[:-df]
            cosines = np.cos(d_theta)
            AC[df] = np.mean(cosines)
        return AC

    @staticmethod
    def auto_correlation_3D(unit_vectors):
        T = len(unit_vectors) // 2
        AC = np.zeros(T)
        AC[0] = 1
        for df in range(1, T):
            first_vecs = unit_vectors[df:]
            second_vecs = unit_vectors[:-df]
            cosines = FlightAnalysis.row_wize_dot(first_vecs, second_vecs)
            AC[df] = np.mean(cosines)
        return AC

    @staticmethod
    def row_wize_dot(arr1, arr2):
        dot = np.sum(arr1 * arr2, axis=1)
        return dot

    @staticmethod
    def find_zero_crossings_up(data):
        # data is 1D
        # find when sign is changing from negative to positive
        return np.where(np.diff(np.sign(data)) > 0)[0]

    @staticmethod
    def remove_close_elements(data, threshold=60):
        diffs = np.abs(np.diff(data))
        indices = np.insert(diffs >= threshold, 0, True)
        return data[indices]


    def get_roni_y_body(self):
        idx4StrkPln = self.choose_span()
        y_bodies = []
        for i, ind in enumerate(idx4StrkPln):
            left = self.wings_tips_left[ind - NUM_TIPS_FOR_PLANE:ind + NUM_TIPS_FOR_PLANE, :]
            right = self.wings_tips_right[ind - NUM_TIPS_FOR_PLANE:ind + NUM_TIPS_FOR_PLANE, :]
            points = np.concatenate((left, right), axis=0)
            wing_tips_plane = self.fit_plane(points)[0]
            plane_normal = wing_tips_plane[:-1]
            y_body = np.cross(plane_normal, self.x_body[ind])
            y_body = y_body / np.linalg.norm(y_body)
            left_span = self.left_wing_span[ind]
            if np.dot(y_body, left_span) < 0:
                y_body = - y_body
            y_bodies.append(y_body)
            # self.plot_plane_and_points(ind, wing_tips_plane, points, y_body)
            pass
        y_bodies = np.array(y_bodies)
        all_y_bodies = np.zeros_like(self.x_body)
        first_y_body_frame = np.min(idx4StrkPln)
        end = np.max(idx4StrkPln)
        x = np.arange(first_y_body_frame, end)
        f1 = interp1d(idx4StrkPln, y_bodies[:, 0], kind='quadratic')
        f2 = interp1d(idx4StrkPln, y_bodies[:, 1], kind='quadratic')
        f3 = interp1d(idx4StrkPln, y_bodies[:, 2], kind='quadratic')
        Ybody_inter = np.vstack((f1(x), f2(x), f3(x))).T

        Ybody_inter = FlightAnalysis.savgol_smoothing(Ybody_inter[:, np.newaxis, :], lam=1, polyorder=1,
                                        window_length=73*2, median_kernel=1)
        Ybody_inter = np.squeeze(Ybody_inter)
        Ybody_inter = normalize(Ybody_inter, axis=1, norm='l2')
        all_y_bodies[first_y_body_frame:end, :] = Ybody_inter
        # make sure that the all_y_bodies are (1) unit vectors and (2) perpendicular to x_body
        y_bodies_corrected = all_y_bodies - self.x_body * self.row_wize_dot(self.x_body, all_y_bodies).reshape(-1, 1)
        y_bodies_corrected = normalize(y_bodies_corrected, 'l2')


        # plt.plot(idx4StrkPln, y_bodies)
        # plt.plot(y_bodies_corrected)
        # plt.show()

        return y_bodies_corrected, first_y_body_frame, end

    def choose_span(self):
        dotspanAx_wing1 = self.row_wize_dot(self.right_wing_span, self.x_body)
        dotspanAx_wing2 = self.row_wize_dot(self.left_wing_span, self.x_body)

        dotspanAx_wing2 = self.filloutliers(dotspanAx_wing2, 'pchip')
        dotspanAx_wing1 = self.filloutliers(dotspanAx_wing1, 'pchip')

        distSpans = np.degrees(
            np.arccos(np.clip(self.row_wize_dot(self.right_wing_span, self.left_wing_span), -1.0, 1.0)))
        angBodSp = np.degrees(
            np.arccos(np.clip(self.row_wize_dot(self.right_wing_span, self.x_body), -1.0, 1.0)))

        mean_strks = np.mean([dotspanAx_wing1, dotspanAx_wing2], axis=0)
        changeSgn = np.vstack((mean_strks < 0, mean_strks >= 0))

        FrbckStrk = self.find_up_down_strk(changeSgn, mean_strks, 0)
        idx4StrkPln = self.choose_grang_wing1_wing2(distSpans, FrbckStrk, 140, 10)

        diff_threshold = 60
        while np.any(np.diff(idx4StrkPln) < diff_threshold):
            idx4StrkPln = np.delete(idx4StrkPln, np.where(np.diff(idx4StrkPln) < diff_threshold)[0] + 1)

        idx4StrkPln = np.unique(idx4StrkPln)
        idx4StrkPln = idx4StrkPln[angBodSp[idx4StrkPln] > 70]

        return idx4StrkPln

    def choose_grang_wing1_wing2(self, distSpans, FrbckStrk, angleTH, Search_cell):
        idx4StrkPln = FrbckStrk[distSpans[FrbckStrk] > angleTH]

        for i in range(len(idx4StrkPln)):
            inds2ch = np.arange(idx4StrkPln[i] - Search_cell, idx4StrkPln[i] + Search_cell + 1)
            inds2ch = inds2ch[(inds2ch >= 0) & (inds2ch < len(distSpans))]
            idx4StrkPln[i] = inds2ch[np.argmax(distSpans[inds2ch])]

        return idx4StrkPln

    @staticmethod
    def filloutliers(data, method):
        # Simple implementation of outlier filling using interpolation
        if method == 'pchip':
            median = np.median(data)
            std_dev = np.std(data)
            threshold = 3 * std_dev
            outliers = np.abs(data - median) > threshold
            data[outliers] = np.interp(np.flatnonzero(outliers), np.flatnonzero(~outliers), data[~outliers])
            return data
        else:
            raise ValueError("Unsupported method")

    @staticmethod
    def find_up_down_strk(changeSgn, mean_strks, up_down_strk):
        """
        Find where the signs change (compared to 0 on the body axis).
        In this position, the wings are farthest apart.

        Parameters:
        changeSgn (ndarray): Array indicating sign changes.
        mean_strks (ndarray): Mean strokes array.
        up_down_strk (int): Value to compare for finding up and down strokes.

        Returns:
        ndarray: Indices of down strokes.
        """
        downstrk = np.where((changeSgn[0, :-1] + changeSgn[1, 1:]) == up_down_strk)[0]
        mean_val = np.vstack((mean_strks[downstrk + 1], mean_strks[downstrk]))
        indMin = np.argmin(np.abs(mean_val), axis=0)
        idx_vec = np.vstack((downstrk + 1, downstrk))
        idxdwnStrk = idx_vec[(indMin, np.arange(indMin.size))]
        return idxdwnStrk
    #################################################################################################################

    def plot_plane_and_points(self, ind, plane, points, y_body):
        size = 0.005
        cm = np.mean(self.head_tail_points[ind], axis=0)
        wing_points = points
        center = np.mean(points, axis=0)
        a, b, c, d = plane
        x = np.linspace(-size / 2, size / 2, 10) + center[0]
        y = np.linspace(-size / 2, size / 2, 10) + center[1]
        xx, yy = np.meshgrid(x, y)
        zz = (-a * xx - b * yy - d) / c
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # plot surface
        ax.plot_surface(xx, yy, zz, color='green', alpha=0.5, label='Plane')
        # plot points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        # plot center of mass
        ax.scatter(*cm, color="r", s=100)
        # plot head and tail
        ax.scatter(self.head_tail_points[ind, :, 0], self.head_tail_points[ind, :, 1], self.head_tail_points[ind, :, 2])
        # Plot the vectors
        ax.quiver(*cm, *(y_body * 0.001), color="b", alpha=.8)
        # ax.quiver(*cm, *(np.array([a, b, c]) * 0.001), color="g", alpha=.8)
        ax.set_box_aspect([1, 1, 1])
        plt.show()

    @staticmethod
    def fit_a_line_to_points(points):
        ##############################
        pca = PCA(n_components=3)
        pca.fit(points)
        best_fit_direction = pca.components_[0]
        print(pca.explained_variance_ratio_)
        ###############################
        print(FlightAnalysis.calculate_straightness(points))

    @staticmethod
    def calculate_straightness(trajectory):
        total_distance = np.sum(np.sqrt(np.sum(np.diff(trajectory, axis=0) ** 2, axis=1)))
        euclidean_distance = np.sqrt(np.sum((trajectory[-1] - trajectory[0]) ** 2))
        return euclidean_distance / total_distance

    @staticmethod
    def sort_trajectories(trajectories):
        straightness_values = [FlightAnalysis.calculate_straightness(trajectory) for trajectory in trajectories]
        sorted_indices = np.argsort(straightness_values)[::-1]  # Sort in descending order
        sorted_trajectories = [trajectories[i] for i in sorted_indices]
        return sorted_trajectories

    @staticmethod
    def enforce_3D_consistency(points_3D, right_inds, left_inds):
        num_frames  = points_3D.shape[0]
        right_points = np.zeros((num_frames, len(right_inds), 3))
        left_points = np.zeros((num_frames, len(left_inds), 3))
        # initialize
        right_points[0, ...] = points_3D[0, right_inds]
        left_points[0, ...] = points_3D[0, left_inds]

        for frame in range(1, num_frames):
            cur_left_points = points_3D[frame, left_inds, :]
            cur_right_points = points_3D[frame, right_inds, :]

            prev_left_points = left_points[frame - 1, :]
            prev_right_points = right_points[frame - 1, :]

            l2l_dist = np.linalg.norm(cur_left_points - prev_left_points)
            r2r_dist = np.linalg.norm(cur_right_points - prev_right_points)
            r2l_dist = np.linalg.norm(cur_right_points - prev_left_points)
            l2r_dist = np.linalg.norm(cur_left_points - prev_right_points)
            do_switch = l2l_dist + r2r_dist > r2l_dist + l2r_dist

            if do_switch:
                # print(f'switch occuered in frame {frame}')
                right_points[frame] = cur_left_points
                left_points[frame] = cur_right_points
            else:
                right_points[frame] = cur_right_points
                left_points[frame] = cur_left_points
        points_3D[:, right_inds, :] = right_points
        points_3D[:, left_inds, :] = left_points
        return points_3D



def plot_movie_html(movie_num):
    if movie_num == 1:
        title = "mov1 dark disturbance smoothed.html"
        point_numpy_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov61_d\points_3D_smoothed_ensemble.npy"
        start_frame = 10
    elif movie_num == 2:
        title = "mov2 dark disturbance smoothed.html"
        point_numpy_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov62_d\points_3D_smoothed_ensemble.npy"
        start_frame = 160
    elif movie_num == 3:
        title = "mov1 free flight smoothed.html"
        point_numpy_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov10_u\points_3D_smoothed_ensemble.npy"
        start_frame = 130
    else:
        title = "mov2 free flight smoothed.html"
        point_numpy_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov11_u\points_3D_smoothed_ensemble.npy"
        start_frame = 10

    FA = FlightAnalysis(point_numpy_path)

    # plt.plot(FA.wings_tips[:, 0, :])
    # plt.show()

    # ybody1 = FA.y_body
    # plt.plot(ybody1)
    # plt.show()
    # ybody2 = FA.wings_joints_vec
    # plt.figure()

    com = FA.center_of_mass
    # FA.fit_a_line_to_points(com)

    x_body = FA.x_body
    y_body = FA.y_body
    points_3D = FA.points_3D

    Visualizer.create_movie_plot(com, x_body, y_body, points_3D, start_frame, title)


def calculate_auto_correlation_roni_movies():
    X_x_body = 23
    X_y_body = 24
    X_z_body = 25
    Y_x_body = 26
    Y_y_body = 27
    Y_z_body = 28
    Z_x_body = 29
    Z_y_body = 30
    Z_z_body = 31

    h5_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\manipulated_05_12_22.hdf5"
    new_h5_path = os.path.join(r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data", "autocorrelations_roni_100.h5")
    # h5_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\manipulated_05_12_22.hdf5"
    # new_h5_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\autocorrelations_roni_500.h5"
    with h5py.File(new_h5_path, "w") as new_h5_file:
        with h5py.File(h5_path, 'r') as h5_file:
            # Access the relevant datasets
            for mov in range(1, 100, 1):
                try:
                    print(f"Movie {mov}")
                    mov_vectors = h5_file[f'mov{mov}/vectors']
                    # Extract the specified columns
                    x_body = np.vstack(mov_vectors[:, [X_x_body, X_y_body, X_z_body]])
                    y_body = np.vstack(mov_vectors[:, [Y_x_body, Y_y_body, Y_z_body]])
                    z_body = np.vstack(mov_vectors[:, [Z_x_body, Z_y_body, Z_z_body]])

                    AC_coordinate_systems = FlightAnalysis.get_auto_correlation_axis_angle(x_body, y_body, z_body,
                                                                                           0, len(x_body))
                    AC_x_body = FlightAnalysis.get_auto_correlation_x_body(x_body)

                    new_h5_file.create_dataset(f'mov{mov}/AC_coordinate_systems', data=AC_coordinate_systems)
                    new_h5_file.create_dataset(f'mov{mov}/AC_x_body', data=AC_x_body)
                except:
                    print(f"Error in movie {mov}")


def extract_auto_correlations(base_path, h5_path, file_name="points_3D_smoothed_ensemble.npy"):
    with h5py.File(h5_path, 'w') as h5_file:
        for mov in os.listdir(base_path):
            points_path = os.path.join(base_path, mov, file_name)
            if os.path.isfile(points_path):
                print(mov)
                FA = FlightAnalysis(points_path)

                AC_coordinate_systems = FA.auto_correlation_axis_angle
                AC_x_body = FA.auto_correlation_x_body

                h5_file.create_dataset(f'{mov}/AC_coordinate_systems', data=AC_coordinate_systems)
                h5_file.create_dataset(f'{mov}/AC_x_body', data=AC_x_body)


def visualize_autocorrelations_plotly_to_html(base_path_1, base_path_2, dataset_name, output_html='plot.html'):
    import plotly.graph_objects as go
    import h5py

    fig = go.Figure()

    # Helper function to add traces with conditional legend labeling and grouped by legend
    def add_dataset_traces(h5_path, name, color, legend_group, dataset_name):
        first_trace = True  # Only show legend for the first trace
        with h5py.File(h5_path, 'r') as h5_file:
            for mov in h5_file.keys():
                dataset_path = f'{mov}/{dataset_name}'  # Construct the path to the dataset
                if dataset_path in h5_file:  # Check if the dataset exists
                    data = h5_file[dataset_path][:]
                    fig.add_trace(go.Scatter(x=list(range(len(data))), y=data, mode='lines',
                                             name=name if first_trace else None,  # Name only for the first trace
                                             showlegend=first_trace,  # Show legend only for the first trace
                                             legendgroup=legend_group,  # Assign all traces to the same legend group
                                             line=dict(color=color)))
                    first_trace = False  # Subsequent traces won't show in the legend but will be grouped
    add_dataset_traces(base_path_1, 'no halter data', 'blue', 'roni_data_group', dataset_name)
    add_dataset_traces(base_path_2, 'regular flies', 'red', 'no_halter_data_group', dataset_name)
    fig.update_layout(title=f'{dataset_name} Arrays',
                      xaxis_title='Dimension/Index',
                      yaxis_title='Value')

    # Save the figure as an HTML file
    fig.write_html(output_html)


def plot_auto_correlations():
    h5_path_1 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\autocorrelations\autocorrelations_undistubed.h5"
    h5_path_2 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\autocorrelations\autocorrelations_roni_100.h5"
    output_html1 = r'C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\plot_autocorelation_CS_angle_100.html'
    visualize_autocorrelations_plotly_to_html(base_path_1=h5_path_1, base_path_2=h5_path_2,
                                              dataset_name='AC_coordinate_systems',
                                              output_html=output_html1)
    output_html2 = r'C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\plot_autocorelation_body_axis_100.html'
    visualize_autocorrelations_plotly_to_html(base_path_1=h5_path_1, base_path_2=h5_path_2,
                                              dataset_name='AC_x_body',
                                              output_html=output_html2)


def get_frequencies_from_all(base_path, csv_path):
    if os.path.isdir(base_path):
        print('Loading data from {}'.format(base_path))

        # Load CSV data
    df = pd.read_csv(csv_path)

    # Initialize columns for frequencies
    df['frequency_right'] = np.nan
    df['frequency_left'] = np.nan
    df['average_frequency'] = np.nan
    df['body_speed'] = np.nan
    df['average_roll_speed'] = np.nan
    df['average_roll_angle'] = np.nan

    all_frequencies = []
    movies = os.listdir(base_path)
    movies = ["mov54"]
    all_freqs = []
    for movie in movies:
        movie_dir = os.path.join(base_path, movie)
        points_3D_path = os.path.join(movie_dir, 'points_3D_smoothed_ensemble_best.npy')
        if os.path.isfile(points_3D_path):
            try:
                FA = FlightAnalysis(points_3D_path)
                left_freq, right_freq = FA.wingbit_frequencies_left, FA.wingbit_frequencies_right
                average = (left_freq + right_freq)/2
                body_speed = np.mean(FA.body_speed)
                average_roll_speed = FA.average_roll_speed
                average_roll_angle = FA.average_roll_angle
                all_freqs.append(average)
                print(movie, int(left_freq), int(right_freq), int(average), flush=True)

                # Find the row in the DataFrame corresponding to this movie
                movie_num = int(movie.replace('mov', ''))  # Assuming movie folder names like 'mov53'
                df.loc[df['movie_num'] == movie_num, 'frequency_right'] = right_freq
                df.loc[df['movie_num'] == movie_num, 'frequency_left'] = left_freq
                df.loc[df['movie_num'] == movie_num, 'average_frequency'] = average
                df.loc[df['movie_num'] == movie_num, 'body_speed'] = body_speed
                df.loc[df['movie_num'] == movie_num, 'average_roll_speed'] = average_roll_speed
                df.loc[df['movie_num'] == movie_num, 'average_roll_angle'] = average_roll_angle

                print(f"{movie}: Avg freq = {average}, body speed = {body_speed}, "
                      f"average roll speed = {average_roll_speed}, average roll angle = {average_roll_angle}")

            except Exception as e:
                print(f"exception occurred: {e} in movie {movie}")
        # Save the updated DataFrame to a new CSV
    updated_csv_path = os.path.splitext(csv_path)[0] + '_updated2.csv'
    df.to_csv(updated_csv_path, index=False)

    print(all_freqs)
    all_freqs  = np.array(all_freqs)
    print(np.nanmean(all_freqs))


def save_movies_data_to_hdf5(base_path, output_hdf5_path, smooth=True, one_h5_for_all=False):
    # Ensure the directory exists
    if not os.path.isdir(base_path):
        print(f"Directory {base_path} does not exist")
        return

    if os.path.isfile(output_hdf5_path) and one_h5_for_all:
        print(f"File {output_hdf5_path} exists")
        return

    # Filter out directories not starting with 'mov'
    movies = [dir for dir in os.listdir(base_path) if dir.startswith('mov')]
    # movies = sorted(movies, key=lambda x: int(x.replace('mov', '')))
    # movies = ["mov101"]
    if one_h5_for_all:
        # Create or open the single HDF5 file
        with h5py.File(output_hdf5_path, 'w') as hdf:
            for movie in movies:
                movie_dir = os.path.join(base_path, movie)
                points_3D_path = os.path.join(movie_dir, 'points_3D_smoothed_ensemble_best_method.npy') if smooth else os.path.join(movie_dir, 'points_3D_ensemble_best_method.npy')

                if os.path.isfile(points_3D_path):
                    try:
                        FA = FlightAnalysis(points_3D_path, create_html=True)  # Assuming FlightAnalysis is properly defined

                        # Create a group for this movie
                        group = hdf.create_group(movie)

                        start_frame = FA.first_y_body_frame
                        end_frame = FA.end_frame

                        # Store attributes as datasets within the group
                        for attr_name, attr_value in FA.__dict__.items():
                            if hasattr(attr_value, '__len__'):  # Check for array-like objects or lists
                                group.create_dataset(attr_name, data=attr_value)

                        print(f"Data saved for {movie}")
                    except Exception as e:
                        print(f"Exception occurred while processing {movie}: {e}")
                else:
                    print(f"Missing data for {movie}")
    else:
        # Create an HDF5 file for each movie
        # movies = ['mov53']
        for movie in movies:
            movie_dir = os.path.join(base_path, movie)
            points_3D_path = os.path.join(movie_dir,
                                          'points_3D_smoothed_ensemble_best_method.npy') if smooth else os.path.join(movie_dir,
                                                                                                              'points_3D_ensemble_best_method.npy')

            if os.path.isfile(points_3D_path):
                # try:
                    FA = FlightAnalysis(points_3D_path, create_html=True)  # Assuming FlightAnalysis is properly defined

                    name = f'{movie}_analysis_smoothed.h5' if smooth else f'{movie}_analysis.h5'

                    # Define the output HDF5 file path for the movie
                    movie_hdf5_path = os.path.join(movie_dir, name)

                    # Create the HDF5 file for this movie
                    with h5py.File(movie_hdf5_path, 'w') as hdf:
                        # Store attributes as datasets within the group
                        for attr_name, attr_value in FA.__dict__.items():
                            if attr_name in ["left_half_wingbits", "right_half_wingbits",
                                             "left_full_wingbits", "right_full_wingbits"]:
                                wing_bit_part = attr_value
                                kind = "half" if "half" in attr_name else "full"
                                group = hdf.create_group(attr_name)  # Create a group for half_wingbits
                                for i, obj in enumerate(wing_bit_part):
                                    obj_group = group.create_group(f'{kind}_wingbit_{i}')
                                    for sub_attr_name, sub_attr_value in obj.__dict__.items():
                                        # Create a dataset inside the group for each attribute of the object
                                        obj_group.create_dataset(sub_attr_name, data=sub_attr_value)
                            else:
                                if hasattr(attr_value, '__len__'):
                                    # print(attr_name) # Check for array-like objects or lists
                                    hdf.create_dataset(attr_name, data=attr_value)
                                elif isinstance(attr_value, (int, float)):
                                    # Check for int or float attributes
                                    hdf.create_dataset(attr_name, data=attr_value)

                    print(f"Data saved for {movie} in {movie_hdf5_path}")
                # except Exception as e:
                #     print(f"Exception occurred while processing {movie}: {e}")
            else:
                print(f"Missing data for {movie}")

    print(f"All data saved to {output_hdf5_path}" if one_h5_for_all else "All data saved to individual movie HDF5 files")



def add_nans(original_array, num_of_nans):
    # Create an array of NaNs with size M, 3
    nan_frames = np.full((num_of_nans, 3), np.nan)
    # Concatenate the NaN frames with the original array
    new_array = np.vstack((nan_frames, original_array))
    return new_array

def compare_autocorrelations_before_and_after_dark(input_hdf5_path, T=20):
    dir = os.path.dirname(input_hdf5_path)
    fig = go.Figure()

    with h5py.File(input_hdf5_path, 'r') as hdf:
        movies = list(hdf.keys())
        for movie in movies:
            print(movie)
            group = hdf[movie]
            x_body = group['x_body'][:]
            y_body = group['y_body'][:]
            z_body = group['z_body'][:]

            first_analysed_frame = int(group['first_analysed_frame'][()])
            first_y_body_frame = int(group['first_y_body_frame'][()])
            start_frame = first_y_body_frame + first_analysed_frame

            end_frame_second_part = int(group['end_frame'][()]) + first_analysed_frame
            end_frame_first_part = T * 16

            if start_frame + 100 >= end_frame_first_part or len(x_body) < 400:
                continue

            # first part
            x_body_first_part = x_body[start_frame:end_frame_first_part]
            y_body_first_part = y_body[start_frame:end_frame_first_part]
            z_body_first_part = z_body[start_frame:end_frame_first_part]
            AC_first = FlightAnalysis.get_auto_correlation_x_body(x_body_first_part)

            # second part
            x_body_second_part = x_body[end_frame_first_part:end_frame_second_part]
            y_body_second_part = y_body[end_frame_first_part:end_frame_second_part]
            z_body_second_part = z_body[end_frame_first_part:end_frame_second_part]
            AC_second = FlightAnalysis.get_auto_correlation_x_body(x_body_second_part)

            fig.add_trace(go.Scatter(x=list(np.arange(len(AC_first)) / 16), y=AC_first,
                                     mode='lines', name=f'{movie} First Part',
                                     line=dict(color='red')))
            fig.add_trace(go.Scatter(x=list(np.arange(len(AC_second)) / 16), y=AC_second,
                                     mode='lines', name=f'{movie} Second Part',
                                     line=dict(color='blue')))

    fig.update_layout(title='Autocorrelation Before and After Dark for Each Movie',
                      xaxis_title='ms',
                      yaxis_title='Autocorrelation',
                      legend_title='Movies')
    html_out_path = os.path.join(dir, 'autocorrelation_plot.html')
    fig.write_html(html_out_path)





if __name__ == '__main__':
    movie = 104
    # points_path = rf"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov{movie}\points_3D_smoothed_ensemble_best.npy"
    # FlightAnalysis(points_path, create_html=False, find_auto_correlation=False, show_phi=False)
    # calculate_auto_correlation_roni_movies()
    # plot_auto_correlations()


    # output_hdf5_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\all_movies_data.h5"
    # # compare_autocorrelations_before_and_after_dark(output_hdf5_path)
    #
    # # plot_movie_html(1)
    # # plot_movie_html(2)    # plot_movie_html(3)
    # # plot_movie_html(4)
    base_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys"
    base_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\arranged movies"
    # base_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster"
    # csv_path = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 dark disturbance\from cluster\summary_results.csv"
    # # get_frequencies_from_all(base_path, csv_path)
    output_hdf5_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\all_movies_data_not_smoothed.h5"
    # output_hdf5_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\all_movies_data_smoothed.h5"
    # save_movies_data_to_hdf5(base_path, output_hdf5_path, smooth=True, one_h5_for_all=False)
    save_movies_data_to_hdf5(base_path, output_hdf5_path=output_hdf5_path, smooth=True, one_h5_for_all=False)

