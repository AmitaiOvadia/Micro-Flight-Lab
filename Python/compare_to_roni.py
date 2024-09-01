import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.widgets import Slider, Button
from matplotlib import colors
import sklearn
from sklearn.preprocessing import normalize
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def row_wize_dot(arr1, arr2):
    dot = np.sum(arr1 * arr2, axis=1)
    return dot


def do_savgol_filter(array, window_size, order):
    if array.ndim == 1:
        array = array[:, np.newaxis]
    smoothed_array = np.zeros_like(array)
    for axis in range(array.shape[-1]):
        smoothed_array[:, axis] = savgol_filter(np.copy(array[:, axis]), window_size, order)
    return smoothed_array.squeeze()


class DataComparison:
    def __init__(self, mov_num, path_my_data, path_roni_data, smoothed, smooth_like_roni=False):
        self.frame_rate = 16000
        self.mov_num = mov_num
        if smoothed:
            self.path_my_data = os.path.join(path_my_data, "all_movies_data_smoothed.h5")
            self.path_roni_data = os.path.join(path_roni_data, "smoothed")
        else:
            self.path_my_data = os.path.join(path_my_data, "all_movies_data_not_smoothed.h5")
            self.path_roni_data = os.path.join(path_roni_data, "not smoothed")
        self.my_data = None
        self.roni_data = None
        self.smooth_like_roni = smooth_like_roni
        self.load_my_data()
        self.load_roni_data()

    def load_my_data(self):
        with h5py.File(self.path_my_data, 'r') as hdf:
            group = hdf[f"mov{self.mov_num}"]
            self.my_data = {
                'points_3D': group['points_3D'][:],
                'x_body': group['x_body'][:],
                'y_body': group['y_body'][:],
                'z_body': group['z_body'][:],
                'frames': np.arange(len(group['x_body'][:])),
                'yaw_angle': group['yaw_angle'][:].squeeze() + 360,
                'pitch_angle': group['pitch_angle'][:].squeeze(),
                'roll_angle': group['roll_angle'][:].squeeze(),
                'center_of_mass': group['center_of_mass'][:],
                'CM_velocity': group['CM_velocity'][:],
                'left_wing_span': group['left_wing_span'][:],
                'right_wing_span': group['right_wing_span'][:],
                'left_wing_CM': group['left_wing_CM'][:],
                'right_wing_CM': group['right_wing_CM'][:],
                'stroke_planes': group['stroke_planes'][:, :-1],
                'wings_phi_left': group['wings_phi_left'][:].squeeze(),
                'wings_theta_left': group['wings_theta_left'][:].squeeze(),
                'wings_psi_left': group['wings_psi_left'][:].squeeze() + 180,
                'wings_phi_right': group['wings_phi_right'][:].squeeze(),
                'wings_theta_right': group['wings_theta_right'][:].squeeze(),
                'wings_psi_right': group['wings_psi_right'][:].squeeze() + 0,
                'left_wing_chord': group['left_wing_chord'][:],
                'right_wing_chord': group['right_wing_chord'][:],
                'wings_tips_left': group['wings_tips_left'][:],
                'wings_tips_right': group['wings_tips_right'][:],
                'all_2_planes': group['all_2_planes'][:],
            }
            if self.smooth_like_roni:
                self.my_data['center_of_mass'] = do_savgol_filter(self.my_data['center_of_mass'], window_size=73*3, order=3)
                # body angles
                self.my_data['yaw_angle'] = do_savgol_filter(self.my_data['yaw_angle'], window_size=15, order=2)
                self.my_data['pitch_angle'] = do_savgol_filter(self.my_data['pitch_angle'], window_size=15, order=2)
                self.my_data['roll_angle'] = do_savgol_filter(self.my_data['roll_angle'], window_size=15, order=2)
                # wings
                self.my_data['wings_phi_left'] = do_savgol_filter(self.my_data['wings_phi_left'], window_size=15, order=2)
                self.my_data['wings_psi_left'] = do_savgol_filter(self.my_data['wings_psi_left'], window_size=15, order=2)
                self.my_data['wings_theta_left'] = do_savgol_filter(self.my_data['wings_theta_left'], window_size=15, order=2)

                self.my_data['wings_phi_right'] = do_savgol_filter(self.my_data['wings_phi_right'], window_size=15,
                                                                  order=2)
                self.my_data['wings_psi_right'] = do_savgol_filter(self.my_data['wings_psi_right'], window_size=15,
                                                                  order=2)
                self.my_data['wings_theta_right'] = do_savgol_filter(self.my_data['wings_theta_right'], window_size=15,
                                                                    order=2)


    def read_specific_columns(self, csv_path, column_names):
        return pd.read_csv(csv_path, usecols=column_names).to_numpy()

    def load_roni_data(self):
        base_path = os.path.join(self.path_roni_data, f"mov{self.mov_num}")
        self.roni_data = {
            'frames': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['frames']).astype(int).squeeze() - 1,
            'x_body': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['X_x_body', 'X_y_body', 'X_z_body']),
            'y_body': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['Y_x_body', 'Y_y_body', 'Y_z_body']),
            'z_body': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['Z_x_body', 'Z_y_body', 'Z_z_body']),
            'yaw_body': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_body.csv"), ['yaw_body']).squeeze(),
            'pitch_body': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_body.csv"), ['pitch_body']).squeeze(),
            'roll_body': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_body.csv"), ['roll_body']).squeeze(),
            'center_of_mass': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_body.csv"), ['CM_real_x_body', 'CM_real_y_body', 'CM_real_z_body']),
            # 'CM_velocity': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_body.csv"), ['CM_real_x_body_dot_dot_smth', 'CM_real_y_body_dot_dot_smth', 'CM_real_z_body_dot_dot_smth']),
            'left_wing_span': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['span_x_lw', 'span_y_lw', 'span_z_lw']),
            'right_wing_span': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['span_x_rw', 'span_y_rw', 'span_z_rw']),
            'stroke_planes': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['strkPlan_x_body', 'strkPlan_y_body', 'strkPlan_z_body']),
            'phi_left': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_wing.csv"), ['phi_lw']).squeeze(),
            'theta_left': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_wing.csv"), ['theta_lw']).squeeze(),
            'psi_left': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_wing.csv"), ['psi_lw']).squeeze(),
            'phi_right': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_wing.csv"), ['phi_rw']).squeeze(),
            'theta_right': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_wing.csv"), ['theta_rw']).squeeze(),
            'psi_right': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_wing.csv"), ['psi_rw']).squeeze(),
            'left_wing_chord': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['chord_x_lw', 'chord_y_lw', 'chord_z_lw']),
            'right_wing_chord': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['chord_x_rw', 'chord_y_rw', 'chord_z_rw']),
            # 'left_wing_tips': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['tip_x_lw', 'tip_y_lw', 'tip_z_lw']),
            # 'right_wing_tips': self.read_specific_columns(os.path.join(base_path, f"mov{self.mov_num}_vectors.csv"), ['tip_x_rw', 'tip_y_rw', 'tip_z_rw'])
        }

    def plot_vectors(self, my_data, roni_data, vector_name):
        labels = ['X', 'Y', 'Z']
        convert_to_ms = self.frame_rate / 1000
        fig = go.Figure()
        for i, label in enumerate(labels):
            # fig.add_trace(go.Scatter(
            #     x=roni_data['frames'][:] ,
            #     y=roni_data[:, i],
            #     mode='lines',
            #     name=f'Roni {label} {vector_name}'
            # ))
            fig.add_trace(go.Scatter(
                x=np.arange(len(self.my_data)) ,
                y=my_data[:, i],
                mode='lines+markers',
                name=f'My {label} {vector_name}',
                # line=dict(dash='dash')
            ))
        fig.update_layout(
            title=f'{vector_name} Vectors',
            xaxis_title='Milliseconds',
            yaxis_title=f'{vector_name} Coordinate',
            legend_title='Legend'
        )
        html_file_name = f'{vector_name}_vectors.html'
        fig.write_html(html_file_name)  # Save the figure as HTML file
        print(f'Saved: {html_file_name}')
        fig.show()

    def compare_body_vectors(self):
        self.plot_vectors(self.my_data['x_body'], self.roni_data['x_body'], "X Body")
        self.plot_vectors(self.my_data['y_body'], self.roni_data['y_body'], "Y Body")
        self.plot_vectors(self.my_data['z_body'], self.roni_data['z_body'], "Z Body")

    def compare_body_angles(self):
        angles = ['yaw', 'pitch', 'roll']
        convert_to_ms = self.frame_rate / 1000
        for angle in angles:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.my_data['frames'],
                y=self.my_data[f'{angle}_angle'],
                mode='lines',
                name=f'My {angle.capitalize()}',
                # line=dict(dash='dash')
            ))
            fig.update_layout(
                title=f'{angle.capitalize()} Body Angle',
                xaxis_title='Milliseconds',
                yaxis_title=f'{angle.capitalize()} Angle (degrees)',
                legend_title='Legend'
            )
            fig.write_html(f'{angle}_body_angle.html')

    def plot_body_pitch_phi_right_phi_left(self):
        # Assuming frame_rate and frames are defined and appropriately calculated in your class
        convert_to_ms = self.frame_rate / 1000

        # Create a figure with subplots
        fig = go.Figure()

        # Plot Pitch Body Angle
        fig.add_trace(go.Scatter(
            x=self.my_data['frames'],
            y=self.my_data['pitch_angle'],
            mode='lines',
            name='My Pitch Body Angle',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=self.my_data['frames'] ,
            y=self.my_data['wings_phi_left'],
            mode='lines',
            name='My Left Wing Phi Angle',
            line=dict(color='red')
        ))

        fig.add_trace(go.Scatter(
            x=self.my_data['frames'] ,
            y=self.my_data['wings_phi_right'],
            mode='lines',
            name='My Right Wing Phi Angle',
            line=dict(color='green')
        ))

        # Update the layout to add titles and axis labels
        fig.update_layout(
            title='Body Pitch and Wing Phi Angles',
            xaxis_title='Milliseconds',
            yaxis_title='Angle (degrees)',
            legend_title='Legend'
        )

        # Save the plot as an HTML file
        html_file_name = 'body_pitch_phi_wings.html'
        fig.write_html(html_file_name)
        print(f'Saved: {html_file_name}')

        # Display the plot
        # fig.show()

    def plot_body_roll_phi_right_phi_left(self):
        # Assuming frame_rate and frames are defined and appropriately calculated in your class
        convert_to_ms = self.frame_rate / 1000

        # Create a figure with subplots
        fig = go.Figure()

        # Plot Phi Angle for Left Wing
        fig.add_trace(go.Scatter(
            x=self.my_data['frames'] ,
            y=self.my_data['wings_phi_left'],
            mode='lines',
            name='My Left Wing Phi Angle',
            line=dict(color='red')
        ))

        # Plot Phi Angle for Right Wing
        fig.add_trace(go.Scatter(
            x=self.my_data['frames'] ,
            y=self.my_data['wings_phi_right'],
            mode='lines',
            name='My Right Wing Phi Angle',
            line=dict(color='green')
        ))

        # Plot Roll Body Angle
        fig.add_trace(go.Scatter(
            x=self.my_data['frames'] ,
            y=self.my_data['roll_angle'],
            mode='lines',
            name='My Roll Body Angle',
            line=dict(color='blue')
        ))

        # Update the layout to add titles and axis labels
        fig.update_layout(
            title='Wing Phi Angles and Body Roll',
            xaxis_title='Milliseconds',
            yaxis_title='Angle (degrees)',
            legend_title='Legend'
        )

        # Save the plot as an HTML file
        html_file_name = 'body_roll_phi_wings.html'
        fig.write_html(html_file_name)
        print(f'Saved: {html_file_name}')

    def plot_body_pitch_theta_left_theta_right(self):
        # Assuming frame_rate and frames are defined and appropriately calculated in your class
        convert_to_ms = self.frame_rate / 1000

        # Create a figure with subplots
        fig = go.Figure()

        # Plot Body Pitch Angle
        fig.add_trace(go.Scatter(
            x=self.my_data['frames'] ,
            y=self.my_data['pitch_angle'],
            mode='lines',
            name='My Body Pitch Angle',
            line=dict(color='blue')
        ))

        # Plot Theta Angle for Left Wing
        fig.add_trace(go.Scatter(
            x=self.my_data['frames'] ,
            y=self.my_data['wings_theta_left'],
            mode='lines',
            name='My Left Wing Theta Angle',
            line=dict(color='red')
        ))

        # Plot Theta Angle for Right Wing
        fig.add_trace(go.Scatter(
            x=self.my_data['frames'] ,
            y=self.my_data['wings_theta_right'],
            mode='lines',
            name='My Right Wing Theta Angle',
            line=dict(color='green')
        ))

        # Update the layout to add titles and axis labels
        fig.update_layout(
            title='Body Pitch and Wing Theta Angles',
            xaxis_title='Milliseconds',
            yaxis_title='Angle (degrees)',
            legend_title='Legend'
        )

        # Save the plot as an HTML file
        html_file_name = 'body_pitch_theta_wings.html'
        fig.write_html(html_file_name)
        print(f'Saved: {html_file_name}')


    def compare_CM(self):
        self.plot_vectors(self.my_data['center_of_mass'], self.roni_data['center_of_mass'], "Center of Mass")

    def compare_CM_velocity(self):
        self.plot_vectors(self.my_data['CM_velocity'], self.roni_data['CM_velocity'], "Center of Mass Velocity")

    def compare_stroke_planes(self):
        self.plot_vectors(self.my_data['stroke_planes'], self.roni_data['stroke_planes'], "Stroke Planes")

    def compare_wings_span(self, wing='left'):
        if wing == 'left':
            self.plot_vectors(self.my_data['left_wing_span'], self.roni_data['left_wing_span'], "Left Wing Span")
        else:
            self.plot_vectors(self.my_data['right_wing_span'], self.roni_data['right_wing_span'], "Right Wing Span")

    def compare_wings_angles(self, wing='left'):
        angles = ['phi', 'theta', 'psi']
        for angle in angles:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.roni_data['frames'],
                y=self.roni_data[f'{angle}_{wing}'],
                mode='lines',
                name=f'Roni {angle.capitalize()} {wing.capitalize()} Wing'
            ))
            fig.add_trace(go.Scatter(
                x=self.my_data['frames'],
                y=self.my_data[f'wings_{angle}_{wing}'],
                mode='lines',
                name=f'My {angle.capitalize()} {wing.capitalize()} Wing',
                # line=dict(dash='dash')
            ))
            fig.update_layout(
                title=f'{angle.capitalize()} {wing.capitalize()} Wing Angle',
                xaxis_title='Frame',
                yaxis_title=f'{angle.capitalize()} Angle (degrees)',
                legend_title='Legend'
            )
            html_file_name = f'{angle}_{wing}_wing_angle.html'
            fig.write_html(html_file_name)  # Save the figure as HTML file
            print(f'Saved: {html_file_name}')
            # fig.show()

    def theta_vs_phi(self, wing='left'):
        # for angle in ['phi', 'theta', 'psi']:
        plt.figure()
        add = 0
        start = 280 + add
        end = 489 + add
        plt.plot(self.my_data[f'wings_phi_{wing}'][start:end], self.my_data[f'wings_psi_{wing}'][start:end], linestyle='--', label=f'')
        plt.plot(self.roni_data[f'phi_{wing}'][start:end], self.roni_data[f'psi_{wing}'][start:end],
                 linestyle='-', label=f'')
        plt.xlabel('phi')
        plt.ylabel(f'theta Angle (degrees)')
        plt.title(f'theta_vs_phi Wing Angle')
        plt.legend()
        plt.axis('equal')
        plt.show()

    def compare_wings_chords(self, wing='left'):
        if wing == 'left':
            self.plot_vectors(self.my_data['left_wing_chord'], self.roni_data['left_wing_chord'], "Left Wing Chord")
        else:
            self.plot_vectors(self.my_data['right_wing_chord'], self.roni_data['right_wing_chord'], "Right Wing Chord")

    def compare_wings_tip(self, wing='left'):
        if wing == 'left':
            self.plot_vectors(self.my_data['wings_tips_left'], self.roni_data['left_wing_tips'], "Left Wing Tips")
        else:
            self.plot_vectors(self.my_data['wings_tips_right'], self.roni_data['right_wing_tips'], "Right Wing Tips")

    @staticmethod
    def row_wize_normalization(array):
        valid_rows = ~np.isnan(array).any(axis=1)
        valid_array = array[valid_rows]
        l2_norms = np.linalg.norm(valid_array, axis=1, keepdims=True)
        l2_norms[l2_norms == 0] = 1
        normalized_valid_array = valid_array / l2_norms
        normalized_array = np.full_like(array, np.nan)
        normalized_array[valid_rows] = normalized_valid_array
        return normalized_array

    def visualize_3D_comparison(self):
        # Assuming points is your (N, M, 3) array
        roni_frames = self.roni_data['frames'][:]
        points = self.my_data['points_3D'][:]
        num_frames = len(points)

        # stroke planes
        my_stroke_planes = self.my_data['stroke_planes'][:]

        # center of masss
        my_CM = self.my_data['center_of_mass'][:]
        attribute = 'center_of_mass'
        roni_CM = self.get_ronis_array(attribute, my_CM, roni_frames)

        # body vectors
        my_x_body = self.my_data['x_body'][:]
        my_y_body = self.my_data['y_body'][:]
        my_z_body = self.my_data['z_body'][:]

        check = row_wize_dot(my_z_body, my_z_body)
        mean = np.nanmean(check)

        # roni's body vectors
        roni_x_body = self.get_ronis_array('x_body', my_x_body, roni_frames)
        roni_y_body = self.get_ronis_array('y_body', my_y_body, roni_frames)
        roni_z_body = self.get_ronis_array('z_body', my_z_body, roni_frames)

        # wing vectors
        my_left_wing_span = self.my_data['left_wing_span'][:]
        my_right_wing_span = self.my_data['right_wing_span'][:]
        my_left_wing_chord = self.my_data['left_wing_chord'][:]
        my_right_wing_chord = self.my_data['right_wing_chord'][:]
        my_left_wing_normal = DataComparison.row_wize_normalization(np.cross(my_left_wing_chord, my_left_wing_span, axis=1))
        my_right_wing_normal = DataComparison.row_wize_normalization(np.cross(my_right_wing_chord, my_right_wing_span, axis=1))

        # roni's wing vectors
        roni_left_wing_span = self.get_ronis_array('left_wing_span', my_left_wing_span, roni_frames)
        roni_right_wing_span = self.get_ronis_array('right_wing_span', my_right_wing_span, roni_frames)
        roni_left_wing_chord = self.get_ronis_array('left_wing_chord', my_left_wing_chord, roni_frames)
        roni_right_wing_chord = self.get_ronis_array('right_wing_chord', my_right_wing_chord, roni_frames)
        roni_left_wing_normal = DataComparison.row_wize_normalization(np.cross(roni_left_wing_chord, roni_left_wing_span, axis=1))
        roni_right_wing_normal = DataComparison.row_wize_normalization(np.cross(roni_right_wing_chord, roni_right_wing_span, axis=1))

        # wing tips
        my_left_wing_tip = self.my_data['wings_tips_left'][:]
        my_right_wing_tip = self.my_data['wings_tips_right'][:]

        # wings cm
        my_left_wing_CM = self.my_data['left_wing_CM'][:]
        my_right_wing_CM = self.my_data['right_wing_CM'][:]

        # roni's cm
        left_dist_from_tip_to_cm = np.linalg.norm(my_left_wing_tip - my_left_wing_CM, axis=1)
        roni_left_wing_CM = my_left_wing_tip - left_dist_from_tip_to_cm[:, np.newaxis] * roni_left_wing_span

        right_dist_from_tip_to_cm = np.linalg.norm(my_right_wing_tip - my_right_wing_CM, axis=1)
        roni_right_wing_CM = my_right_wing_tip - right_dist_from_tip_to_cm[:, np.newaxis] * roni_right_wing_span

        points = self.my_data['points_3D'][:1000]
        # Calculate the limits of the plot
        x_min = np.nanmin(points[:, :, 0])
        y_min = np.nanmin(points[:, :, 1])
        z_min = np.nanmin(points[:, :, 2])

        x_max = np.nanmax(points[:, :, 0])
        y_max = np.nanmax(points[:, :, 1])
        z_max = np.nanmax(points[:, :, 2])

        # Create a color array
        my_points_to_show = np.array([my_left_wing_CM, my_right_wing_CM, my_CM, my_left_wing_tip, my_right_wing_tip])
        roni_points_to_show = np.array([roni_left_wing_CM, roni_right_wing_CM, roni_CM])
        num_points = points.shape[1]
        color_array = colors.hsv_to_rgb(np.column_stack((np.linspace(0, 1, num_points), np.ones((num_points, 2)))))

        fig = plt.figure(figsize=(20, 20))  # Adjust the figure size here
        ax = fig.add_subplot(111, projection='3d')

        # Set the limits of the plot
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        # Define the connections between points
        connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6),
                       (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (8, 14),
                       # (7, 15),
                       (16, 17)]

        # Create the slider
        axframe = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(axframe, 'Frame', 0, len(points) - 1, valinit=0, valstep=1)

        def add_quiver_axes(ax, origin, x_vec, y_vec, z_vec, scale=0.001, labels=None,  color='r'):
            x, y, z = origin
            if x_vec is not None:
                ax.quiver(x, y, z, x_vec[0], x_vec[1], x_vec[2], length=scale, color=color)
            if y_vec is not None:
                ax.quiver(x, y, z, y_vec[0], y_vec[1], y_vec[2], length=scale, color=color)
            if z_vec is not None:
                ax.quiver(x, y, z, z_vec[0], z_vec[1], z_vec[2], length=scale, color=color)

            # Add optional labels if provided
            if labels:
                if x_vec is not None:
                    ax.text(x + x_vec[0] * scale, y + x_vec[1] * scale, z + x_vec[2] * scale, labels[0], color=color)
                if y_vec is not None:
                    ax.text(x + y_vec[0] * scale, y + y_vec[1] * scale, z + y_vec[2] * scale, labels[1], color=color)
                if z_vec is not None:
                    ax.text(x + z_vec[0] * scale, y + z_vec[1] * scale, z + z_vec[2] * scale, labels[2], color=color)

        def plot_plane(ax, center, normal, size=0.005, color='g', alpha=0.25):
            d = size / 2

            # Create two orthogonal vectors on the plane
            if np.abs(normal[2]) > 0.99:
                u = np.array([1.0, 0.0, 0.0])
            else:
                u = np.cross(normal, [0.0, 0.0, 1.0])
            u = u / np.linalg.norm(u)

            v = np.cross(normal, u)
            v = v / np.linalg.norm(v)

            # Calculate the corners of the plane
            corners = np.array([
                center + d * (u + v),
                center + d * (u - v),
                center + d * (-u - v),
                center + d * (-u + v),
            ])

            # Create the polygon for the plane
            vertices = [corners]
            plane = Poly3DCollection(vertices, color=color, alpha=alpha)
            ax.add_collection3d(plane)

        def update(val, plot_head_tail_points=False, plot_all_my_points=True, show_roni=True, show_me=True):
            ax.cla()  # Clear current axes
            frame = int(slider.val)

            if plot_all_my_points:
                for i in range(num_points):
                    if i not in [7, 15]:
                        ax.scatter(points[frame, i, 0], points[frame, i, 1], points[frame, i, 2], c=color_array[i])
                for i, j in connections:
                    ax.plot(points[frame, [i, j], 0], points[frame, [i, j], 1], points[frame, [i, j], 2], c='k')

            # Plot points
            if plot_head_tail_points:
                ax.scatter(points[frame, -1, 0], points[frame, -1, 1], points[frame, -1, 2], c=color_array[0])
                ax.scatter(points[frame, -2, 0], points[frame, -2, 1], points[frame, -2, 2], c=color_array[0])
                ax.plot(points[frame, [-1, -2], 0], points[frame, [-1, -2], 1], points[frame, [-1, -2], 2], c='k')

            # Plot wing tips
            ax.scatter(my_left_wing_tip[frame, 0], my_left_wing_tip[frame, 1], my_left_wing_tip[frame, 2],
                       c=color_array[0])
            ax.scatter(my_right_wing_tip[frame, 0], my_right_wing_tip[frame, 1], my_right_wing_tip[frame, 2],
                       c=color_array[0])

            # Plot the center of mass
            my_cm_x, my_cm_y, my_cm_z = my_CM[frame]
            ax.scatter(my_cm_x, my_cm_y, my_cm_z, c=color_array[0])

            # Add the stroke plane
            plot_plane(ax, my_CM[frame], my_stroke_planes[frame])

            if show_me:
                add_quiver_axes(ax, (my_cm_x, my_cm_y, my_cm_z), my_x_body[frame], my_y_body[frame], my_z_body[frame],
                                color='r', labels=['Xb', 'Yb', 'Zb'])
                add_quiver_axes(ax, my_left_wing_CM[frame], my_left_wing_span[frame], my_left_wing_chord[frame], None,
                                color='r', labels=['Span', 'Chord'])
                add_quiver_axes(ax, my_right_wing_CM[frame], my_right_wing_span[frame], my_right_wing_chord[frame], None,
                                color='r', labels=['Span', 'Chord'])
            if show_roni:
                # Plot Roni's center of mass
                roni_cm_x, roni_cm_y, roni_cm_z = roni_CM[frame]
                ax.scatter(roni_cm_x, roni_cm_y, roni_cm_z, c=color_array[8])
                add_quiver_axes(ax, (roni_cm_x, roni_cm_y, roni_cm_z), roni_x_body[frame], roni_y_body[frame],
                                roni_z_body[frame], color='b', labels=['Xb', 'Yb', 'Zb'])
                add_quiver_axes(ax, roni_left_wing_CM[frame], roni_left_wing_span[frame], roni_left_wing_chord[frame], None,
                                color='b', labels=['Span', 'Chord'])
                add_quiver_axes(ax, roni_right_wing_CM[frame], roni_right_wing_span[frame], roni_right_wing_chord[frame],
                                None, color='b', labels=['Span', 'Chord'])
            zoom_factor = 0.5  # Adjust as needed
            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * zoom_factor
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2
            mid_z = (z_max + z_min) / 2

            ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
            ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
            ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

            ax.set_box_aspect([1, 1, 1])

            fig.canvas.draw_idle()

        slider.on_changed(update)

        # Function to handle keyboard events
        def handle_key_event(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, slider.valmax))
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, slider.valmin))

        fig.canvas.mpl_connect('key_press_event', handle_key_event)

        # Initial plot
        update(0)
        plt.show()

    def get_ronis_array(self, attribute, my_array, roni_frames):
        roni_CM = np.full(my_array.shape, np.nan)
        roni_CM[roni_frames] = self.roni_data[attribute][:]
        return roni_CM


def export_to_csv_and_organize(hdf5_path, output_base_dir, movies):
    with h5py.File(hdf5_path, 'r') as file:
        for movie in movies:
            movie_dir = os.path.join(output_base_dir, movie)
            os.makedirs(movie_dir, exist_ok=True)  # Create directory for the movie if it doesn't exist

            for data_attr in file[movie].keys():
                data = file[movie][data_attr][()]
                df = pd.DataFrame(data)

                if 'header' in file[movie][data_attr].attrs:
                    df.columns = file[movie][data_attr].attrs['header']

                attribute_name = data_attr.split('_')[0]  # Assuming the attribute name format is '{name}_raw'
                csv_filename = f"{movie}_{attribute_name}.csv"
                df.to_csv(os.path.join(movie_dir, csv_filename), index=False)


if __name__ == "__main__":
    # hdf5_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\Roni analisys\cliped_2023_08_09_60ms.hdf5"
    # output_base_dir = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\Roni analisys\not smoothed"
    # movies = ["mov78", "mov101", "mov104"]
    # export_to_csv_and_organize(hdf5_path, output_base_dir, movies)

    path_my_data = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys"
    path_roni_data = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\Roni analisys"
    movies = [78, 101, 104]
    mov_num = movies[1]
    smoothed = True

    comparison = DataComparison(mov_num, path_my_data, path_roni_data, smoothed, smooth_like_roni=smoothed)
    comparison.plot_body_pitch_phi_right_phi_left()
    comparison.plot_body_roll_phi_right_phi_left()
    comparison.plot_body_pitch_theta_left_theta_right()
    #
    comparison.visualize_3D_comparison()
    comparison.compare_CM()
    comparison.compare_CM_velocity()
    comparison.compare_body_vectors()
    comparison.compare_body_angles()
    comparison.compare_stroke_planes()
    comparison.compare_wings_span(wing='left')
    comparison.theta_vs_phi()
    comparison.compare_wings_angles(wing='right')
    comparison.compare_wings_angles(wing='left')
    comparison.compare_wings_chords(wing='right')
    comparison.compare_wings_tip(wing='right')
