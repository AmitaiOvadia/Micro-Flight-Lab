import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.widgets import Slider
from skimage import morphology


class Visualizer:
    @staticmethod
    def show_predictions_one_cam(box, points_2D):
        # Assuming movie is your 5D numpy array and points is your 3D array
        movie = Visualizer.get_display_box(box)
        points = points_2D[..., :2]

        # Assuming movie is your 5D numpy array and points is your 3D array
        camera = [0]  # initial camera as a one-element list

        def update(val):
            frame = slider.val
            ax.clear()  # clear the previous scatter points
            ax.imshow(movie[int(slider.val), camera[0]])
            colors = plt.cm.rainbow(np.linspace(0, 1, len(points[int(slider.val), camera[0]])))  # create a color array
            ax.scatter(*points[frame, camera[0]].T, edgecolors=colors, facecolors='none',
                       marker='o')  # scatter points on image
            plt.draw()

        def on_key_press(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, movie.shape[0] - 1))  # increment slider value
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, 0))  # decrement slider value
            elif event.key == 'up':
                camera[0] = min(camera[0] + 1, movie.shape[1] - 1)  # switch to next camera
                update(None)
            elif event.key == 'down':
                camera[0] = max(camera[0] - 1, 0)  # switch to previous camera
                update(None)

        fig, ax = plt.subplots(figsize=(10, 10))  # single camera view
        plt.subplots_adjust(bottom=0.2)  # make room for the slider

        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_ax, 'Frame', 0, movie.shape[0] - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event',
                               on_key_press)  # connect the key press event to the on_key_press function

        plt.show()

    @staticmethod
    def show_predictions_all_cams(box, points_2D):
        movie = Visualizer.get_display_box(box)
        # movie = box[..., [1, 1, 1]]
        points = points_2D[..., :2]

        # Assuming movie is your 5D numpy array and points is your 3D array
        def update(val):
            frame = int(slider.val)
            for i, ax in enumerate(axes.flat):
                ax.clear()
                ax.imshow(movie[frame, i])
                colors = plt.cm.rainbow(np.linspace(0, 1, len(points[frame, i])))  # create a color array
                ax.scatter(*points[frame, i].T, edgecolors=colors, facecolors='none', marker='o')  #
            plt.draw()

        def on_key_press(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, movie.shape[0] - 1))  # increment slider value
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, 0))  # decrement slider value

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid of camera views
        axes = axes.ravel()  # flatten the grid to easily iterate over it
        plt.subplots_adjust(bottom=0.2)  # make room for the slider

        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_ax, 'Frame', 0, movie.shape[0] - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event',
                               on_key_press)  # connect the key press event to the on_key_press function

        plt.show()

    @staticmethod
    def show_predictions_vs_reprojections_one_cam(box, points_2D, reprojections_2D):
        # Assuming movie is your 5D numpy array and points is your 3D array
        movie = Visualizer.get_display_box(box)
        points = points_2D[..., :2]
        reprojections = reprojections_2D[..., :2]

        # Assuming movie is your 5D numpy array and points is your 3D array
        camera = [0]  # initial camera as a one-element list

        def update(val):
            frame = slider.val
            ax.clear()  # clear the previous scatter points
            ax.imshow(movie[int(slider.val), camera[0]])
            colors = plt.cm.rainbow(np.linspace(0, 1, len(points[int(slider.val), camera[0]])))  # create a color array
            ax.scatter(*points[frame, camera[0]].T, edgecolors=colors, facecolors='none',
                       marker='o')  # scatter points on image
            ax.scatter(*reprojections[frame, camera[0]].T, edgecolors=colors, facecolors='none',
                       marker='d')  # scatter points on image
            for point, point_reprojected in zip(points[frame, camera[0]], reprojections[frame, camera[0]]):
                ax.plot(*zip(point, point_reprojected), color='yellow')
            plt.draw()

        def on_key_press(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, movie.shape[0] - 1))  # increment slider value
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, 0))  # decrement slider value
            elif event.key == 'up':
                camera[0] = min(camera[0] + 1, movie.shape[1] - 1)  # switch to next camera
                update(None)
            elif event.key == 'down':
                camera[0] = max(camera[0] - 1, 0)  # switch to previous camera
                update(None)

        fig, ax = plt.subplots(figsize=(10, 10))  # single camera view
        plt.subplots_adjust(bottom=0.2)  # make room for the slider

        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_ax, 'Frame', 0, movie.shape[0] - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event',
                               on_key_press)  # connect the key press event to the on_key_press function

        plt.show()

    @staticmethod
    def show_predictions_vs_reprejections_all_cams(box, points_2D, points_2D_reprojected):
        movie = Visualizer.get_display_box(box)
        points = points_2D[..., :2]
        points_reprojected = points_2D_reprojected[..., :2]

        # Assuming movie is your 5D numpy array and points is your 3D array
        def update(val):
            frame = int(slider.val)
            for i, ax in enumerate(axes.flat):
                ax.clear()
                ax.imshow(movie[frame, i])
                colors = plt.cm.rainbow(np.linspace(0, 1, len(points[frame, :])))  # create a color array for points_2D
                ax.scatter(*points[frame, i].T, edgecolors=colors, facecolors='none', marker='o')  # display points_2D
                ax.scatter(*points_reprojected[frame, i].T, edgecolors=colors, facecolors='none', marker='d')
                # Add a line between every corresponding point in points and points_reprojected
                for point, point_reprojected in zip(points[frame, i], points_reprojected[frame, i]):
                    ax.plot(*zip(point, point_reprojected), color='yellow')

            plt.draw()

        def on_key_press(event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, movie.shape[0] - 1))  # increment slider value
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, 0))  # decrement slider value

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid of camera views
        axes = axes.ravel()  # flatten the grid to easily iterate over it
        plt.subplots_adjust(bottom=0.2)  # make room for the slider

        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_ax, 'Frame', 0, movie.shape[0] - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        fig.canvas.mpl_connect('key_press_event',
                               on_key_press)  # connect the key press event to the on_key_press function

        plt.show()

    @staticmethod
    def show_points_in_3D(points):
        # Assuming points is your (N, M, 3) array
        # Calculate the limits of the plot
        x_min, y_min, z_min = points.min(axis=(0, 1))
        x_max, y_max, z_max = points.max(axis=(0, 1))

        # Create a color array
        num_points = points.shape[1]
        color_array = colors.hsv_to_rgb(np.column_stack((np.linspace(0, 1, num_points), np.ones((num_points, 2)))))

        fig = plt.figure(figsize=(20, 20))  # Adjust the figure size here
        ax = fig.add_subplot(111, projection='3d')

        # Set the limits of the plot
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        # Define the connections between points
        # connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0),
        #                (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 7),
        #                (14, 15)]
        connections = [(0,1), (1,2), (2,3),   (3,4),   (4,5),   (5,6),  (0,6),
                        (8,9), (9,10),(10,11), (11,12), (12,13), (13,14), (8,14),
                        (7, 15),
                        (16, 17)]

        # Create the slider
        axframe = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(axframe, 'Frame', 0, len(points) - 1, valinit=0, valstep=1)

        def update(val):
            ax.cla()  # Clear the current axes
            frame = int(slider.val)
            for i in range(num_points):
                ax.scatter(points[frame, i, 0], points[frame, i, 1], points[frame, i, 2], c=color_array[i])
            for i, j in connections:
                ax.plot(points[frame, [i, j], 0], points[frame, [i, j], 1], points[frame, [i, j], 2], c='k')
            # Reset the limits of the plot
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
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

    @staticmethod
    def get_display_box(box):
        masks = box[..., -2:]
        num_frames, num_cams, _, _, num_masks = masks.shape
        for frame in range(num_frames):
            for cam in range(num_cams):
                for wing in range(num_masks):
                    mask = masks[frame, cam, :, :, wing]
                    dilated = morphology.binary_dilation(mask)
                    eroded = morphology.binary_erosion(mask)
                    perimeters = dilated ^ eroded
                    masks[frame, cam, :, :, wing] = perimeters
        box[..., -2:] = masks
        box[..., -2:] += box[..., [1, 1]]
        movie = box[..., [1, 3, 4]]
        movie[movie > 1] = 1
        return movie
