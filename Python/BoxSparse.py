import h5py
import numpy as np
from scipy.sparse import csr_matrix


class BoxSparse:
    def __init__(self, box_path=None, shape=None):
        # Convert entire box to sparse upon initialization or lazily as needed
        self.num_channels = None
        self.num_frames, self.num_cams, self.height, self.width, self.num_times_channels = None, None, None, None, None
        self.shape = None
        self.box_path = box_path
        if box_path is not None:
            self.sparse_frames = self.convert_to_sparse(self.get_box())
        else:
            self.shape = shape
            assert self.shape != None, "shape is None"
            box = np.zeros(shape, dtype=np.int8)
            self.sparse_frames = self.convert_to_sparse(box)


    def get_box(self):
        box = h5py.File(self.box_path, "r")["/box"]
        if len(box.shape) == 5:
            print("box already has wings masks")
            return box
        box = np.transpose(box, (0, 3, 2, 1))
        x1 = np.expand_dims(box[:, :, :, 0:3], axis=1)
        x2 = np.expand_dims(box[:, :, :, 3:6], axis=1)
        x3 = np.expand_dims(box[:, :, :, 6:9], axis=1)
        x4 = np.expand_dims(box[:, :, :, 9:12], axis=1)
        box = np.concatenate((x1, x2, x3, x4), axis=1)
        return box

    def convert_to_sparse(self, box):
        box = BoxSparse.preprocess_box(box)
        self.shape = box.shape
        self.num_frames, self.num_cams, self.height, self.width, self.num_times_channels = box.shape
        sparse_box = {}
        self.num_channels = self.num_times_channels
        # Adjust num_channels to include 2 additional channels
        if self.num_times_channels > 1:
            self.num_channels = self.num_times_channels + 2

        for frame_idx in range(self.num_frames):
            for camera_idx in range(self.num_cams):
                for channel_idx in range(self.num_channels):
                    key = (frame_idx, camera_idx, channel_idx)
                    if channel_idx < self.num_times_channels:
                        # Convert existing channels to sparse matrices
                        frame_channel = box[frame_idx, camera_idx, :, :, channel_idx]
                        channel_sparse = csr_matrix(frame_channel)
                    else:
                        # Handle new channels filled with zeros
                        # Create a sparse matrix with the shape of the channel, but don't store any zeros
                        channel_sparse = csr_matrix((self.height, self.width), dtype=box.dtype)

                    sparse_box[key] = channel_sparse
        return sparse_box

    def get_frame_camera_channel_dense(self, frame_idx, camera_idx, channel_idx):
        # Retrieve a specific frame-camera-channel combination as a dense array
        key = (frame_idx, camera_idx, channel_idx)
        if key in self.sparse_frames:
            return self.sparse_frames[key].toarray()
        else:
            # Handle the case where the requested frame-camera-channel combination does not exist
            print("Requested frame-camera-channel combination does not exist.")
            return None

    def set_frame_camera_channel_dense(self, frame_idx, camera_idx, channel_idx, image):
        """Update the specified frame, camera, and channel with a new image."""
        # Ensure the input image is a dense 2D array and has the correct dimensions
        assert image.shape == (self.height, self.width), "Image dimensions must match frame dimensions."

        # Convert the image to a sparse representation
        image_sparse = csr_matrix(image)

        # Generate the key for the sparse_frames dictionary
        key = (frame_idx, camera_idx, channel_idx)

        # Update the sparse representation with the new image
        self.sparse_frames[key] = image_sparse

    @staticmethod
    def preprocess_box(box):
        # Add singleton dim for single train_images
        if box.ndim == 3:
            box = box[None, ...]
        if box.dtype == "uint8" or np.max(box) > 1:
            box = box.astype("float32") / 255
        return box

    def retrieve_dense_box(self):
        """Retrieve the entire box in its dense form."""
        # Initialize an empty array for the dense box with an additional dimension for channels
        dense_box = np.zeros((self.num_frames, self.num_cams, self.height, self.width, self.num_channels),
                             dtype=np.float32)

        # Iterate over all frames, cameras, and channels
        for frame_idx in range(self.num_frames):
            for camera_idx in range(self.num_cams):
                for channel_idx in range(self.num_channels):
                    key = (frame_idx, camera_idx, channel_idx)
                    # Check if the current frame-camera-channel combination exists in the sparse representation
                    if key in self.sparse_frames:
                        # Convert the sparse matrix to a dense array and store it in the dense box
                        dense_box[frame_idx, camera_idx, :, :, channel_idx] = self.sparse_frames[key].toarray()
                    # Note: If a key is not found, it implies the channel is meant to be all zeros,
                    # which is already the default value in the initialized dense_box array.

        return dense_box

    def get_camera_dense(self, camera_idx, channels=None, frames=None):
        """
        Retrieve frames for a specific camera in dense format.

        :param camera_idx: Index of the camera.
        :param frames: Optional list of frame indexes to retrieve. If None, all frames are retrieved.
        :return: Numpy array of shape (selected_num_frames, height, width, num_channels) for the specified camera.
        """
        # Initialize an empty array for the dense camera data
        if channels is None:
            channels = range(self.num_channels)

        if frames is None:
            frames = range(self.num_frames)
        else:
            # Validate the provided frame indexes
            frames = [f for f in frames if f < self.num_frames and f >= 0]

        selected_num_frames = len(frames)
        dense_camera = np.zeros((selected_num_frames, self.height, self.width, len(channels)), dtype=np.float32)

        # Iterate over the specified frames and channels for the specific camera
        for i, frame_idx in enumerate(frames):
            for j, channel_idx in enumerate(channels):
                key = (frame_idx, camera_idx, channel_idx)
                if key in self.sparse_frames:
                    # Convert the sparse matrix to a dense array and assign it to the correct channel
                    dense_camera[i, :, :, j] = self.sparse_frames[key].toarray()

        return dense_camera

    def set_camera_dense(self, camera_idx, dense_camera_data, channels=None, frames=None):
        """
        Update the sparse representation for specific frames of a specific camera based on dense input.

        :param camera_idx: Index of the camera to update.
        :param dense_camera_data: Dense array representing the new data for the specified frames of the camera.
                                  Shape should match (selected_num_frames, height, width, len(channels)).
        :param frames: Optional list of frame indexes to update. If None, updates all frames.
        """
        if channels is None:
            channels = range(self.num_channels)

        if frames is None:
            frames = range(self.num_frames)
        else:
            # Validate the provided frame indexes
            frames = [f for f in frames if self.num_frames > f >= 0]

        # Ensure the input dense data has the correct dimensions
        assert dense_camera_data.shape == (len(frames), self.height, self.width, len(channels)), \
            "Input dense data must match the shape of (selected_num_frames, height, width, len(channels))."

        # Iterate over the specified frames and channels
        for i, frame_idx in enumerate(frames):
            for j, channel_idx in enumerate(channels):
                # Generate the key for the sparse_frames dictionary
                key = (frame_idx, camera_idx, channel_idx)

                # Extract the corresponding channel data for the current frame from the dense array
                channel_data = dense_camera_data[i, :, :, j]

                # Convert the channel data to a sparse representation
                channel_sparse = csr_matrix(channel_data)

                # Update the sparse representation with the new data
                self.sparse_frames[key] = channel_sparse

