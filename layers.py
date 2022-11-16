from packaging.version import parse as parse_version
from tensorflow.keras.engine import Layer
from tensorflow.keras.utils import conv_utils

class UpSampling2D():
    """Upsampling layer for 2D inputs.
    Repeats the rows and columns of the data
    by size[0] and size[1] respectively with bilinear interpolation.
    # Arguments
        size: int, or tuple of 2 integers.
            The upsampling factors for rows and columns.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        interpolation: A string,
            one of 'nearest' (default), 'bilinear', or 'bicubic'
    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    # Output shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, upsampled_rows, upsampled_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, upsampled_rows, upsampled_cols)`
    """

    @interfaces.legacy_upsampling2d_support
    def __init__(self, size=(2, 2), data_format=None, interpolation='nearest', **kwargs):
        super(UpSampling2D, self).__init__(**kwargs)
        # Update to K.normalize_data_format after keras 2.2.0
        if parse_version(tensorflow.keras.__version__) > parse_version("2.2.0"):
            self.data_format = K.normalize_data_format(data_format)
        else:
            self.data_format = conv_utils.normalize_data_format(data_format)

        self.interpolation = interpolation
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs):
        return resize_images(inputs, self.size[0], self.size[1],
                             self.interpolation, self.data_format)

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(UpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _find_maxima(x):

    x = K.cast(x, K.floatx())

    col_max = K.max(x, axis=1)
    row_max = K.max(x, axis=2)

    maxima = K.max(col_max, 1)
    maxima = K.expand_dims(maxima, -2)

    cols = K.cast(K.argmax(col_max, -2), K.floatx())
    rows = K.cast(K.argmax(row_max, -2), K.floatx())
    cols = K.expand_dims(cols, -2)
    rows = K.expand_dims(rows, -2)

    # maxima = K.concatenate([rows, cols, maxima], -2) # y, x, val
    maxima = K.concatenate([cols, rows, maxima], -2) # x, y, val

    return maxima


def find_maxima(x, data_format):
    """Finds the 2D maxima contained in a 4D tensor.
    # Arguments
        x: Tensor or variable.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    if data_format == 'channels_first':
        x = permute_dimensions(x, [0, 2, 3, 1])
        x = _find_maxima(x)
        x = permute_dimensions(x, [0, 2, 1])
        return x
    elif data_format == 'channels_last':
        x = _find_maxima(x)
        return x
    else:
        raise ValueError('Invalid data_format:', data_format)
