from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Add, MaxPooling2D
import tensorflow as tf

from tensorflow.keras.optimizers import Adam


def basic_nn(img_size, num_output_channels, filters=64, num_blocks=2, kernel_size=3,
             loss_function="mean_squared_error", dilation_rate=2):
    x_in = Input(img_size, name="x_in")
    encoder = encoder2d_atrous((img_size[0], img_size[1], img_size[2]), filters, num_blocks, kernel_size, dilation_rate)
    encoder.summary()
    decoder = decoder2d(
        (encoder.output_shape[1], encoder.output_shape[2], encoder.output_shape[3])
        , num_output_channels, filters, num_blocks, kernel_size)
    decoder.summary()

    x_out = decoder(encoder(x_in))

    net = Model(inputs=x_in, outputs=x_out, name="basic_nn")
    net.summary()
    net.compile(optimizer=Adam(learning_rate=0.001),
                loss=loss_function)
    return net


def encoder2d_atrous(img_size, filters, num_blocks, kernel_size, dilation_rate=2):
    x_in = Input(img_size)
    dilation_rate = (dilation_rate, dilation_rate)
    for block_ind in range(num_blocks):
        if block_ind == 0:
            x_out = Conv2D(filters * (2 ** block_ind), kernel_size, dilation_rate=dilation_rate,
                                         padding="same", activation="relu")(x_in)

        else:
            x_out = Conv2D(filters * (2 ** block_ind), kernel_size, dilation_rate=dilation_rate,
                                         padding="same", activation="relu")(x_out)

        x_out = Conv2D(filters * (2 ** block_ind), kernel_size, dilation_rate=dilation_rate,
                                     padding="same", activation="relu")(x_out)

        x_out = Conv2D(filters * (2 ** block_ind), kernel_size, dilation_rate=dilation_rate,
                                     padding="same", activation="linear")(x_out)

        x_out = MaxPooling2D(pool_size=2, strides=2, padding="same")(x_out)
        x_out = Activation('relu')(x_out)
        x_out = Dropout(0.5)(x_out)

    x_out = Conv2D(filters * (2 ** num_blocks), kernel_size, dilation_rate=dilation_rate,
                                 padding="same", activation="relu")(x_out)
    x_out = Conv2D(filters * (2 ** num_blocks), kernel_size, dilation_rate=dilation_rate,
                                 padding="same", activation="relu")(x_out)
    x_out = Conv2D(filters * (2 ** num_blocks), kernel_size, dilation_rate=dilation_rate,
                                 padding="same", activation="relu")(x_out)
    x_out = Dropout(0.5)(x_out)
    return Model(inputs=x_in, outputs=x_out, name="Encoder2DAtrous")


def decoder2d(input_shape, num_output_channels, filters, num_blocks, kernel_size):
    x_in = Input(input_shape)
    for block_ind in range(num_blocks - 1, 0, -1):
        if block_ind == (num_blocks - 1):
            x_out = Conv2DTranspose(filters * (2 ** (block_ind)), kernel_size=kernel_size, strides=2,
                                                  padding="same", activation="relu",
                                                  kernel_initializer="glorot_normal")(x_in)
        else:
            x_out = Conv2DTranspose(filters * (2 ** (block_ind)), kernel_size=kernel_size, strides=2,
                                                  padding="same", activation="relu",
                                                  kernel_initializer="glorot_normal")(x_out)

        x_out = Conv2D(filters * (2 ** (block_ind)), kernel_size=kernel_size, padding="same", activation="relu")(x_out)
        x_out = Conv2D(filters * (2 ** (block_ind)), kernel_size=kernel_size, padding="same", activation="relu")(x_out)

    x_out = Conv2DTranspose(num_output_channels, kernel_size=kernel_size, strides=2, padding="same",
                            activation="linear",
                            kernel_initializer="glorot_normal")(x_out)

    return Model(inputs=x_in, outputs=x_out, name="Decoder2D")


# def residual_bottleneck_module(x_in, output_filters=32, bottleneck_factor=2, prefix="res", activation="relu",
#                                initializer="glorot_normal"):
#     # Get input shape and channels
#     in_shape = K.int_shape(x_in)
#     input_filters = in_shape[3]
#
#     # Bottleneck filters are proportional to the output filters
#     bottleneck_filters = output_filters // bottleneck_factor
#
#     # Bottleneck block
#     x = Conv2D(filters=bottleneck_filters, kernel_size=1, padding="same", activation=activation,
#                kernel_initializer=initializer, name=prefix + "_Conv1")(x_in)
#     x = Conv2D(filters=bottleneck_filters, kernel_size=3, padding="same", activation=activation,
#                kernel_initializer=initializer, name=prefix + "_Conv2")(x)
#     x = Conv2D(filters=output_filters, kernel_size=1, padding="same", activation=activation,
#                kernel_initializer=initializer, name=prefix + "_Conv3")(x)
#
#     # 1x1 conv if input channels are different from output channels
#     if output_filters != input_filters:
#         x_in = Conv2D(filters=output_filters, kernel_size=1, padding="same", activation=activation,
#                       kernel_initializer=initializer, name=prefix + "_ConvSkip")(x_in)
#
#     # Residual connection
#     x = Add(name=prefix + "_AddRes")([x_in, x])
#
#     return x
