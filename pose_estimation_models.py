import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Add, MaxPooling2D, Concatenate, Lambda,Reshape
import tensorflow as tf

from tensorflow.keras.optimizers import Adam


def basic_nn(img_size, num_output_channels, filters=64, num_blocks=2, kernel_size=3):
    x_in = Input(img_size, name="x_in")
    dilation_rate = 2
    encoder = encoder2d_atrous((img_size[0], img_size[1], img_size[2]), filters, num_blocks, kernel_size, dilation_rate)
    encoder.summary()
    decoder = decoder2d((encoder.output_shape[1], encoder.output_shape[2], encoder.output_shape[3]),
                         num_output_channels, filters, num_blocks, kernel_size)
    decoder.summary()

    x_out = decoder(encoder(x_in))

    net = Model(inputs=x_in, outputs=x_out, name="basic_nn")
    net.summary()
    net.compile(optimizer=Adam(learning_rate=0.001),
                loss="mean_squared_error")
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


def two_wings_net(img_size, num_output_channels, filters=64, num_blocks=2, kernel_size=3):
    x_in = Input(img_size, name="x_in")
    num_wings = 2
    num_time_channels = img_size[2] - 2
    # decoder gets an image of shape (M, N, num_time_channels + 1 wing)
    shared_encoder = encoder2d_atrous((img_size[0], img_size[1], num_time_channels + 1),
                                      filters, num_blocks, kernel_size)
    shared_encoder.summary()

    # decoder gets 2 encoder outputs, [wing 1, wing 2] and outputs the points for wing 1 (7 points)
    shared_decoder = decoder2d(
        (shared_encoder.output_shape[1], shared_encoder.output_shape[2],
         (num_wings + 1) * shared_encoder.output_shape[3]),
        num_output_channels // num_wings, filters, num_blocks, kernel_size)
    shared_decoder.summary()

    # todo a more general input
    x_in_wing_1 = Lambda(lambda x: tf.gather(x, [0, 1, 2, 3], axis=-1), name="lambda_1")(x_in)
    x_in_wing_2 = Lambda(lambda x: tf.gather(x, [0, 1, 2, 4], axis=-1), name="lambda_2")(x_in)

    # get inputs through the encoder and get a latents space representation
    code_out_1 = shared_encoder(x_in_wing_1)
    code_out_2 = shared_encoder(x_in_wing_2)

    merged_1 = Concatenate()([code_out_1, code_out_2])
    merged_2 = Concatenate()([code_out_2, code_out_1])

    # send to encoder: encoder_i = [wing_i, wing_j]
    # map_out_1 = shared_decoder(Concatenate()([code_out_1, code_out_2]))
    # map_out_2 = shared_decoder(Concatenate()([code_out_2, code_out_1]))

    map_out_1 = shared_decoder(Concatenate()([code_out_1, merged_1]))
    map_out_2 = shared_decoder(Concatenate()([code_out_2, merged_2]))

    # arrange output
    x_maps_merge = Concatenate()([map_out_1, map_out_2])

    # get the model
    net = Model(inputs=x_in, outputs=x_maps_merge, name="two_wings_net")
    net.summary()
    net.compile(optimizer=Adam(amsgrad=False),
                loss="mean_squared_error")
    return net


def pretrained_per_wing_vgg(img_size, num_output_channels, filters=64, num_blocks=2, kernel_size=3):
    from keras.applications.vgg16 import VGG16
    input_layer = Input(img_size, name="x_in")
    pre_trained_model = tf.keras.applications.VGG16(input_shape=(img_size[0], img_size[1], 3),
                                          weights='imagenet', include_top=False)

    encoder = Conv2D(3, (3, 3), activation='relu', padding='same', name='conv_layer')(input_layer)

    k = 9
    selected_layers = pre_trained_model.layers[:k]

    # Create a new input layer for the selected layers
    for i, layer in enumerate(selected_layers):
        encoder = layer(encoder)

    encoder = Model(inputs=input_layer, outputs=encoder, name="Encoder2DAtrous")

    for layer in encoder.layers[2:]:
        layer.trainable = False

    encoder.summary()

    decoder = decoder2d((encoder.output_shape[1], encoder.output_shape[2], encoder.output_shape[3]),
                         num_output_channels, filters, num_blocks, kernel_size)

    x_out = decoder(encoder(input_layer))

    net = Model(inputs=input_layer, outputs=x_out, name="pretrained_cnn")
    net.summary()
    net.compile(optimizer=Adam(learning_rate=0.001),
                loss="mean_squared_error")
    return net

def ed3d(img_size, num_output_channels, filters=64, num_blocks=2, kernel_size=3):
    x_in = Input(img_size, name="x_in")  # image size should be (M, M, num_channels * num cameras)
    num_cameras = 4
    # encoder encodes 1 image at a time
    shared_encoder = encoder2d_atrous((img_size[0], img_size[1], img_size[2] // num_cameras),
                                      filters, num_blocks, kernel_size)
    shared_encoder.summary()

    # for average
    # shared_decoder = decoder2d(
    #    (shared_encoder.output_shape[1], shared_encoder.output_shape[2], 2 * shared_encoder.output_shape[3])
    #    , num_output_channels, filters, num_blocks, kernel_size)
    # for cat

    # decoder accepts 1 encoder output concatenated with all other cameras encoders output so 1 + num cams
    shared_decoder = decoder2d(
        (shared_encoder.output_shape[1], shared_encoder.output_shape[2],
         (1 + num_cameras) * shared_encoder.output_shape[3]),
         num_output_channels // num_cameras, filters, num_blocks, kernel_size)
    shared_decoder.summary()

    # spliting input of 12 channels to 3 different cameras
    x_in_split_1 = Lambda(lambda x: x[..., 0:4], name="lambda_1")(x_in)
    x_in_split_2 = Lambda(lambda x: x[..., 4:8], name="lambda_2")(x_in)
    x_in_split_3 = Lambda(lambda x: x[..., 8:12], name="lambda_3")(x_in)
    x_in_split_4 = Lambda(lambda x: x[..., 12:16], name="lambda_4")(x_in)


    # when using wing masks add 2 more images (wings) per camera
    # x_in_split_1 = Lambda(lambda x: x[..., 0:5])(x_in)
    # x_in_split_2 = Lambda(lambda x: x[..., 5:10])(x_in)
    # x_in_split_3 = Lambda(lambda x: x[..., 10:15])(x_in)

    # x_in_split_1 = Lambda(lambda x: x[..., 0:3])(x_in)
    # x_in_split_2 = Lambda(lambda x: x[..., 3:6])(x_in)
    # x_in_split_3 = Lambda(lambda x: x[..., 6:9])(x_in)

    # different outputs of encoder
    code_out_1 = shared_encoder(x_in_split_1)
    code_out_2 = shared_encoder(x_in_split_2)
    code_out_3 = shared_encoder(x_in_split_3)
    code_out_4 = shared_encoder(x_in_split_4)

    # x_code_avg = Average()([code_out_1,code_out_2,code_out_3])
    # map_out_1 = shared_decoder(Concatenate()([code_out_1,x_code_avg]))
    # map_out_2 = shared_decoder(Concatenate()([code_out_2,x_code_avg]))
    # map_out_3 = shared_decoder(Concatenate()([code_out_3,x_code_avg]))

    # concatenated output of the 3 different encoders
    x_code_merge = Concatenate()([code_out_1, code_out_2, code_out_3, code_out_4])

    # prepare encoder's input as camera + concatenated latent vector of all cameras
    map_out_1 = shared_decoder(Concatenate()([code_out_1, x_code_merge]))
    map_out_2 = shared_decoder(Concatenate()([code_out_2, x_code_merge]))
    map_out_3 = shared_decoder(Concatenate()([code_out_3, x_code_merge]))
    map_out_4 = shared_decoder(Concatenate()([code_out_4, x_code_merge]))

    # merging all the encoders outputs, meaning we get a (M, M, num_pnts_per_wing * num_cams) confmaps
    x_maps_merge = Concatenate()([map_out_1, map_out_2, map_out_3, map_out_4])

    net = Model(inputs=x_in, outputs=x_maps_merge, name="ed3d")
    net.summary()
    net.compile(optimizer=Adam(amsgrad=False),
                loss="mean_squared_error")
    # loss="categorical_crossentropy")
    return net

