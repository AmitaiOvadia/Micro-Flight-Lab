from constants import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Add, MaxPooling2D, Concatenate, Lambda, Reshape, \
                                    Activation, Dropout
import tensorflow as tf

from tensorflow.keras.optimizers import Adam


class Network:
    def __init__(self, config, image_size, number_of_output_channels):
        self.model_type = config['model type']
        self.image_size = image_size
        self.number_of_output_channels = number_of_output_channels
        self.num_base_filters = config["number of base filters"]
        self.num_blocks = config["number of encoder decoder blocks"]
        self.kernel_size = config["convolution kernel size"]
        self.learning_rate = config["learning rate"]
        self.optimizer = config["optimizer"]
        self.loss_function = config["loss_function"]
        self.dilation_rate = config["dilation rate"]
        self.dropout = config["dropout ratio"]
        self.model = self.config_model()

    def get_model(self):
        return self.model

    def config_model(self):
        if self.model_type == ALL_CAMS:
            model = self.ed3d()
        elif self.model_type == TWO_WINGS_TOGATHER:
            model = self.two_wings_net()
        else:
            model = self.basic_nn()
        return model

    def basic_nn(self):
        x_in = Input(self.image_size, name="x_in")
        encoder = self.encoder2d_atrous((self.image_size[0], self.image_size[1], self.image_size[2]),
                                         self.num_base_filters,
                                         self.num_blocks, self.kernel_size,
                                         self.dilation_rate,
                                         self.dropout)
        encoder.summary()
        decoder = self.decoder2d((encoder.output_shape[1], encoder.output_shape[2], encoder.output_shape[3]),
                            self.number_of_output_channels, self.num_base_filters, self.num_blocks, self.kernel_size)
        decoder.summary()

        x_out = decoder(encoder(x_in))

        net = Model(inputs=x_in, outputs=x_out, name="basic_nn")
        net.summary()
        net.compile(optimizer=Adam(learning_rate=self.learning_rate),
                    loss=self.loss_function)
        return net

    def two_wings_net(self):
        x_in = Input(self.image_size, name="x_in")
        num_wings = 2
        num_time_channels = self.image_size[2] - 2
        # decoder gets an image of shape (M, N, num_time_channels + 1 wing)
        shared_encoder = self.encoder2d_atrous((self.image_size[0], self.image_size[1], num_time_channels + 1),
                                          self.num_base_filters, self.num_blocks, self.kernel_size,
                                               self.dilation_rate,self.dropout)
        shared_encoder.summary()

        # decoder gets 2 encoder outputs, [wing 1, wing 2] and outputs the points for wing 1 (7 points)
        shared_decoder = self.decoder2d(
            (shared_encoder.output_shape[1], shared_encoder.output_shape[2],
             (num_wings + 0) * shared_encoder.output_shape[3]),
            self.number_of_output_channels // num_wings, self.num_base_filters, self.num_blocks, self.kernel_size)
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
        map_out_1 = shared_decoder(Concatenate()([code_out_1, code_out_2]))
        map_out_2 = shared_decoder(Concatenate()([code_out_2, code_out_1]))
        #
        # map_out_1 = shared_decoder(Concatenate()([code_out_1, merged_1]))
        # map_out_2 = shared_decoder(Concatenate()([code_out_2, merged_2]))

        # arrange output
        x_maps_merge = Concatenate()([map_out_1, map_out_2])

        # get the model
        net = Model(inputs=x_in, outputs=x_maps_merge, name="two_wings_net")
        net.summary()
        net.compile(optimizer=Adam(amsgrad=False),
                    loss=self.loss_function)
        return net

    def ed3d(self):
        x_in = Input(self.image_size, name="x_in")  # image size should be (M, M, num_channels * num cameras)
        num_cameras = 4
        # encoder encodes 1 image at a time
        shared_encoder = self.encoder2d_atrous((self.image_size[0], self.image_size[1],
                                                self.image_size[2] // num_cameras),
                                          self.num_base_filters, self.num_blocks, self.kernel_size, self.dilation_rate, self.dropout)
        shared_encoder.summary()

        # for average
        # shared_decoder = decoder2d(
        #    (shared_encoder.output_shape[1], shared_encoder.output_shape[2], 2 * shared_encoder.output_shape[3])
        #    , num_output_channels, filters, num_blocks, kernel_size)
        # for cat

        # decoder accepts 1 encoder output concatenated with all other cameras encoders output so 1 + num cams
        shared_decoder = self.decoder2d(
            (shared_encoder.output_shape[1], shared_encoder.output_shape[2],
             (1 + num_cameras) * shared_encoder.output_shape[3]),
            self.number_of_output_channels // num_cameras, self.num_base_filters, self.num_blocks, self.kernel_size)
        shared_decoder.summary()

        # spliting input of 12 channels to 3 different cameras
        x_in_split_1 = Lambda(lambda x: x[..., 0:4], name="lambda_1")(x_in)
        x_in_split_2 = Lambda(lambda x: x[..., 4:8], name="lambda_2")(x_in)
        x_in_split_3 = Lambda(lambda x: x[..., 8:12], name="lambda_3")(x_in)
        x_in_split_4 = Lambda(lambda x: x[..., 12:16], name="lambda_4")(x_in)

        # different outputs of encoder
        code_out_1 = shared_encoder(x_in_split_1)
        code_out_2 = shared_encoder(x_in_split_2)
        code_out_3 = shared_encoder(x_in_split_3)
        code_out_4 = shared_encoder(x_in_split_4)

        # concatenated output of the 3 different encoders
        x_code_merge = Concatenate()([code_out_1, code_out_2, code_out_3, code_out_4])

        # shorter latent vector : each map_out gets only the other encoded vectors
        # map_out_1 = shared_decoder(Concatenate()([code_out_1, code_out_2, code_out_3, code_out_4]))
        # map_out_2 = shared_decoder(Concatenate()([code_out_2, code_out_1, code_out_3, code_out_4]))
        # map_out_3 = shared_decoder(Concatenate()([code_out_3, code_out_1, code_out_2, code_out_4]))
        # map_out_4 = shared_decoder(Concatenate()([code_out_4, code_out_1, code_out_2, code_out_3]))

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
                    loss=self.loss_function)
        # loss="categorical_crossentropy")
        return net

    @staticmethod
    def encoder2d_atrous(img_size, filters, num_blocks, kernel_size, dilation_rate, dropout):
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
            x_out = Dropout(dropout)(x_out)

        x_out = Conv2D(filters * (2 ** num_blocks), kernel_size, dilation_rate=dilation_rate,
                       padding="same", activation="relu")(x_out)
        x_out = Conv2D(filters * (2 ** num_blocks), kernel_size, dilation_rate=dilation_rate,
                       padding="same", activation="relu")(x_out)
        x_out = Conv2D(filters * (2 ** num_blocks), kernel_size, dilation_rate=dilation_rate,
                       padding="same", activation="relu")(x_out)
        x_out = Dropout(dropout)(x_out)
        return Model(inputs=x_in, outputs=x_out, name="Encoder2DAtrous")

    @staticmethod
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

            x_out = Conv2D(filters * (2 ** (block_ind)), kernel_size=kernel_size, padding="same", activation="relu")(
                x_out)
            x_out = Conv2D(filters * (2 ** (block_ind)), kernel_size=kernel_size, padding="same", activation="relu")(
                x_out)

        x_out = Conv2DTranspose(num_output_channels, kernel_size=kernel_size, strides=2, padding="same",
                                activation="linear",
                                kernel_initializer="glorot_normal")(x_out)

        return Model(inputs=x_in, outputs=x_out, name="Decoder2D")
