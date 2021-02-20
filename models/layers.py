import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, LeakyReLU,\
     ReLU, Conv2DTranspose
import tensorflow_addons as tfa


#WEIGHT_INITIALIZER = "glorot_uniform"
WEIGHT_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.02)


def get_norm_layer(norm_type):
    if norm_type.lower() == "batchnorm":
        return BatchNormalization()
    elif norm_type.lower() == "instancenorm":
        #return InstanceNormalization()
        return tfa.layers.InstanceNormalization(axis=-1, 
                                                center=True, 
                                                scale=True)
    else:
        raise ValueError("arg `norm_type` has to be either batchnorm "
                         "or instancenorm. What you specified is "
                         "{}".format(norm_type))


def get_activation(activation):
    if activation.lower() == "relu":
        return tf.keras.layers.ReLU()
    elif activation.lower() == "lrelu":
        return tf.keras.layers.LeakyReLU(0.01)
    elif activation.lower() == "tanh":
        return tf.keras.activations.tanh
    elif activation.lower() == "sigmoid":
        return tf.keras.activations.sigmoid
    else:
        raise ValueError("arg `norm_type` has to be either relu "
                         "or tanh. What you specified is "
                         "{}".format(norm_type))

                        
class ResidualBlock(Layer):
    # Define Re block
    def __init__(self,
                 filters, 
                 size=3, 
                 strides=1, 
                 padding="same",
                 norm_type="instancenorm",
                 name="residual_block"):

        super(ResidualBlock, self).__init__(name=name)
        self.norm_type = norm_type
        self.size = size
        self.conv2d_1 = Conv2D(filters, 
                               size, 
                               strides, 
                               padding=padding,
                               use_bias=False,
                               kernel_initializer=WEIGHT_INITIALIZER)
        if self.norm_type:
            self.norm_layer_1 = get_norm_layer(norm_type)
        self.ReLU = ReLU()
        self.conv2d_2 = Conv2D(filters,
                               size,
                               strides,
                               padding=padding,
                               use_bias=False,
                               kernel_initializer=WEIGHT_INITIALIZER)
        if self.norm_type:
            self.norm_layer_2 = get_norm_layer(norm_type)

    def call(self, inputs):
        x = self.conv2d_1(inputs)
        if self.norm_type:
            self.norm_layer_1(x)
        x = self.ReLU(x)
        x = self.conv2d_2(x)
        if self.norm_type:
            self.norm_layer_2(x)
        return x + inputs


class Downsample(Layer):
    """
     Conv2D -> BatchNorm(or InstanceNorm) -> LeakyReLU
     Args:
        filters: number of filters
           size: filter size
      norm_type: normalization type. Either "batchnorm", "instancenorm" or None
           name: name of the layer
    Return:
        Downsample functional model
    """

    def __init__(self, 
                 filters, 
                 size,
                 strides=1,
                 padding="same",
                 norm_type="instancenorm",
                 activation="relu",
                 name="downsample"):

        super(Downsample, self).__init__(name=name)
        self.norm_type = norm_type
        use_bias = False
        if self.norm_type:
            self.norm_layer = get_norm_layer(norm_type)
        else:
            use_bias = True
        self.conv2d = Conv2D(filters,
                             size,
                             strides=strides,
                             padding=padding,
                             use_bias=use_bias,
                             kernel_initializer=WEIGHT_INITIALIZER)
        self.activation = get_activation(activation)

    def call(self, inputs):
        x = self.conv2d(inputs)
        if self.norm_type:
            x = self.norm_layer(x)
        x = self.activation(x)

        return x


class Upsample(Layer):
    """
    Conv2DTranspose -> BatchNorm(or InstanceNorm) -> Dropout -> ReLU
     Args:
        filters: number of filters
           size: filter size
      norm_type: normalization type. Either "batchnorm", "instancenorm" or None
  apply_dropout: If True, apply the dropout layer
           name: name of the layer
    Return:
        Upsample functional model
    """
    def __init__(self, 
                 filters, 
                 size,
                 strides,
                 padding,
                 norm_type="instancenorm",
                 apply_dropout=False,
                 activation="relu",
                 name="upsample"):

        super(Upsample, self).__init__(name=name)
        self.norm_type = norm_type
        use_bias = False
        if self.norm_type:
            self.norm_layer = get_norm_layer(norm_type)
        else:
            use_bias = True
        self.apply_dropout = apply_dropout
        self.conv2dtranspose = Conv2DTranspose(filters,
                                               size,
                                               strides=strides,
                                               padding=padding,
                                               use_bias=use_bias,
                                               kernel_initializer=WEIGHT_INITIALIZER)
        if apply_dropout:
            self.dropout = Dropout(0.5)
        self.activation = get_activation(activation)

    def call(self, inputs):
        x = self.conv2dtranspose(inputs)
        if self.norm_type:
            x = self.norm_layer(x)
        if self.apply_dropout:
            x = self.dropout(x)
        x = self.activation(x)

        return x