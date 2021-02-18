import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input

from .layers import Downsample, Upsample, ResidualBlock


INPUT_SHAPE = (128, 128, 3)

class Discriminator(Model):
    """
    Referred from StarGAN paper(https://arxiv.org/abs/1711.09020).
    [Network Architecture]
    Input Layer:
        (h, w, 3) → (h/2, w/2, 64) CONV-(N64, K4x4, S2, P1), Leaky ReLU
    Hidden Layer:
        (h/2, w/2, 64) → (h/4, w/4, 128)        CONV-(N128, K4x4, S2, P1), Leaky ReLU
        (h/4, w/4, 128) → (h/8, w/8, 256)       CONV-(N256, K4x4, S2, P1), Leaky ReLU
        (h/8, w/8, 256) → (h/16, w16 , 512)     CONV-(N512, K4x4, S2, P1), Leaky ReLU
        (h/16, w/16, 512) → (h/32, w/32, 1024)  CONV-(N1024, K4x4, S2, P1), Leaky ReLU
        (h/32, w/32, 1024) → (h/64, w/64, 2048) CONV-(N2048, K4x4, S2, P1), Leaky ReLU
    Output Layer:
        (Dsrc) (h/64, w/64 , 2048) → (h/64, w/64, 1) CONV-(N1, K3x3, S1, P1)
        (Dcls) (h/64, w/64 , 2048) → (1, 1, nd)      CONV-(N(nd), K h/64 x w/64 , S1, P0)
    Args:
        norm_type: normalization type. Either "batchnorm", "instancenorm" or None
        
    Return:
        Discriminator model
    """
    def __init__(self,
                 c_dim,
                 first_filters=64,
                 size=4,
                 norm_type=None,
                 img_size=128,
                 name="discriminator"):

        super(Discriminator, self).__init__(name=name)
        self.downsample_1 = Downsample(filters=first_filters, 
                                       size=size,
                                       strides=2,
                                       padding="same",
                                       norm_type=norm_type,
                                       activation="lrelu",
                                       name="d_downsample_1")
        self.downsample_2 = Downsample(filters=first_filters * 2, 
                                       size=size,
                                       strides=2,
                                       padding="same",
                                       norm_type=norm_type,
                                       activation="lrelu",
                                       name="d_downsample_2")
        self.downsample_3 = Downsample(filters=first_filters * 4, 
                                       size=size,
                                       strides=2,
                                       padding="same",
                                       norm_type=norm_type,
                                       activation="lrelu",
                                       name="d_downsample_3")
        self.downsample_4 = Downsample(filters=first_filters * 8, 
                                       size=size, 
                                       strides=2,
                                       padding="same",
                                       norm_type=norm_type,
                                       activation="lrelu",
                                       name="d_downsample_4")
        self.downsample_5 = Downsample(filters=first_filters * 16, 
                                       size=size, 
                                       strides=2,
                                       padding="same",
                                       norm_type=norm_type,
                                       activation="lrelu",
                                       name="d_downsample_5")
        self.downsample_6 = Downsample(filters=first_filters * 32, 
                                       size=size, 
                                       strides=2,
                                       padding="same",
                                       norm_type=norm_type,
                                       activation="lrelu",
                                       name="d_downsample_6")
        self.conv2d_real = Conv2D(filters=1, 
                                  kernel_size=3, 
                                  strides=1,
                                  padding="same",
                                  use_bias=False)
        self.conv2d_aux = Conv2D(filters=c_dim,
                                 kernel_size=img_size//64,
                                 strides=1,
                                 padding="valid",
                                 use_bias=False)

    def call(self, inputs):
        x = self.downsample_1(inputs)
        x = self.downsample_2(x) 
        x = self.downsample_3(x)
        x = self.downsample_4(x)
        x = self.downsample_5(x)
        x = self.downsample_6(x)
        out_real = self.conv2d_real(x)
        out_aux = self.conv2d_aux(x)

        return tf.reshape(out_real, [-1, out_real.shape[1], out_real.shape[2]]), tf.reshape(out_aux, [-1, out_aux.shape[-1]])

    def summary(self):
        x = Input(shape=INPUT_SHAPE)
        model = Model(inputs=[x], outputs=self.call(x), name=self.name)
        return model.summary()


class Generator(Model):
    """
    Referred from StarGAN paper(https://arxiv.org/abs/1711.09020).
    [Network Architecture]
    Down-sampling:
        (h, w, 3 + nc) → (h, w, 64)       CONV-(N64, K7x7, S1, P3), IN, ReLU
        (h, w, 64) → (h/2, w/2, 128)      CONV-(N128, K4x4, S2, P1), IN, ReLU
        (h/2, w/2, 128) → (h/4, w/4, 256) CONV-(N256, K4x4, S2, P1), IN, ReLU
    Bottleneck:
        (h/4, w/4, 256) → (h/4, w/4, 256) Residual Block: CONV-(N256, K3x3, S1, P1), IN, ReLU
        (h/4, w/4, 256) → (h/4, w/4, 256) Residual Block: CONV-(N256, K3x3, S1, P1), IN, ReLU
        (h/4, w/4, 256) → (h/4, w/4, 256) Residual Block: CONV-(N256, K3x3, S1, P1), IN, ReLU
        (h/4, w/4, 256) → (h/4, w/4, 256) Residual Block: CONV-(N256, K3x3, S1, P1), IN, ReLU
        (h/4, w/4, 256) → (h/4, w/4, 256) Residual Block: CONV-(N256, K3x3, S1, P1), IN, ReLU
        (h/4, w/4, 256) → (h/4, w/4, 256) Residual Block: CONV-(N256, K3x3, S1, P1), IN, ReLU
    Up-sampling:
        (h/4, w/4, 256) → (h/2, w/2, 128) DECONV-(N128, K4x4, S2, P1), IN, ReLU
        (h/2, w/2, 128) → (h, w, 64) DECONV-(N64, K4x4, S2, P1), IN, ReLU
        (h, w, 64) → (h, w, 3) CONV-(N3, K7x7, S1, P3), Tanh
    Args:
    output_channels: number of output channels
          norm_type: normalization type. Either "batchnorm", "instancenorm" or None
    Return:
        Generator model
    """
    def __init__(self,
                 c_dim=5,
                 first_filters=64,
                 output_channels=3,
                 norm_type="instancenorm", 
                 name="generator"):

        super(Generator, self).__init__(name=name)
        self.c_dim = c_dim
        self.downsample_1 = Downsample(filters=first_filters, 
                                       size=7,
                                       strides=1,
                                       padding="valid",
                                       norm_type=norm_type, 
                                       name="g_downsample_1")
        self.downsample_2 = Downsample(filters=first_filters*2, 
                                       size=4,
                                       strides=2,
                                       norm_type=norm_type, 
                                       name="g_downsample_2")
        self.downsample_3 = Downsample(filters=first_filters*4, 
                                       size=4,
                                       strides=2,
                                       norm_type=norm_type, 
                                       name="g_downsample_3")
        self.residualblock_1 = ResidualBlock(first_filters*4, name="g_residualblock_1")
        self.residualblock_2 = ResidualBlock(first_filters*4, name="g_residualblock_2")
        self.residualblock_3 = ResidualBlock(first_filters*4, name="g_residualblock_3")
        self.residualblock_4 = ResidualBlock(first_filters*4, name="g_residualblock_4")
        self.residualblock_5 = ResidualBlock(first_filters*4, name="g_residualblock_5")
        self.residualblock_6 = ResidualBlock(first_filters*4, name="g_residualblock_6")
        self.upsample_1 = Upsample(filters=first_filters*2, 
                                   size=4,
                                   strides=2,
                                   padding="same",
                                   norm_type=norm_type,
                                   name="g_upsample_1")
        self.upsample_2 = Upsample(filters=first_filters, 
                                   size=4,
                                   strides=2,
                                   padding="same",
                                   norm_type=norm_type,
                                   name="g_upsample_2")
        self.last_conv2d_color_mask = Conv2D(filters=output_channels, 
                                             kernel_size=7,
                                             strides=1,
                                             padding="valid",
                                             activation="tanh",
                                             name="g_last_conv2d_c_mask",
                                             use_bias=False)
        self.last_conv2d_attention_mask = Conv2D(filters=1,
                                                 kernel_size=7,
                                                 strides=1,
                                                 padding="valid",
                                                 activation="sigmoid",
                                                 name="g_last_conv2d_a_mask",
                                                 use_bias=False)

    def call(self, x, c_dim):
        # Convert the shape of 'c': (bs, c_dim) -> (bs, 1, 1, c_dim)
        c_dim = tf.cast(tf.reshape(c_dim, [-1, 1, 1, c_dim.shape[-1]]), tf.float32)
        c_dim = tf.tile(c_dim, [1, x.shape[1], x.shape[2], 1])
        # After oncatnating x and c, the shape is (bs, h, w, 3+c_dim)
        x = tf.concat([x, c_dim], axis=-1)
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        x = self.downsample_1(x)
        x = self.downsample_2(x)
        x = self.downsample_3(x)
        x = self.residualblock_1(x)
        x = self.residualblock_2(x)
        x = self.residualblock_3(x)
        x = self.residualblock_4(x)
        x = self.residualblock_5(x)
        x = self.residualblock_6(x)
        x = self.upsample_1(x)
        x = self.upsample_2(x)
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        color_mask = self.last_conv2d_color_mask(x)
        attn_mask = self.last_conv2d_attention_mask(x)

        return color_mask, attn_mask

    def summary(self):
        x = Input(shape=INPUT_SHAPE)
        c = Input(shape=(self.c_dim,))
        model = Model(inputs=[x, c], outputs=self.call(x, c), name=self.name)
        return model.summary()

def build_model(c_dim):
    generator = Generator(c_dim)
    discriminator = Discriminator(c_dim)
    
    tf.print("Check Generator's model architecture")
    generator.summary()

    tf.print("\n\n")

    tf.print("Check Discriminator's model architecture")
    discriminator.summary()

    return generator, discriminator


if __name__ == "__main__":
    # Test the shapes of the models
    c_dim = 17
    gen, disc = build_model(c_dim)
