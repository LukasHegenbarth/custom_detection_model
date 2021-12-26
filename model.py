import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tempfile

#input_dim = (896, 512)
input_dim = (1152, 640)
# input_dim = (640,384)
num_stages = 4
# num_blocks = [4,4,4,4]
num_blocks = [3, 4, 6, 3]
num_channels = [64, 128, 256, 512]
# group_width = 1
group_width = 8
# group_width = 16
num_classes = 600
num_anchors = 5


class BiFPN(layers.Layer):
    def __init__(self, in_channels):
        """Bi-directional feature pyramid network (BiFPN)
        Args:
          in_channels: (Variable) list of features' size of each layer from backbone
                        with [(width, channel)].
        e.g.
        if block 1,2,4,7,14 in MobileNetV2 is used,
        in_channels: [(10,160),(19,64),(38,32),(75,24),(150,32)]
        (ascending of width size)
        I make 'in_channels' with
        self.bb_size = [(output.shape.as_list()[1], output.shape.as_list()[3])
                            for output in self.backbone.outputs]
        """

        super(BiFPN, self).__init__()
        self.epsilon = 0.0001
        self.input_layer_cnt = len(in_channels)
        in_wd, in_ch = zip(*in_channels)

        self.td_weights = []
        self.out_weights = []
        self.td_convs = []
        self.out_convs = []

        self.out_weights.append(tf.random.normal([3]))
        self.out_convs.append(
            tf.keras.Sequential(
                [
                    layers.SeparableConv2D(in_ch[0], 3, padding="same"),
                    # self.out_convs.append(tf.keras.Sequential([layers.Conv2D(in_ch[0], 3, padding='same'),
                    layers.Activation("relu"),
                    layers.BatchNormalization(),
                ]
            )
        )
        for i in range(self.input_layer_cnt - 2):
            self.td_weights.append(tf.random.normal([2]))
            self.td_convs.append(
                tf.keras.Sequential(
                    [
                        layers.SeparableConv2D(in_ch[i + 1], 3, padding="same"),
                        # self.td_convs.append(tf.keras.Sequential([layers.Conv2D(in_ch[i+1], 3, padding='same'),
                        layers.Activation("relu"),
                        layers.BatchNormalization(),
                    ]
                )
            )
            self.out_weights.append(tf.random.normal([3]))
            self.out_convs.append(
                tf.keras.Sequential(
                    [
                        layers.SeparableConv2D(in_ch[i + 1], 3, padding="same"),
                        # self.out_convs.append(tf.keras.Sequential([layers.Conv2D(in_ch[i+1], 3, padding='same'),
                        layers.Activation("relu"),
                        layers.BatchNormalization(),
                    ]
                )
            )
        self.td_weights.append(tf.random.normal([2]))
        self.td_convs.append(
            tf.keras.Sequential(
                [
                    layers.SeparableConv2D(in_ch[-1], 3, padding="same"),
                    # self.td_convs.append(tf.keras.Sequential([layers.Conv2D(in_ch[-1], 3, padding='same'),
                    layers.Activation("relu"),
                    layers.BatchNormalization(),
                ]
            )
        )

        self.upconvs = [
            tf.keras.Sequential(
                [layers.UpSampling2D(u), layers.SeparableConv2D(c, k, padding=pad)]
            )
            # layers.Conv2D(c,k,padding=pad)])
            for u, c, k, pad in zip(
                [2, 2, 2, 2],
                in_ch[1:],
                # [2,3,2,3],
                [3, 3, 3, 3],
                ["same", "same", "same", "same"],
            )
        ]
        #print("updonvs ", self.upconvs)
        # self.downconvs= [tf.keras.Sequential([layers.ZeroPadding2D(pad),
        #                                      layers.AveragePooling2D(p),
        #                                      layers.Conv2D(c,3,padding='same')])
        #                                      for c,p,pad in zip(in_ch[:-1],
        #                                                         [2,2,2,2],
        #                                                         #[1,0,1,0])]
        #                                                         [1,0,1,0])]
        # self.downconvs= [layers.SeparableConv2D(c,3,2,padding='same')
        self.downconvs = [layers.Conv2D(c, 3, 2, padding="same") for c in in_ch[:-1]]

        #print("downconvs ", self.downconvs)

    def call(self, xs):
        """
        Args:
            xs: (Variable) list of features of each layer
        e.g.
        in block 1,2,4,7,14 in MobileNetV2 is used,
        shape of xs: [[?,10,10,160],[?,19,19,64],[?,38,38,32],[?,75,75,24],[?,150,150,32]]
        """
        tds = [xs[0]]
        for i in range(self.input_layer_cnt - 1):
            tds.append(
                self.td_convs[i](
                    (
                        self.td_weights[i][0] * xs[i + 1]
                        + self.td_weights[i][1] * self.upconvs[i](tds[-1])
                    )
                    / (tf.math.reduce_sum(self.td_weights[i]) + self.epsilon)
                )
            )

        outs = [tds[-1]]
        for i in range(self.input_layer_cnt - 2, 0, -1):
            outs.append(
                self.out_convs[i](
                    (
                        self.out_weights[i][0] * xs[i]
                        + self.out_weights[i][1] * tds[i]
                        + self.out_weights[i][2] * self.downconvs[i](outs[-1])
                    )
                    / (tf.math.reduce_sum(self.out_weights[i]) + self.epsilon)
                )
            )
        outs.append(
            self.out_convs[0](
                (
                    self.out_weights[0][0] * xs[0]
                    + self.out_weights[0][1] * self.downconvs[0](outs[-1])
                )
                / (tf.math.reduce_sum(self.out_weights[0]) + self.epsilon)
            )
        )

        return outs


def ResBlock(x, features, strides, group_width, name=None):
    print("resblock")
    shortcut = layers.Conv2D(
        features,
        1,
        strides=strides,
        padding="valid",
        use_bias=False,
        name=name + "_shortcut_conv",
    )(x)
    shortcut = layers.BatchNormalization(name=name + "_shortcut_batchnorm")(shortcut)

    x = layers.Conv2D(
        features, 1, strides=1, padding="same", use_bias=False, name=name + "_conv1"
    )(x)
    x = layers.BatchNormalization(name=name + "_batchnorm1")(x)
    x = layers.Activation("relu", name=name + "_relu1")(x)

    if group_width == 1:
        x = layers.SeparableConv2D(
            features,
            3,
            strides=strides,
            padding="same",
            use_bias=False,
            name=name + "_conv2",
        )(x)
        # x = layers.Conv2D(features, 3, strides=strides, padding='same', use_bias=False, name=name+'_conv2')(x)
    else:
        channel_per_group = features // group_width
        group_list = []
        for g in range(group_width):
            x_g = layers.Lambda(
                lambda z: z[..., g * channel_per_group : (g + 1) * channel_per_group]
            )(x)
            x_g = layers.SeparableConv2D(
                channel_per_group,
                3,
                strides=strides,
                padding="same",
                use_bias=False,
                name=name + "_groupconv{}".format(g + 1),
            )(x_g)
            # x_g = layers.Conv2D(channel_per_group, 3, strides=strides, padding='same', use_bias=False, name=name+'_groupconv{}'.format(g+1))(x_g)
            group_list.append(x_g)
        x = layers.Concatenate(name=name + "_groupconcat")(group_list)

    x = layers.BatchNormalization(name=name + "_batchnorm2")(x)
    x = layers.Activation("relu", name=name + "_relu2")(x)

    x = layers.Conv2D(
        features, 1, strides=1, padding="valid", use_bias=False, name=name + "_conv3"
    )(x)
    x = layers.BatchNormalization(name=name + "_batchnorm3")(x)
    x = layers.Add(name=name + "_add")([x, shortcut])
    x = layers.Activation("relu", name=name + "_relu3")(x)

    return x


def CustomNet(input_dim, num_stages, num_blocks, num_classes, group_width, name="None"):
    #
    img_inputs = layers.Input(shape=(input_dim[0], input_dim[1], 3), name="main_input")

    ###
    # Stem
    ###
    # x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False, name=name+'stem_conv')(img_inputs)
    x = layers.SeparableConv2D(
        32, 3, strides=2, padding="same", use_bias=False, name=name + "stem_conv"
    )(img_inputs)
    x = layers.BatchNormalization(name=name + "stem_batchnorm")(x)
    x = layers.Activation("relu", name=name + "stem_relu")(x)

    ###
    # Stage
    ###
    xs = []
    for s in range(num_stages):
        for b in range(num_blocks[s]):
            strides = 2 if b == 0 else 1
            x = ResBlock(
                x,
                num_channels[s],
                strides,
                group_width,
                name="stage{}_block{}".format(s + 1, b + 1),
            )
        xs.append(x)

    output_list = [
        (output.shape.as_list()[1], output.shape.as_list()[3]) for output in xs
    ]
    #print(output_list[::-1])
    #print(xs[::-1])
    x = BiFPN(output_list[::-1])(xs[::-1])
    x = BiFPN(output_list[::-1])(x[::-1])
    x = BiFPN(output_list[::-1])(x[::-1])

    img_outputs = []

    for bifpn_output in x[1:]:
        #print(bifpn_output)
        # detection
        outfilters = (5 + num_classes) * num_anchors
        x = layers.Conv2D(outfilters, 1, strides=1, use_bias=False)(bifpn_output)
        x = layers.Reshape((num_anchors, bifpn_output.shape[1], bifpn_output.shape[2], num_classes+5))(x)
        #x = layers.Activation("sigmoid")(x) # no sigmoid for whole layer, as only some values need to be between 0 and 1
        img_outputs.append(x)
        # objectness
        # x = layers.Conv2D(1,1, strides=1, use_bias=False)(bifpn_output)
        # x = layers.Activation('sigmoid')(x)
        # img_outputs.append(x)

    ###
    # Head
    ###
    # x = layers.Conv2D(512, 3)(x)
    # x = layers.Concatenate()(x)
    # x = layers.Dense(num_classes, name='logit')(x)
    # img_outputs = layers.Activation('softmax', name='main_output')(x)

    model = Model(img_inputs, img_outputs, name=name)

    return model


model = CustomNet(input_dim, num_stages, num_blocks, num_classes, group_width, 'CustomNet')
x = tf.random.uniform((2,input_dim[0], input_dim[1], 3), minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None)
print(model(x)[0].shape)

#model.summary()

#################################
# small testing of model building
#################################

assert model(x)[0].shape == (2, num_anchors, input_dim[0]//8, input_dim[1]//8, num_classes+5)
assert model(x)[1].shape == (2, num_anchors, input_dim[0]//16, input_dim[1]//16, num_classes+5)
assert model(x)[2].shape == (2, num_anchors, input_dim[0]//32, input_dim[1]//32, num_classes+5)
print("Model built successfully! 🚀🚀🚀")

#
# input_tensor = tf.random.uniform(shape = [896, 512, 3])
# input_tensor = tf.expand_dims(input_tensor, axis=0)
#
# prediction = model(input_tensor)
##print(prediction)


#
#tf.keras.models.save_model(model, './exported_models/custom_model_untrained_3bifpn_600class_big')
