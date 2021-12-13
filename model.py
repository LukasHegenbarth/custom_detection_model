import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

#input_dim = (896,512)
input_dim = (640,384)
num_stages = 4
#num_blocks = [4,4,4,4]
num_blocks = [3,4,6,3]
num_channels = [64, 128, 256, 512]
group_width = 8
#group_width = 16
num_classes = 1000


def CustomNet(input_dim, num_stages, num_blocks, num_classes, group_width, name='None'):
    # 
    img_inputs = layers.Input(shape=(input_dim[0], input_dim[1], 3), name='main_input')
    
    ###
    # Stem
    ###
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False, name=name+'stem_conv')(img_inputs)
    x = layers.BatchNormalization(name=name+'stem_batchnorm')(x)
    x = layers.Activation('relu', name=name+'stem_relu')(x)
    
    ###
    # Stage
    ###
    for s in range(num_stages):
        for b in range(num_blocks[s]):
            strides = 2 if b==0 else 1
            x = ResBlock(x, num_channels[s], strides, group_width, name='stage{}_block{}'.format(s+1, b+1))

    ###
    # Head
    ###
    x = layers.Dense(num_classes, name='logit')(x)
    img_outputs = layers.Activation('softmax', name='main_output')(x)

    model = Model(img_inputs, img_outputs, name=name)

    return model

def ResBlock(x, features, strides, group_width, name=None):
    print('resblock')
    shortcut = layers.Conv2D(features, 1, strides=strides, padding='valid', use_bias=False, name=name+'_shortcut_conv')(x)
    shortcut = layers.BatchNormalization(name=name+'_shortcut_batchnorm')(shortcut)

    x = layers.Conv2D(features, 1, strides=1, padding='same', use_bias=False, name=name+'_conv1')(x)
    x = layers.BatchNormalization(name=name+'_batchnorm1')(x)
    x = layers.Activation('relu', name=name+'_relu1')(x)

    if group_width == 1:
        x = layers.SeparableConv2D(features, 3, strides=strides, padding='same', use_bias=False, name=name+'_conv2')(x)
    else:
        channel_per_group = (features//group_width)
        group_list = []
        for g in range(group_width):
            x_g = layers.Lambda(lambda z: z[..., g*channel_per_group:(g+1)*channel_per_group])(x)
            #x_g = layers.SeparableConv2D(channel_per_group, 3, strides=strides, padding='same', use_bias=False, name=name+'_groupconv{}'.format(g+1))(x_g)
            x_g = layers.Conv2D(channel_per_group, 3, strides=strides, padding='same', use_bias=False, name=name+'_groupconv{}'.format(g+1))(x_g)
            group_list.append(x_g)
        x = layers.Concatenate(name=name+'_groupconcat')(group_list)
    
    x = layers.BatchNormalization(name=name+'_batchnorm2')(x)
    x = layers.Activation('relu', name=name+'_relu2')(x)

    x = layers.Conv2D(features, 1, strides=1, padding='valid', use_bias=False, name=name+'_conv3')(x)
    x = layers.BatchNormalization(name=name+'_batchnorm3')(x)
    x = layers.Add(name=name+'_add')([x, shortcut])
    x = layers.Activation('relu', name=name+'_relu3')(x)

    return x


model = CustomNet(input_dim, num_stages, num_blocks, num_classes, group_width, 'CustomNet')
model.summary()
