################################
# Config file for Model Training
################################

# Input shape of model
input_dim = (1052, 640, 3)
#input_dim = (896, 512, 3)
#input_dim = (1152, 640, 3)
#input_dim = (640,384, 3)

# Number of classes for from dataset
num_classes = 600

# Number of anchors for each pyramid resulution
num_anchors = 3

# Number of stages in RegNet
num_stages = 4

# Number of blocks per stage
# num_blocks = [4,4,4,4]
num_blocks = [3, 4, 6, 3]

# Number of channels in each stage 
num_channels = [64, 128, 256, 512]

# Group width for each residual block
# group_width = 1
group_width = 8
# group_width = 16
