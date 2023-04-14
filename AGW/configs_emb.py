from yacs.config import CfgNode as CN
_C = CN()

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C.MODEL = CN() 
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'


# Strategy of FineTune. 
# Strategy  
# "add_nl_layer", "ft_blknl4","baseline"
_C.MODEL.STRATEGY = 'ft_blknl4'
# 1 means fine tune layer 4 and NL4 in resnet,classifier also changes
# 2 means fine tune layer 4 and NL4 in resnet,classifier don't change
# _C.MODEL.FINE_TUNE_LAST = 3
#  Whether add an NL layer after the resnet and tune
_C.MODEL.FINE_TUNE_NL = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''
# './pretrained/market1501_AGW.pth'
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'
_C.MODEL.PRETRAIN_CHOICE = 'self'
# 'self'


# Name of backbone
_C.MODEL.NAME = 'resnet50_nl'
# 'resnet50_nl'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# If train with center loss, options: 'bnneck' or 'no'
# _C.MODEL.CENTER_LOSS = 'on'
_C.MODEL.CENTER_FEAT_DIM = 2048
# If train with weighted regularized triplet loss, options: 'on', 'off'
# _C.MODEL.WEIGHT_REGULARIZED_TRIPLET = 'on'
# If train with generalized mean pooling, options: 'on', 'off'
_C.MODEL.GENERALIZED_MEAN_POOL = 'on'

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image
_C.INPUT.IMG_SIZE = [256, 128]
# Random probability for image horizontal flip
# _C.INPUT.PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128



