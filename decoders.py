
import torch
import torch.nn as nn

"""Decoder models. Layer-specific, editable information is stored
in the dictionary observed_layers at the bottom of the file.
"""


feature_invertor_conv1_1 = nn.Sequential(  # Sequential,
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64, 3, (3, 3)),
)

feature_invertor_conv2_1 = nn.Sequential(  # Sequential,
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128, 64, (3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64, 64, (3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64, 3, (3, 3)),
)

feature_invertor_conv3_1 = nn.Sequential(  # Sequential,
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256, 128, (3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128, 128, (3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128, 64, (3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64, 64, (3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64, 3, (3, 3)),
)

feature_invertor_conv4_1 = nn.Sequential(  # Sequential,
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512, 256, (3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256, 256, (3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256, 256, (3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256, 256, (3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256, 128, (3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128, 128, (3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128, 64, (3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64, 64, (3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64, 3, (3, 3)),
)

feature_invertor_conv5_1 = nn.Sequential( # Sequential,
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,512,(3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,512,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,512,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,512,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,256,(3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,128,(3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,128,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,64,(3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,64,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,3,(3, 3)),
)

observed_layers = {
    'Relu5_1': {
        'index': 29,
        'decoder': feature_invertor_conv5_1,
        'n_slices': 10,
        'n_epochs': 500
    },
    'Relu4_1': {
        'index': 20,
        'decoder': feature_invertor_conv4_1,
        'n_slices': 10,
        'n_epochs': 500
    },
    'Relu3_1': {
        'index': 11,
        'decoder': feature_invertor_conv3_1,
        'n_slices': 10,
        'n_epochs': 500
    },
    'Relu2_1': {
        'index': 6,
        'decoder': feature_invertor_conv2_1,
        'n_slices': 10,
        'n_epochs': 500
    },
    'Relu1_1': {
        'index': 1,
        'decoder': feature_invertor_conv1_1,
        'n_slices': 10,
        'n_epochs': 500
    }
}
