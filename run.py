import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from decoders import *
from generator import Generator

image_path = './texture.jpg'
image_path = './texture2.jpg'

n_passes = 1

observed_layers = {
    'Relu5_1': {
        'index': 29,
        'decoder': feature_invertor_conv5_1,
        'n_epochs': 10,
        'n_slices': 10,
    },
    'Relu4_1': {
        'index': 20,
        'decoder': feature_invertor_conv4_1,
        'n_epochs': 10,
        'n_slices': 10,
    },
    'Relu3_1': {
        'index': 11,
        'decoder': feature_invertor_conv3_1,
        'n_epochs': 150,
        'n_slices': 10,
    },
    'Relu2_1': {
        'index': 6,
        'decoder': feature_invertor_conv2_1,
        'n_epochs': 50,
        'n_slices': 10,
    },
    'Relu1_1': {
        'index': 1,
        'decoder': feature_invertor_conv1_1,
        'n_epochs': 100,
        'n_slices': 10,
    }
}

image = Image.open(image_path)

generator = Generator(image, observed_layers)
generator.train_layer_decoders()
with torch.no_grad():
    pass_generated_images = generator.generate(n_passes)

for index, image in enumerate(pass_generated_images):
    plt.subplot(n_passes, len(generator.observed_layers), index+1)
    plt.imshow(image)
# for index, (key, value) in enumerate(generator.error_values.items()):
#     plt.subplot(5, 1, index+1)
#     plt.plot(value, label=key)
# plt.legend()
plt.show()
