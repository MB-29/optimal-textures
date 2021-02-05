import numpy as np
from torchvision import transforms

vgg_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
image_preprocessing = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


def output_value_hook(name, activation_register):
    def hook(model, input, output):
        activation_register[name] = output.data
    return hook


def sliced_transport(source_layer, target_layer):

    n_channels = source_layer.shape[0]
    assert n_channels == target_layer.shape[0]

    for dimension in range(n_channels):
        source_histogram = source_layer[dimension, :]
        target_histogram = target_layer[dimension, :]

        # match 1D histograms
        target_histogram[np.argsort(
            target_histogram)] = source_histogram[np.argsort(source_histogram)]

    return target_layer
