import os
import numpy as np
from numpy.core.einsumfunc import _optimal_path
from numpy.core.fromnumeric import argsort
import torch
from torchvision.models import vgg19
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.models.vgg import VGG
from scipy.stats import ortho_group

from decoders import *


model_preprocessing = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def output_value_hook(name, activation_register):
    def hook(model, input, output):
        activation_register[name] = output.data
    return hook


decoder_state_dir_path = 'decoder_states'

observed_layers = {
    'Relu5_1': 29,
    'Relu4_1': 20,
    'Relu3_1': 11,
    'Relu2_1': 6,
    'Relu1_1': 1
}

# decoder models imported from /.decoders.py

layer_decoders = {
    'Relu5_1': feature_invertor_conv5_1,
    'Relu4_1': feature_invertor_conv4_1,
    'Relu3_1': feature_invertor_conv3_1,
    'Relu2_1': feature_invertor_conv2_1,
    'Relu1_1': feature_invertor_conv1_1,
}


class Generator:

    def __init__(self, image, n_global_passes=5, n_bins=128):

        self.width, self.height = image.width, image.height
        self.n_global_passes = n_global_passes
        self.observed_layers = observed_layers

        self.input_tensor = model_preprocessing(image)
        self.input_batch = self.input_tensor.unsqueeze(0)

        self.encoder = vgg19(pretrained=True).float()
        self.encoder_layers = {}
        self.layer_decoders = layer_decoders

        self.error_values ={layer_name : [] for layer_name in observed_layers.keys()}

        self.n_bins = n_bins

        self.encoder.eval()

        # observed layers iteration
        for layer_name, layer_index in self.observed_layers.items():

            # set a hook in the encoder
            layer = self.encoder.features[layer_index]
            layer.register_forward_hook(
                output_value_hook(layer_name, self.encoder_layers))

            # load symmetric decoder
            decoder = layer_decoders[layer_name]
            decoder_state_path = os.path.join(
                decoder_state_dir_path, f'{layer_name}_decoder_state.pth')
            decoder.load_state_dict(torch.load(decoder_state_path))
            decoder = decoder.float()
            decoder.eval()

    def run(self):
        
        # initialize with noise
        self.target_batch = torch.randn_like(self.input_batch)

        for global_pass in range(self.n_global_passes):
            print(f'global pass {global_pass}')
            for layer_name, layer_index in self.observed_layers.items():
                print(f'layer {layer_name}')
                self.encoder(self.input_batch)
                source_layer = self.encoder_layers[layer_name]

                self.encoder(self.target_batch)
                target_layer = self.encoder_layers[layer_name]

                target_layer = self.optimal_transport(
                    source_layer.squeeze(), target_layer.squeeze())
                target_layer = target_layer.view_as(source_layer)

                # Compute error
                error = torch.norm(target_layer - source_layer)
                self.error_values[layer_name].append(error)

                decoder = self.layer_decoders[layer_name]
                self.target_batch = decoder(target_layer)

        return self.target_batch[0].numpy().T

    def optimal_transport(self, source_layer, target_layer):

        n_channels = source_layer.shape[0]
        assert n_channels == target_layer.shape[0]

        n_slices = n_channels // self.n_global_passes

        # random orthonormal basis
        basis = torch.from_numpy(ortho_group.rvs(n_channels)).float()

        # project on the basis
        source_rotated_layer = basis @ source_layer.view(
            n_channels, -1)
        target_rotated_layer = basis @ target_layer.view(
            n_channels, -1)

        # sliced transport
        target_rotated_layer = sliced_transport(
            source_rotated_layer, target_rotated_layer)

        return basis.t() @ target_rotated_layer


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
