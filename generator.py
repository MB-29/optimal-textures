import os
import numpy as np
import torch
from torchvision.models import vgg19
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.stats import ortho_group



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


class Generator:

    def __init__(self, image, observed_layers, n_bins=128):

        self.width, self.height = image.width, image.height

        self.input_tensor = image_preprocessing(image)
        self.normalized_input_batch = vgg_normalization(
            self.input_tensor).unsqueeze(0)
        self.input_batch = self.input_tensor.unsqueeze(0)

        self.encoder = vgg19(pretrained=True).float()
        self.encoder.eval()
        self.encoder_layers = {}
        self.set_encoder_hooks(observed_layers)

        self.n_bins = n_bins

    def set_encoder_hooks(self, observed_layers):

        self.observed_layers = observed_layers
        self.error_values = {layer_name: []
                             for layer_name in observed_layers.keys()}
        self.decoder_loss_values = {}

        for layer_name, layer_information in observed_layers.items():

            # set a hook in the encoder
            layer_index = layer_information['index']
            layer = self.encoder.features[layer_index]
            layer.register_forward_hook(
                output_value_hook(layer_name, self.encoder_layers))

    def train_decoders(self, n_epochs=10):
        for layer_name, layer_information in self.observed_layers.items():
            decoder = layer_information['decoder']
            # decoder_state_path = os.path.join(
            #     decoder_state_dir_path, f'{layer_name}_decoder_state.pth')
            # decoder.load_state_dict(torch.load(decoder_state_path))
            # decoder = decoder.float()
            # shape = layer_information['shape']
            print(f'training decoder for layer {layer_name}')
            loss_values = self.train_decoder(layer_name, n_epochs)
            self.decoder_loss_values[layer_name] = loss_values

            decoder.eval()

    def generate(self, n_global_passes=5):

        pass_generated_images = []

        # initialize with noise
        self.target_batch = torch.randn_like(self.input_batch)

        for global_pass in range(n_global_passes):
            print(f'global pass {global_pass}')
            for layer_name, layer_information in self.observed_layers.items():
                print(f'layer {layer_name}')

                self.encoder(self.normalized_input_batch)
                source_layer = self.encoder_layers[layer_name]

                self.encoder(vgg_normalization(self.target_batch))
                target_layer = self.encoder_layers[layer_name]
                print(f'target layer shape {target_layer.shape}')

                target_layer = self.optimal_transport(
                    source_layer.squeeze(), target_layer.squeeze(), n_global_passes)
                target_layer = target_layer.view_as(source_layer)

                # Compute error
                error = torch.norm(target_layer - source_layer)
                self.error_values[layer_name].append(error)

                decoder = layer_information['decoder']
                self.target_batch = decoder(target_layer)
                input_reconstruction = decoder(source_layer)

                # self.pass_generated_images.append(self.target_batch[0].numpy().T.copy())
                pass_generated_images.append(
                    input_reconstruction[0].numpy().T.copy())

        return pass_generated_images

    def optimal_transport(self, source_layer, target_layer, n_global_passes):

        n_channels = source_layer.shape[0]
        assert n_channels == target_layer.shape[0]

        n_slices = n_channels // n_global_passes
        n_slices = 10

        for slice in range(n_slices):
            # random orthonormal basis
            basis = torch.from_numpy(ortho_group.rvs(n_channels)).float()
            # basis = torch.eye(n_channels)

            # project on the basis
            source_rotated_layer = basis @ source_layer.view(
                n_channels, -1)
            target_rotated_layer = basis @ target_layer.view(
                n_channels, -1)

            # sliced transport
            target_rotated_layer = sliced_transport(
                source_rotated_layer, target_rotated_layer)
            target_layer = basis.t() @ target_rotated_layer

        return target_layer

    def reconstruct(self):

        generated_images = []

        for layer_name, layer_information in self.observed_layers.items():
            print(f'layer {layer_name}')

            self.encoder(self.normalized_input_batch)
            source_layer = self.encoder_layers[layer_name]

            decoder = layer_information['decoder']
            input_reconstruction = decoder(source_layer)

            generated_images.append(
                input_reconstruction[0].numpy().T.copy())

        return generated_images

    def train_decoder(self, layer_name, n_epochs, learning_rate=1e-3):
        decoder = self.observed_layers[layer_name]['decoder']
        training_loss_values = []
        image_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

        for epoch_index in range(n_epochs):
            #print(f'Epoch {epoch_index}')
            epoch_loss = 0

            # reconstruct
            self.encoder(vgg_normalization(self.input_tensor).unsqueeze(0))
            embedding = self.encoder_layers[layer_name]
            generated_tensor = decoder(embedding).squeeze()

            # re-embed
            self.encoder(vgg_normalization(generated_tensor).unsqueeze(0))
            generated_embedding = self.encoder_layers[layer_name]

            loss = image_loss(self.input_tensor, generated_tensor) + \
                torch.norm(embedding - generated_embedding)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss

            training_loss_values.append(epoch_loss)
        return training_loss_values
        
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



