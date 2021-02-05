import numpy as np
import torch

from generator import Generator
from utils import output_value_hook, sliced_transport, image_preprocessing, vgg_normalization


class StyleTransfer(Generator):
    """Style transfer model.
    """

    def __init__(self, style_image, content_image, observed_layers, n_bins=128):
        """
        :param content_image: content image
        :type content_image: PIL Image object
        """

        super().__init__(style_image, observed_layers, n_bins=n_bins)
        # input image
        self.content_tensor = image_preprocessing(content_image)
        self.normalized_content_batch = vgg_normalization(
            self.content_tensor).unsqueeze(0)
        self.content_batch = self.content_tensor.unsqueeze(0)

    def transfer(self, n_passes=5, content_strength=0.5):
        """Style transfer 

        :param n_passes: number of global passes, defaults to 5
        :type n_passes: int, optional
        :param content_strength: content strength, defaults to 0.5
        :type content_strength: float, optional
        :return: generated images layer by layer, step by step
        :rtype: list
        """

        self.n_passes = n_passes
        pass_generated_images = []

        # initialize with noise
        self.target_tensor = torch.randn_like(self.source_tensor)

        for global_pass in range(n_passes):
            print(f'global pass {global_pass}')
            for layer_name, layer_information in self.observed_layers.items():
                print(f'layer {layer_name}')

                # forward pass on source image
                self.encoder(self.normalized_source_batch)
                source_layer = self.encoder_layers[layer_name]

                # forward pass on content image
                self.encoder(self.normalized_content_batch)
                content_layer = self.encoder_layers[layer_name]

                # forward pass on target image
                target_batch = vgg_normalization(
                    self.target_tensor).unsqueeze(0)
                self.encoder(target_batch)
                target_layer = self.encoder_layers[layer_name]

                # transport
                target_layer = self.optimal_transport(layer_name,
                                                      source_layer.squeeze(), target_layer.squeeze())
                target_layer = target_layer.view_as(source_layer)

                # feature style transfer
                target_layer += content_strength * \
                    (content_layer - target_layer)

                # decode
                decoder = layer_information['decoder']
                self.target_tensor = decoder(target_layer).squeeze()

                generated_image = np.transpose(
                    self.target_tensor.numpy(), (1, 2, 0)).copy()
                pass_generated_images.append(generated_image)

        return pass_generated_images
