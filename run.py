import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from PIL import Image

import argparse

from decoders import *
from generator import Generator

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('input_image_path', type=str,
                        help='relative path to the input texture image')
arg_parser.add_argument('decoder_state_path', type=str,
                        help='relative path to the decoder states')
arg_parser.add_argument('--output_path', '-o', type=str,
                        default='.', help='relative path to the output directory')
arg_parser.add_argument('-n', '--n_passes', type=int,
                        default=5, help='number of global passes')
arg_parser.add_argument(
    '-t', '--train', action='store_true', help='train the decoders')
args = arg_parser.parse_args()


def generate(input_image_path, decoder_state_path, n_passes=5, train=False):
    input_image = Image.open(input_image_path)
    generator = Generator(input_image, observed_layers)
    generator.set_layer_decoders(
        train=train, state_dir_path=decoder_state_path)
    with torch.no_grad():
        pass_generated_images = generator.generate(n_passes)
    return generator, pass_generated_images


# with torch.no_grad():
#     reconstructed_images = generator.reconstruct()

# for index, reconstructed_image in enumerate(reconstructed_images):
#     plt.subplot(1, len(generator.observed_layers), index+1)
#     plt.imshow(reconstructed_image)


# for index, (key, value) in enumerate(generator.error_values.items()):
#     plt.subplot(5, 1, index+1)
#     plt.plot(value, label=key)
# plt.legend()


if __name__ == '__main__':
    generator, pass_generated_images = generate(args.input_image_path, args.decoder_state_path,
                                                n_passes=args.n_passes, train=args.train)


    for index, image in enumerate(pass_generated_images):
        plt.axis('off')
        plt.subplot(generator.n_passes, len(generator.observed_layers), index+1)
        plt.imshow(image)
    plt.show()

    plt.axis('off')
    plt.imshow(pass_generated_images[-1])
    output_file_path = os.path.join(args.output_path, 'generated_texture.png')
    plt.savefig(output_file_path)
