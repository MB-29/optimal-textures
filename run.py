import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from PIL import Image

from decoders import *
from generator import Generator
from style_transfer import StyleTransfer

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('source_image_path', type=str,
                        help='relative path to the input texture image')
arg_parser.add_argument('-c', '--content_image_path', type=str,
                        help='relative path to the content image')
arg_parser.add_argument('decoder_state_path', type=str,
                        help='relative path to the decoder states')
arg_parser.add_argument('-o', '--output_path', type=str,
                        default='.', help='relative path to the output directory')
arg_parser.add_argument('-n', '--n_passes', type=int,
                        default=5, help='number of global passes')
arg_parser.add_argument('-s', '--content_strength', type=float,
                        default=0.5, help='strength of the content image')
arg_parser.add_argument(
    '-t', '--train', action='store_true', help='train the decoders')
args = arg_parser.parse_args()


def generate(source_image_path, decoder_state_path, n_passes=5, train=False):
    source_image = Image.open(source_image_path)
    generator = Generator(source_image, observed_layers)
    generator.set_layer_decoders(
        train=train, state_dir_path=decoder_state_path)
    with torch.no_grad():
        pass_generated_images = generator.generate(n_passes)
    return generator, pass_generated_images


def style_transfer(source_image_path, content_image_path, decoder_state_path, content_strength=0.5, n_passes=5, train=False):
    assert content_strength >= 0 and content_strength <=1
    source_image = Image.open(source_image_path)
    content_image = Image.open(content_image_path)
    transfer = StyleTransfer(source_image, content_image, observed_layers)
    transfer.set_layer_decoders(train=train, state_dir_path=decoder_state_path)
    with torch.no_grad():
        pass_generated_images = transfer.transfer(n_passes, content_strength)
    return transfer, pass_generated_images


if __name__ == '__main__':

    # texture generation
    if args.content_image_path is None:
        generator, pass_generated_images = generate(
            args.source_image_path,
            args.decoder_state_path,
            n_passes=args.n_passes,
            train=args.train)
    
    # style transfer
    else:
        generator, pass_generated_images = style_transfer(
            args.source_image_path,
            args.content_image_path,
            args.decoder_state_path,
            content_strength=args.content_strength,
            n_passes=args.n_passes,
            train=args.train
        )

    # plot generated images, layer after layer, pass after pass
    for index, image in enumerate(pass_generated_images):
        plt.subplot(generator.n_passes, len(
            generator.observed_layers), index+1)
        plt.imshow(image)
        plt.axis('off')
    plt.show()

    # save final output image
    output_file_path = os.path.join(args.output_path, 'generated_texture.png')
    final_image = (pass_generated_images[-1] * 255).astype(np.uint8)
    output_image = Image.fromarray(final_image)
    output_image.save(output_file_path)
