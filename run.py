import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from generator import Generator

image_path = './texture.jpg'
image_path = './texture2.jpg'

image = Image.open(image_path)


generator = Generator(image)
with torch.no_grad():
    generated_image = generator.run()
