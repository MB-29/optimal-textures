import torch
from torch._C import BoolTensor
from torchvision.models import vgg19
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

model_preprocessing = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

layer_outputs = {}

def output_value_hook(name):
    def hook(model, input, output):
        layer_outputs[name] = output.data.norm()
    return hook


observed_layers = {
    'Relu5_1': 29,
    'Relu4_1': 20,
    'Relu3_1': 11,
    'Relu2_1': 6,
    'Relu1_1': 1
}



class Generator:

    def __init__(self, image, n_global_passes=10):

        self.width, self.height = image.width, image.height
        self.n_global_passes = n_global_passes

        self.input_tensor = model_preprocessing(image)
        self.input_batch = self.input_tensor.unsqueeze(0)

        self.encoder = vgg19(pretrained=True)
        self.encoder.eval()
        for layer_name, layer_index in observed_layers.items():
            layer = self.encoder.features[layer_index]
            layer.register_forward_hook(output_value_hook(layer_name))

    def run(self):

        self.generated_batch = torch.randn_like(self.input_batch)
        for global_pass in self.n_global_passes:
            for layer_name, layer_index in observed_layers.items():
                self.encoder(self.input_batch)

                self.encoder(self.generated_batch)
                
            
