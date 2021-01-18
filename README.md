# Optimal textures


An implementation of the texture generation algorithm proposed in [Optimal Textures: Fast and Robust Texture Synthesis and Style Transfer through Optimal Transport](https://arxiv.org/abs/2010.14702).

### Example output


## Run 

Download the decoder weights from [this repository](https://github.com/sunshineatnoon/PytorchWCT), convert them to torch-compatible files and put the `.path` weight files in a folder `decoder_states` at the root of the repository.

```bash
python run.py
```



## Requirements
* Python 3
* torch
