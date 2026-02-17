# Uthana Comfy

A collection of Uthana custom nodes for ComfyUI

## Quickstart

1. Install [ComfyUI](https://docs.comfy.org/get_started).
1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
1. Look up this extension in ComfyUI-Manager. If you are installing manually, clone this repository under `ComfyUI/custom_nodes`.
1. Restart ComfyUI.

# Features

- CreateCharacter - Upload a 3D character mesh to Uthana, this will generate a skinned character that can be animated.
- TextToMotionVqvaeV1 - Generate a 3D character animation, using our vqvae model.
- TextToMotionDiffusionV1 - Generate a 3D character animation, using our SOTA Diffusion model.
- DownloadMotion - Download a generated animation as a GLB or FBX file.
- DownloadCharacter - Donwload a skinned character as a GLB or FBX file.

## Develop

To install the dev dependencies:

```bash
cd uthana-comfy
pip install -e .[dev]
pre-commit install
```

The `-e` flag above will result in a "live" install, in the sense that any changes you make to your node extension will automatically be picked up the next time you run ComfyUI.
