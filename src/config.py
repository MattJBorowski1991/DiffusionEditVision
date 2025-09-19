import torch
import os

# Model IDs
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
INPAINTING_MODEL_ID = "runwayml/stable-diffusion-inpainting"
CONTROLNET_ID = "lllyasviel/sd-controlnet-canny"  # For Canny edges

# Inference defaults
DEFAULT_NUM_STEPS = 10
DEFAULT_IMG2IMG_STEPS = 10
DEFAULT_INPAINT_STEPS = 10
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_STRENGTH = 0.8
DEFAULT_CONTROLNET_CONDITIONING_SCALE = 1.0  # How strongly to apply control map
DEFAULT_NUM_IMAGES = 1
DEFAULT_RESOLUTION = 512
DEFAULT_EVAL_RESOLUTION = 256
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Control map types
SUPPORTED_CONTROL_TYPES = ['none', 'canny']

# Paths
DATA_DIR = "data"
OUTPUTS_DIR = "outputs"
EVAL_DIR = os.path.join(OUTPUTS_DIR, "eval")

# Image processing
IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]

# Evaluation
DEFAULT_NUM_EVAL_SAMPLES = 10
DEFAULT_BATCH_SIZE = 4
DEFAULT_NEGATIVE_PROMPT = ""

# Memory optimization
ENABLE_ATTENTION_SLICING = True
ENABLE_VAE_SLICING = True
ENABLE_MODEL_CPU_OFFLOAD = True
