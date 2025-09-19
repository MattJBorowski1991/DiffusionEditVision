import os
import argparse
import gc
import sys
import torch
import numpy as np
from PIL import Image, ImageFilter
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

# Diffusers imports
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderTiny
)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor

# Import configuration
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

def load_model(use_img2img: bool = False, use_controlnet: bool = False, 
               control_type: str = 'none', use_inpainting: bool = False):
    """
    Load the appropriate Stable Diffusion model based on the specified parameters.
    
    Args:
        use_img2img: Whether to load the img2img pipeline
        use_controlnet: Whether to use ControlNet
        control_type: Type of ControlNet to use (must be in SUPPORTED_CONTROL_TYPES)
        use_inpainting: Whether to use the inpainting pipeline
        
    Returns:
        Loaded pipeline ready for inference
    """
    # Import necessary modules in the function scope
    from diffusers import (
        StableDiffusionPipeline, 
        StableDiffusionImg2ImgPipeline, 
        StableDiffusionInpaintPipeline,
        StableDiffusionControlNetPipeline, 
        StableDiffusionControlNetImg2ImgPipeline
    )
    
    # Determine model ID based on task
    model_id = INPAINTING_MODEL_ID if use_inpainting else BASE_MODEL_ID
    controlnet = None
    
    # Validate control type
    if control_type not in SUPPORTED_CONTROL_TYPES:
        raise ValueError(f"Unsupported control_type: {control_type}. Must be one of {SUPPORTED_CONTROL_TYPES}")
    
    # Initialize ControlNet if needed
    if use_controlnet and control_type != 'none':
        controlnet_id = f"lllyasviel/sd-controlnet-{control_type}"
        controlnet = ControlNetModel.from_pretrained(controlnet_id)
    
    try:
        print(f"Loading model: {model_id}")
        print(f"Mode: {'inpainting' if use_inpainting else 'img2img' if use_img2img else 'text2img'}")
        if use_controlnet:
            print(f"Using ControlNet: {control_type}")
        
        # Determine the appropriate pipeline class
        if use_inpainting:
            pipe_cls = StableDiffusionInpaintPipeline
        elif use_img2img:
            pipe_cls = StableDiffusionControlNetImg2ImgPipeline if (use_controlnet and controlnet) else StableDiffusionImg2ImgPipeline
        else:
            pipe_cls = StableDiffusionControlNetPipeline if (use_controlnet and controlnet) else StableDiffusionPipeline
        
        # Initialize the pipeline
        pipe_kwargs = {
            'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
            'safety_checker': None,
            'requires_safety_checker': False
        }
        
        # Add ControlNet if needed
        if use_controlnet and controlnet:
            pipe_kwargs['controlnet'] = controlnet
        
        pipe = pipe_cls.from_pretrained(model_id, **pipe_kwargs)
        
        # Apply memory optimizations
        if ENABLE_ATTENTION_SLICING:
            pipe.enable_attention_slicing(1)  # More aggressive slicing
        if ENABLE_VAE_SLICING:
            pipe.enable_vae_slicing()
        if ENABLE_MODEL_CPU_OFFLOAD and torch.cuda.is_available():
            pipe.enable_model_cpu_offload()
        
        # Move to appropriate device
        device = torch.device(DEFAULT_DEVICE)
        pipe = pipe.to(device)
        
        # Clean up
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return pipe
        
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        raise

def log_step(step, total_steps, step_name):
    print(f"\nStep {step + 1}/{total_steps}: {step_name}")

def log_system_info():
    import platform
    import psutil
    import torch
    
    print("\n=== System Information ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # CPU info
    print(f"CPU Cores: {psutil.cpu_count(logical=True)} (Physical: {psutil.cpu_count(logical=False)})")
    
    # Memory info
    vm = psutil.virtual_memory()
    print(f"Total RAM: {vm.total / (1024**3):.2f} GB")
    print(f"Available RAM: {vm.available / (1024**3):.2f} GB")
    print(f"Used RAM: {vm.used / (1024**3):.2f} GB")
    print(f"RAM Usage: {vm.percent}%")
    
    # Disk info
    disk = psutil.disk_usage('/')
    print(f"\nDisk Space Total: {disk.total / (1024**3):.2f} GB")
    print(f"Disk Space Used: {disk.used / (1024**3):.2f} GB")
    print(f"Disk Space Free: {disk.free / (1024**3):.2f} GB")
    print(f"Disk Usage: {disk.percent}%")
    print("=" * 30 + "\n")

def ensure_output_dir():
    """Ensure the outputs directory exists"""
    output_dir = os.path.abspath(OUTPUTS_DIR)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def generate_image(pipe, prompt, image_path=None, mask_path=None, control_type='none', 
                 strength=DEFAULT_STRENGTH, num_inference_steps=None, resolution=512, 
                 controlnet_conditioning_scale=None, output_path=None, **kwargs):
    """
    Generate an image using either text-to-image, image-to-image, or inpainting
    
    Args:
        pipe: The loaded pipeline
        prompt: Text prompt for generation (required)
        image_path: Path to input image for img2img/inpainting (None for text2img)
        mask_path: Path to mask image for inpainting (None for non-inpainting)
        strength: How much to modify the input image (0.0 to 1.0)
        num_inference_steps: Number of denoising steps
        resolution: Resolution of the output image (width and height)
        controlnet_conditioning_scale: ControlNet conditioning scale
        
    Returns:
        Path to the generated image
    """
    # Set default values from config if not provided
    if num_inference_steps is None:
        num_inference_steps = DEFAULT_INPAINT_STEPS if mask_path else DEFAULT_IMG2IMG_STEPS if image_path else DEFAULT_NUM_STEPS
    
    if resolution is None:
        resolution = DEFAULT_RESOLUTION
    
    if controlnet_conditioning_scale is None:
        controlnet_conditioning_scale = DEFAULT_CONTROLNET_CONDITIONING_SCALE
    
    width = height = resolution
    
    # Validate inputs
    if prompt is None:
        prompt = "A high quality, detailed image"
    
    if mask_path and not image_path:
        raise ValueError("image_path must be provided when using mask_path")
    
    try:
        # If strength is very small and not inpainting, just return the original image
        min_strength = 0.01  # Minimum strength to avoid VAE decoder errors
        if image_path and os.path.exists(image_path) and strength < min_strength and not mask_path:
            print(f"Strength ({strength}) is below minimum threshold ({min_strength}), returning original image")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"output_{timestamp}_original.jpg"
            output_dir = ensure_output_dir()
            output_path = os.path.join(output_dir, output_filename)
            
            # Save the original image to the output path
            import shutil
            shutil.copy2(image_path, output_path)
            return output_path
            
        # Common parameters for generation
        common_params = {
            'prompt': prompt,  # Include prompt here to ensure it's always provided
            'num_inference_steps': num_inference_steps or DEFAULT_NUM_STEPS,
            'guidance_scale': DEFAULT_GUIDANCE_SCALE,
            'num_images_per_prompt': DEFAULT_NUM_IMAGES,
            'output_type': 'pil'
        }
        
        # Handle inpainting
        if mask_path and os.path.exists(mask_path):
            if not image_path or not os.path.exists(image_path):
                raise ValueError("Image path is required for inpainting")
                
            # Load and prepare the input image and mask
            init_image = Image.open(image_path).convert("RGB").resize((resolution, resolution))
            mask_image = Image.open(mask_path).convert("L").resize((resolution, resolution))
            
            # Inpainting parameters
            inpainting_params = {
                'image': init_image,
                'mask_image': mask_image,
                'strength': strength,
                'num_inference_steps': num_inference_steps,
                **common_params
            }
            
            # Generate the image
            print(f"Generating inpainting with {num_inference_steps} steps...")
            result = pipe(**inpainting_params)
            
            # Save the result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"output_{timestamp}_inpaint.png"
            output_dir = ensure_output_dir()
            output_path = os.path.join(output_dir, output_filename)
            result.images[0].save(output_path)
            print(f"Inpainting saved to {output_path}")
            return output_path
        
        if image_path and os.path.exists(image_path):
            try:
                # Generate control image if needed
                control_image = None
                if control_type != 'none':
                    # Generate canny edge detection
                    control_img = Image.open(image_path).convert('L')
                    control_img = control_img.filter(ImageFilter.FIND_EDGES)
                    control_image = Image.merge('RGB', (control_img, control_img, control_img))
                    control_image = control_image.resize((resolution, resolution), Image.LANCZOS)
                    
                    # Save control image for debugging
                    control_dir = os.path.join(OUTPUTS_DIR, "controls")
                    os.makedirs(control_dir, exist_ok=True)
                    control_image_path = os.path.join(control_dir, f"control_{os.path.basename(image_path)}")
                    control_image.save(control_image_path)
                    print(f"Saved control image to {control_image_path}")
                
                # Ensure we have a PIL Image
                from PIL import Image
                
                if isinstance(image_path, (str, os.PathLike)):
                    init_image = Image.open(image_path)
                elif hasattr(image_path, 'convert'):  # Already a PIL Image
                    init_image = image_path
                elif hasattr(image_path, 'numpy'):  # PyTorch Tensor
                    import torch
                    init_image = image_path.squeeze(0).permute(1, 2, 0).numpy()
                    init_image = (init_image * 255).astype('uint8')
                    init_image = Image.fromarray(init_image)
                elif isinstance(image_path, np.ndarray):  # NumPy array
                    if image_path.dtype == np.float32 or image_path.dtype == np.float64:
                        init_image = (image_path * 255).astype('uint8')
                    init_image = Image.fromarray(init_image)
                else:
                    raise ValueError(f"Unsupported image input type: {type(image_path)}. "
                                   f"Supported types: PIL.Image, numpy.ndarray, torch.Tensor, file path")
                
                # Ensure RGB mode
                if init_image.mode != 'RGB':
                    init_image = init_image.convert('RGB')
                    
                # Resize if needed
                if init_image.size != (resolution, resolution):
                    init_image = init_image.resize((resolution, resolution), Image.LANCZOS)
                
                # Prepare generation parameters
                gen_params = {
                    **common_params,
                    'image': init_image,
                    'strength': strength,
                    'height': resolution,
                    'width': resolution
                }
                
                # Add ControlNet parameters if needed
                if control_type != 'none' and control_image:
                    gen_params.update({
                        'control_image': control_image,
                        'controlnet_conditioning_scale': controlnet_conditioning_scale or DEFAULT_CONTROLNET_CONDITIONING_SCALE
                    })
                
                # Generate the image
                debug_params = {k: v for k, v in gen_params.items() if k != 'image'}
                print(f"Generating image with params: {debug_params}")
                try:
                    # Remove height/width from gen_params if they exist to avoid duplicates
                    gen_params.pop('height', None)
                    gen_params.pop('width', None)
                    result = pipe(**gen_params)
                except Exception as e:
                    print(f"Error in pipe(): {str(e)}")
                    raise
                
                if not result or not hasattr(result, 'images') or not result.images:
                    raise ValueError("Failed to generate image: No images returned from pipeline")
                    
                image = result.images[0]
                
                if not image or image.size[0] == 0:
                    raise ValueError("Generated image is empty")
                    
            except Exception as e:
                print(f"Error during image generation: {str(e)}")
                # Fall back to text-to-image if img2img fails
                print("Falling back to text-to-image generation...")
                text_to_image_params = common_params.copy()
                text_to_image_params.pop('image', None)
                text_to_image_params.pop('strength', None)
                result = pipe(**text_to_image_params)
                if not result or not hasattr(result, 'images') or not result.images:
                    raise ValueError("Fallback text-to-image generation also failed")
                image = result.images[0]
        else:
            # Text-to-image generation
            text_to_image_params = common_params.copy()
            text_to_image_params.pop('image', None)
            text_to_image_params.pop('strength', None)
            result = pipe(**text_to_image_params)
            if not result or not hasattr(result, 'images') or not result.images:
                raise ValueError("Text-to-image generation failed")
            image = result.images[0]
            
        # Generate timestamp and save the image
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"output_{timestamp}.png"
            output_dir = ensure_output_dir()
            output_path = os.path.join(output_dir, output_filename)
        else:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"Image saved to {output_path}")
        
        # Force garbage collection
        import gc
        import torch  # Ensure torch is imported for CUDA operations
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return output_path
        
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        raise

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate images using Stable Diffusion with optional ControlNet and Inpainting')
    
    # Basic generation parameters
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for image generation')
    parser.add_argument('--image_path', type=str, default=None, help='Path to input image for img2img/inpainting (optional)')
    parser.add_argument('--mask_path', type=str, default=None, help='Path to mask image for inpainting (optional)')
    parser.add_argument('--output_dir', type=str, default=OUTPUTS_DIR, help=f'Directory to save generated images (default: {OUTPUTS_DIR})')
    parser.add_argument('--strength', type=float, default=DEFAULT_STRENGTH, 
                       help=f'Strength of the transformation (0.0 to 1.0, default: {DEFAULT_STRENGTH})')
    parser.add_argument('--steps', type=int, default=None, 
                       help=f'Number of denoising steps (default: {DEFAULT_NUM_STEPS} for txt2img, {DEFAULT_IMG2IMG_STEPS} for img2img, {DEFAULT_INPAINT_STEPS} for inpainting)')
    parser.add_argument('--resolution', type=int, default=DEFAULT_RESOLUTION, 
                       help=f'Image resolution (width and height, default: {DEFAULT_RESOLUTION})')
    
    # ControlNet parameters
    parser.add_argument('--control_type', type=str, default='none', 
                       choices=SUPPORTED_CONTROL_TYPES, 
                       help=f'Type of control to use (default: none, options: {", ".join(SUPPORTED_CONTROL_TYPES)})')
    parser.add_argument('--control_scale', type=float, default=DEFAULT_CONTROLNET_CONDITIONING_SCALE,
                       help=f'ControlNet conditioning scale (0.0 to 2.0, default: {DEFAULT_CONTROLNET_CONDITIONING_SCALE})')
    
    # Memory optimization
    parser.add_argument('--attention_slicing', type=str, default=None,
                       help='Enable/disable attention slicing (auto/None to disable)')
    parser.add_argument('--vae_slicing', action='store_true',
                       help='Enable VAE slicing (reduces memory usage)')
    parser.add_argument('--cpu_offload', action='store_true',
                       help='Enable model CPU offload (reduces memory usage)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Update global config with command line arguments
    global ENABLE_ATTENTION_SLICING, ENABLE_VAE_SLICING, ENABLE_MODEL_CPU_OFFLOAD, OUTPUTS_DIR
    
    if args.attention_slicing is not None:
        ENABLE_ATTENTION_SLICING = args.attention_slicing
    if args.vae_slicing:
        ENABLE_VAE_SLICING = True
    if args.cpu_offload:
        ENABLE_MODEL_CPU_OFFLOAD = True
    
    # Update output directory
    OUTPUTS_DIR = args.output_dir
    
    # Ensure output directory exists
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Log system info
    log_system_info()
    
    try:
        # Determine the type of generation
        is_img2img = args.image_path is not None
        use_controlnet = args.control_type != 'none'
        use_inpainting = args.mask_path is not None
        
        if use_inpainting and not is_img2img:
            raise ValueError("Inpainting requires both --image_path and --mask_path")
        
        # Determine mode for logging
        mode = 'inpainting' if use_inpainting else 'image-to-image' if is_img2img else 'text-to-image'
        print(f"\n=== {mode.upper()} MODE ===")
        print(f"Prompt: {args.prompt}")
        if is_img2img:
            print(f"Input image: {args.image_path}")
        if use_inpainting:
            print(f"Mask image: {args.mask_path}")
        if use_controlnet:
            print(f"Using ControlNet with type: {args.control_type}, scale: {args.control_scale}")
        
        # Load the appropriate model
        print("\nLoading model...")
        pipe = load_model(
            use_img2img=is_img2img,
            use_controlnet=use_controlnet,
            control_type=args.control_type,
            use_inpainting=use_inpainting
        )
        
        # Generate the image
        print("\nGenerating image...")
        output_path = generate_image(
            pipe,
            prompt=args.prompt,
            image_path=args.image_path,
            mask_path=args.mask_path,
            control_type=args.control_type,
            strength=args.strength,
            num_inference_steps=args.steps,
            resolution=args.resolution,
            controlnet_conditioning_scale=args.control_scale
        )
        
        # Print result
        if output_path and os.path.exists(output_path):
            print(f"\n✅ Image successfully generated and saved to: {os.path.abspath(output_path)}")
            print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
        else:
            print("\n❌ Failed to generate image. Please check the error messages above.")
        
        return output_path
            
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        if hasattr(e, 'args') and len(e.args) > 1:
            print(f"Error details: {e.args[1]}")
        import traceback
        
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
