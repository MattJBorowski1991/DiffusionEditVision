import os
import sys
import io
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from PIL import Image

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))

# Import configuration and inference
from config import *
from inference import load_model, generate_image, ensure_output_dir
from metrics import calculate_metrics, print_metrics

# Third-party imports
from datasets import load_dataset
from PIL import Image

def download_instructpix2pix(num_samples=None):
    """Download a subset of the InstructPix2Pix dataset"""
    if num_samples is None:
        num_samples = DEFAULT_NUM_EVAL_SAMPLES
        
    print(f"Downloading {num_samples} samples from InstructPix2Pix dataset...")
    try:
        dataset = load_dataset("fusing/instructpix2pix-1000-samples")
        
        # Handle different dataset formats
        if isinstance(dataset, dict):
            # If dataset is split into train/validation/test, combine them
            all_data = []
            for split_data in dataset.values():
                if hasattr(split_data, 'to_list'):
                    all_data.extend(split_data.to_list())
                else:
                    all_data.extend(split_data)
            dataset = all_data
        elif hasattr(dataset, 'to_list'):
            # Convert to list if it's a Dataset object
            dataset = dataset.to_list()
        
        # Select random samples if needed
        if num_samples is not None and num_samples < len(dataset):
            import random
            dataset = random.sample(dataset, num_samples)
        
        # Print the structure of the first item for debugging
        if len(dataset) > 0:
            print("\nFirst item structure:")
            for key, value in dataset[0].items():
                print(f"- {key}: {type(value).__name__}")
            
        print(f"\nSuccessfully downloaded {len(dataset)} random samples")
        return dataset
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        raise

def preprocess_image(image, resolution=None):
    """Preprocess image for evaluation
    
    Args:
        image: PIL Image to preprocess
        resolution: Target resolution (width=height). If None, uses DEFAULT_EVAL_RESOLUTION
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    if resolution is None:
        resolution = DEFAULT_EVAL_RESOLUTION
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize and normalize
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])
    
    return transform(image)

def generate_predictions(dataset, output_dir=None, device=None, batch_size=1):
    """Generate predictions for the evaluation dataset
    
    Args:
        dataset: Dataset containing 'input_image' and 'edit_prompt' for each sample
        output_dir: Directory to save outputs. If None, uses EVAL_DIR from config
        device: Device to use ('cuda' or 'cpu'). If None, uses DEFAULT_DEVICE from config
        batch_size: Number of samples to process in parallel (default: 1 for stability)
        
    Returns:
        List of dicts containing prediction results with paths to saved images
    """
    if output_dir is None:
        output_dir = EVAL_DIR
    
    # Create output directories
    pred_dir = os.path.join(output_dir, 'predictions')
    inputs_dir = os.path.join(output_dir, 'inputs')
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(inputs_dir, exist_ok=True)
    
    # Initialize the model
    print("\nLoading model for evaluation...")
    pipe = load_model()
    pipe = pipe.to(device)  # Move model to the specified device
    
    predictions = []
    prompts_file = os.path.join(output_dir, 'prompts.txt')
    
    # Clear prompts file if it exists
    if os.path.exists(prompts_file):
        os.remove(prompts_file)
    
    print(f"\nGenerating predictions (batch_size={batch_size})...")
    
    for i, item in enumerate(tqdm(dataset, desc="Generating predictions")):
        try:
            # Get the prompt and input image
            prompt = item.get('edit_prompt', '')
            input_img = item.get('input_image')
            
            if not prompt or not input_img:
                print(f"Warning: Missing prompt or input image for sample {i}")
                continue
                
            # Debug: Print dataset item structure for first few samples
            if i < 5:  # Only print for first 5 samples to avoid too much output
                print(f"\nDebug - Dataset item keys: {list(item.keys())}")
                print(f"Debug - Prompt: {prompt}")
                print(f"Debug - Input image type: {type(input_img)}")
                if hasattr(input_img, 'keys'):
                    print(f"Debug - Input image keys: {list(input_img.keys())}")
            
            # Save input image
            input_path = os.path.join(inputs_dir, f'input_{i:04d}.jpg')
            try:
                if isinstance(input_img, dict):
                    if 'bytes' in input_img:
                        with Image.open(io.BytesIO(input_img['bytes'])) as img:
                            img = img.convert('RGB')
                            img.save(input_path, 'JPEG', quality=95)
                    elif 'path' in input_img and os.path.exists(input_img['path']):
                        with Image.open(input_img['path']) as img:
                            img = img.convert('RGB')
                            img.save(input_path, 'JPEG', quality=95)
                else:
                    # Try to handle other input types (e.g., PIL Image, numpy array)
                    if hasattr(input_img, 'save'):
                        input_img.save(input_path, 'JPEG', quality=95)
                    else:
                        print(f"Warning: Could not save input image {i}")
                        continue
            except Exception as e:
                print(f"Error saving input image {i}: {str(e)}")
                continue
                
            # Save ground truth if available
            gt_path = None
            if 'edited_image' in item:
                gt_img = item['edited_image']
                gt_path = os.path.join(inputs_dir, f'gt_{i:04d}.jpg')
                try:
                    if isinstance(gt_img, dict):
                        if 'bytes' in gt_img:
                            with Image.open(io.BytesIO(gt_img['bytes'])) as img:
                                img = img.convert('RGB')
                                img.save(gt_path, 'JPEG', quality=95)
                        elif 'path' in gt_img and os.path.exists(gt_img['path']):
                            with Image.open(gt_img['path']) as img:
                                img = img.convert('RGB')
                                img.save(gt_path, 'JPEG', quality=95)
                    elif hasattr(gt_img, 'save'):
                        gt_img.save(gt_path, 'JPEG', quality=95)
                except Exception as e:
                    print(f"Error saving ground truth {i}: {str(e)}")
            
            # Generate prediction
            output_path = os.path.join(pred_dir, f'pred_{i:04d}.png')
            
            # Generate the image
            try:
                gen_path = generate_image(
                    pipe=pipe,
                    prompt=prompt,
                    image_path=input_path,
                    strength=DEFAULT_STRENGTH,
                    steps=DEFAULT_IMG2IMG_STEPS,
                    guidance_scale=DEFAULT_GUIDANCE_SCALE,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    resolution=DEFAULT_EVAL_RESOLUTION,
                    output_path=output_path
                )
                
                if gen_path and os.path.exists(gen_path):
                    # Save prompt to file
                    with open(prompts_file, 'a', encoding='utf-8') as f:
                        f.write(f"{i}: {prompt}\n")
                    
                    predictions.append({
                        'input': input_path,
                        'prediction': gen_path,
                        'ground_truth': gt_path if gt_path and os.path.exists(gt_path) else None,
                        'prompt': prompt
                    })
                else:
                    print(f"Warning: Failed to generate prediction for sample {i}")
                    
            except Exception as e:
                print(f"Error generating prediction for sample {i}: {str(e)}")
                continue
                
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nGenerated {len(predictions)} predictions")
    return predictions
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    
    # Create a file to save all prompts
    prompts_file = os.path.join(output_dir, 'prompts.txt')
    with open(prompts_file, 'w', encoding='utf-8') as f:
        f.write("Index\tPrompt\n")
    
    predictions = []
    
    # Load the model
    print("\nLoading model for evaluation...")
    pipe = load_model(use_img2img=True)  # We're doing image-to-image translation
    pipe = pipe.to(device)  # Move model to the specified device
    
    # Process each sample
    for i, item in enumerate(tqdm(dataset, desc="Generating predictions")):
        try:
            # Get the prompt and input image
            prompt = item.get('edit_prompt', '')
            input_img = item.get('input_image')
            
            if not prompt or not input_img:
                print(f"Warning: Missing prompt or input image for sample {i}")
                continue
                
            # Save input image
            input_path = os.path.join(inputs_dir, f'input_{i:04d}.jpg')
            try:
                if isinstance(input_img, dict):
                    if 'bytes' in input_img:
                        with Image.open(io.BytesIO(input_img['bytes'])) as img:
                            img = img.convert('RGB')
                            img.save(input_path, 'JPEG', quality=95)
                    elif 'path' in input_img and os.path.exists(input_img['path']):
                        with Image.open(input_img['path']) as img:
                            img = img.convert('RGB')
                            img.save(input_path, 'JPEG', quality=95)
                elif hasattr(input_img, 'save'):
                    input_img.convert('RGB').save(input_path, 'JPEG', quality=95)
                else:
                    print(f"Warning: Could not save input image {i}")
                    continue
            except Exception as e:
                print(f"Error saving input image {i}: {str(e)}")
                continue
                
            # Save ground truth if available
            gt_path = None
            if 'edited_image' in item:
                gt_img = item['edited_image']
                gt_path = os.path.join(inputs_dir, f'gt_{i:04d}.jpg')
                try:
                    if isinstance(gt_img, dict):
                        if 'bytes' in gt_img:
                            with Image.open(io.BytesIO(gt_img['bytes'])) as img:
                                img = img.convert('RGB')
                                img.save(gt_path, 'JPEG', quality=95)
                        elif 'path' in gt_img and os.path.exists(gt_img['path']):
                            with Image.open(gt_img['path']) as img:
                                img = img.convert('RGB')
                                img.save(gt_path, 'JPEG', quality=95)
                    elif hasattr(gt_img, 'save'):
                        gt_img.convert('RGB').save(gt_path, 'JPEG', quality=95)
                except Exception as e:
                    print(f"Error saving ground truth {i}: {str(e)}")
            
            # Generate prediction
            output_path = os.path.join(pred_dir, f'pred_{i:04d}.png')
            
            # Generate the image
            try:
                gen_path = generate_image(
                    pipe=pipe,
                    prompt=prompt,
                    image_path=input_path,
                    strength=DEFAULT_STRENGTH,
                    steps=DEFAULT_IMG2IMG_STEPS,
                    guidance_scale=DEFAULT_GUIDANCE_SCALE,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    resolution=DEFAULT_EVAL_RESOLUTION,
                    output_path=output_path
                )
                
                if gen_path and os.path.exists(gen_path):
                    # Save prompt to file
                    with open(prompts_file, 'a', encoding='utf-8') as f:
                        f.write(f"{i}: {prompt}\n")
                    
                    predictions.append({
                        'input': input_path,
                        'prediction': gen_path,
                        'ground_truth': gt_path if gt_path and os.path.exists(gt_path) else None,
                        'prompt': prompt
                    })
                else:
                    print(f"Warning: Failed to generate prediction for sample {i}")
                    
            except Exception as e:
                print(f"Error generating prediction for sample {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
                
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nGenerated {len(predictions)} predictions")
    return predictions


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Stable Diffusion model on InstructPix2Pix dataset')
    
    # Model parameters
    parser.add_argument('--model_id', type=str, default=BASE_MODEL_ID,
                      help=f'Model ID from Hugging Face (default: {BASE_MODEL_ID})')
    
    # Dataset parameters
    parser.add_argument('--num_samples', type=int, default=DEFAULT_NUM_EVAL_SAMPLES,
                      help=f'Number of samples to evaluate on (default: {DEFAULT_NUM_EVAL_SAMPLES})')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default=EVAL_DIR,
                      help=f'Directory to save outputs (default: {EVAL_DIR})')
    
    # Device and performance
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE,
                      help=f'Device to use (default: {DEFAULT_DEVICE})')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for generation (default: 1)')
    
    # Generation parameters
    parser.add_argument('--strength', type=float, default=DEFAULT_STRENGTH,
                      help=f'Strength for image-to-image (default: {DEFAULT_STRENGTH})')
    parser.add_argument('--steps', type=int, default=None,
                      help=f'Number of inference steps (default: {DEFAULT_IMG2IMG_STEPS} for img2img)')
    
    # Output options
    parser.add_argument('--save_results', action='store_true',
                      help='Save results to a JSON file (default: False)')
    
    return parser.parse_args()

def save_results(metrics, output_dir, predictions=None):
    """Save evaluation results to a JSON file
    
    Args:
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save results
        predictions: Optional list of prediction details
    """
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results dictionary
    results = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'config': {
            'model': BASE_MODEL_ID,
            'resolution': DEFAULT_EVAL_RESOLUTION,
            'num_samples': metrics.get('num_samples', 0)
        },
        'predictions': predictions if predictions is not None else []
    }
    
    # Add predictions if provided
    if predictions is not None:
        results['predictions'] = [
            {k: v for k, v in pred.items() if k != 'ground_truth'}
            for pred in predictions
        ]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'eval_results_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return output_file

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Print header
    print("\n" + "="*50)
    print(f"Evaluating model: {BASE_MODEL_ID}")
    print(f"Device: {args.device}")
    print("="*50)
    
    # Download dataset
    try:
        print(f"\nDownloading dataset (samples: {args.num_samples})...")
        dataset = download_instructpix2pix(num_samples=args.num_samples)
        
        if not dataset or len(dataset) == 0:
            print("Error: No data samples found in the dataset")
            return
            
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return
    
    # Generate predictions
    print(f"\nGenerating predictions (batch_size={args.batch_size})...")
    predictions = generate_predictions(
        dataset,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size
    )
    
    if not predictions:
        print("No predictions were generated. Exiting...")
        return
    
    # Calculate and print metrics
    metrics = calculate_metrics(predictions, device=args.device)
    print_metrics(metrics, BASE_MODEL_ID)
    
    # Save detailed results if requested
    if args.save_results:
        save_results(metrics, args.output_dir, predictions)
    
    return metrics

if __name__ == "__main__":
    # Set up error handling
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
