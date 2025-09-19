import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchvision.transforms.functional import to_tensor
from PIL import Image

# Initialize LPIPS model once to avoid reloading
_lpips_model = None

def get_lpips_model(device='cuda'):
    """Get or initialize LPIPS model."""
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    return _lpips_model

def preprocess_image(image, target_size=(299, 299)):
    """Preprocess image for metric calculation.
    
    Args:
        image: PIL Image or tensor
        target_size: Target size for resizing
        
    Returns:
        Preprocessed tensor
    """
    transform = Compose([
        Resize(target_size),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image or file path")
        
    return transform(image)

def calculate_metrics(predictions, device=None):
    """Calculate evaluation metrics for generated images.
    
    Args:
        predictions: List of dicts with 'prediction' and 'ground_truth' paths
        device: Device to run calculations on (cpu/cuda)
        
    Returns:
        Dictionary with metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize metrics
    metrics = {
        'num_samples': len(predictions),
        'fid': float('nan'),
        'fid_message': 'N/A (requires at least 2 samples)',
        'inception_score_mean': float('nan'),
        'inception_score_std': float('nan'),
        'inception_score_message': 'N/A (requires at least 2 samples for meaningful score)',
        'ssim': float('nan'),
        'ssim_message': 'N/A (requires ground truth)',
        'psnr': float('nan'),
        'psnr_message': 'N/A (requires ground truth)',
        'lpips': float('nan'),
        'lpips_message': 'N/A (requires ground truth)',
        'mse': float('nan'),
        'mse_message': 'N/A (requires ground truth)'
    }
    
    if len(predictions) < 2:
        return metrics
        
    try:
        # Initialize metrics
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        inception = InceptionScore().to(device)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)  # data_range=2.0 because our images are in [-1, 1] range
        lpips_model = get_lpips_model(device)
        
        # Initialize accumulators
        ssim_values = []
        psnr_values = []
        lpips_values = []
        mse_values = []
        
        # Process each prediction
        for pred in tqdm(predictions, desc="Processing samples for metrics"):
            try:
                # Debug ground truth path
                gt_path = pred.get('ground_truth')
                print(f"\nDebug - Checking ground truth for prediction {pred.get('prediction')}")
                print(f"Ground truth path: {gt_path}")
                if gt_path:
                    print(f"File exists: {os.path.exists(gt_path)}")
                
                # Skip if ground truth is not available
                if gt_path is None or not os.path.exists(gt_path):
                    print("Skipping - No valid ground truth available")
                    continue
                    
                # Load and preprocess prediction image
                pred_img = Image.open(pred['prediction']).convert('RGB')
                pred_tensor = preprocess_image(pred_img).unsqueeze(0).to(device)
                
                # Convert to uint8 for metrics calculation
                pred_tensor_uint8 = (pred_tensor * 255).byte()
                
                # Load and preprocess ground truth image
                gt_img = Image.open(pred['ground_truth']).convert('RGB')
                gt_tensor = preprocess_image(gt_img).unsqueeze(0).to(device)
                
                # Convert to uint8 for metrics calculation
                gt_tensor_uint8 = (gt_tensor * 255).byte()
                
                # Update FID with both real and generated images
                fid.update(gt_tensor_uint8, real=True)
                fid.update(pred_tensor_uint8, real=False)
                
                # Calculate SSIM (requires float32 in [0,1] range)
                if pred_tensor.shape == gt_tensor.shape:
                    ssim_val = ssim(pred_tensor, gt_tensor)
                    ssim_values.append(ssim_val.item())
                
                # Calculate PSNR (handles different ranges)
                if pred_tensor.shape == gt_tensor.shape:
                    psnr_val = psnr(pred_tensor, gt_tensor)
                    psnr_values.append(psnr_val.item())
                
                # Calculate LPIPS (perceptual similarity)
                if pred_tensor.shape == gt_tensor.shape:
                    lpips_val = lpips_model(pred_tensor, gt_tensor)
                    lpips_values.append(lpips_val.item())
                
                # Calculate MSE
                if pred_tensor.shape == gt_tensor.shape:
                    mse = F.mse_loss(pred_tensor, gt_tensor)
                    mse_values.append(mse.item())
                
                # Update Inception Score with prediction
                inception.update(pred_tensor_uint8)
                
            except Exception as e:
                print(f"Error processing {pred.get('prediction', 'unknown')}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Compute metrics if we have ground truth
        has_ground_truth = any('ground_truth' in p and p['ground_truth'] is not None and os.path.exists(p['ground_truth']) for p in predictions)
        
        if has_ground_truth:
            # FID
            metrics['fid'] = fid.compute().item()
            metrics['fid_message'] = 'Computed successfully'
            
            # SSIM
            if ssim_values:
                metrics['ssim'] = np.mean(ssim_values)
                metrics['ssim_message'] = 'Computed successfully'
            
            # PSNR
            if psnr_values:
                metrics['psnr'] = np.mean(psnr_values)
                metrics['psnr_message'] = 'Computed successfully'
            
            # LPIPS
            if lpips_values:
                metrics['lpips'] = np.mean(lpips_values)
                metrics['lpips_message'] = 'Computed successfully (lower is better)'
            
            # MSE
            if mse_values:
                metrics['mse'] = np.mean(mse_values)
                metrics['mse_message'] = 'Computed successfully (lower is better)'
        else:
            metrics['fid_message'] = 'N/A (no ground truth available)'
        
        # Compute Inception Score (doesn't need ground truth)
        is_mean, is_std = inception.compute()
        metrics.update({
            'inception_score_mean': is_mean.item() if not torch.isnan(is_mean) else float('nan'),
            'inception_score_std': is_std.item() if not torch.isnan(is_std) else float('nan'),
            'inception_score_message': 'Computed successfully'
        })
        
    except Exception as e:
        metrics.update({
            'fid_message': f'Error: {str(e)}',
            'inception_score_message': f'Error: {str(e)}'
        })
        print(f"Error calculating metrics: {str(e)}")
    
    return metrics

def print_metrics(metrics, model_id):
    """Print formatted metrics to console.
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics()
        model_id: Model identifier for the report
    """
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {model_id}")
    print(f"Samples evaluated: {metrics['num_samples']}")
    
    # FID
    print("\nFrechet Inception Distance (FID):")
    print(f"  Score: {metrics['fid']:.2f}")
    print(f"  Note: {metrics['fid_message']}")
    
    # Inception Score
    print("\nInception Score:")
    print(f"  Score: {metrics['inception_score_mean']:.2f} ± {metrics['inception_score_std']:.2f}")
    print(f"  Note: {metrics['inception_score_message']}")
    
    # SSIM
    print("\nStructural Similarity Index (SSIM):")
    print(f"  Score: {metrics['ssim']:.4f}" if not np.isnan(metrics['ssim']) else "  Score: N/A")
    print(f"  Note: {metrics['ssim_message']} (range: -1 to 1, higher is better)")
    
    # PSNR
    print("\nPeak Signal-to-Noise Ratio (PSNR):")
    print(f"  Score: {metrics['psnr']:.2f} dB" if not np.isnan(metrics['psnr']) else "  Score: N/A")
    print(f"  Note: {metrics['psnr_message']} (higher is better)")
    
    # LPIPS
    print("\nLearned Perceptual Image Patch Similarity (LPIPS):")
    print(f"  Score: {metrics['lpips']:.4f}" if not np.isnan(metrics['lpips']) else "  Score: N/A")
    print(f"  Note: {metrics['lpips_message']} (range: 0 to 1, lower is better)")
    
    # MSE
    print("\nMean Squared Error (MSE):")
    print(f"  Score: {metrics['mse']:.6f}" if not np.isnan(metrics['mse']) else "  Score: N/A")
    print(f"  Note: {metrics['mse_message']} (range: 0 to ∞, lower is better)")
    
    print("\nNote: For meaningful metrics, use at least 10-50 samples.")
    print("FID, SSIM, PSNR, LPIPS, and MSE require ground truth images for comparison.")
    print("="*50)
