"""
Test your trained gaze estimation model on test data.
Auto-detects model configuration (bins, binwidth, angle) from checkpoint.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from argparse import Namespace
import matplotlib
# Use 'Agg' backend for headless environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from utils.helpers import get_model, get_dataloader, angular_error, gaze_to_3d


def test_model(model_path, dataset="mpiigaze", arch="resnet18", data_dir="MPIIGaze"):
    """
    Test a trained gaze estimation model.
    Auto-detects configuration from checkpoint.
    
    Args:
        model_path: Path to saved model (.pt file)
        dataset: Dataset name (for loading test data)
        arch: Model architecture
        data_dir: Path to data directory
    """
    print(f"\n{'='*80}")
    print(f"TESTING GAZE ESTIMATION MODEL")
    print(f"{'='*80}\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {model_path}\n")
    
    # Load checkpoint to extract configuration
    print("Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if checkpoint contains config info
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        bins = checkpoint.get('bins', 28)
        binwidth = checkpoint.get('binwidth', 3.0)
        angle = checkpoint.get('angle', 42.0)
        model_state = checkpoint['model_state_dict']
        print("✓ Loaded configuration from checkpoint")
    else:
        # Old format - detect from model structure
        print("No config in checkpoint, detecting from model structure...")
        model_state = checkpoint
        # Infer bins from model structure
        if 'fc_pitch.weight' in model_state:
            bins = model_state['fc_pitch.weight'].shape[0]
            print(f"✓ Detected {bins} bins from model structure")
            # Set corresponding config
            if bins == 28:
                binwidth = 3.0
                angle = 42.0
                print("  Using Gaze360 configuration")
            elif bins == 90:
                binwidth = 4.0
                angle = 180.0
                print("  Using MPIIGaze configuration")
            else:
                print(f"⚠ Unknown bin count, using default MPIIGaze config")
                binwidth = 4.0
                angle = 180.0
        else:
            # Fallback
            bins = 28
            binwidth = 3.0
            angle = 42.0
            print("⚠ Could not detect bins, using Gaze360 defaults")
    
    print(f"\nModel Configuration:")
    print(f"  bins: {bins}")
    print(f"  binwidth: {binwidth}°")
    print(f"  angle range: [{-angle:.1f}°, {angle - binwidth:.1f}°]")
    print(f"  total range: {2 * angle}°")
    print()
    
    params = Namespace(
        data=data_dir,
        dataset=dataset,
        bins=bins,
        binwidth=binwidth,
        angle=angle,
        batch_size=64,
        num_workers=4,
        arch=arch
    )
    
    # Load model with correct configuration
    print("Initializing model...")
    model = get_model(arch, bins).to(device)
    model.load_state_dict(model_state)
    model.eval()
    print("✓ Model loaded successfully\n")
    
    # Load test data
    print("Loading test data...")
    test_loader = get_dataloader(params, mode="train")
    print(f"✓ Test samples: {len(test_loader.dataset):,}\n")
    
    # Prepare for evaluation
    idx_tensor = torch.arange(bins, device=device, dtype=torch.float32)
    
    # Metrics storage
    errors = []
    pitch_errors = []
    yaw_errors = []
    predictions = []
    ground_truths = []
    out_of_range_count = 0
    
    print("Running inference...")
    with torch.no_grad():
        for images, labels_gaze, regression_labels_gaze, meta in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            
            # Ground truth angles (in degrees first)
            gt_pitch_deg = regression_labels_gaze[:, 0].numpy()
            gt_yaw_deg = regression_labels_gaze[:, 1].numpy()
            
            # Check if ground truth is within model's range
            for i in range(len(gt_pitch_deg)):
                if abs(gt_pitch_deg[i]) > angle or abs(gt_yaw_deg[i]) > angle:
                    out_of_range_count += 1
            
            # Convert to radians
            gt_pitch = np.radians(gt_pitch_deg)
            gt_yaw = np.radians(gt_yaw_deg)
            
            # Forward pass
            pitch_out, yaw_out = model(images)
            
            # Convert logits to angles
            pitch_probs = F.softmax(pitch_out, dim=1)
            yaw_probs = F.softmax(yaw_out, dim=1)
            
            pitch_pred_deg = torch.sum(pitch_probs * idx_tensor, 1) * binwidth - angle
            yaw_pred_deg = torch.sum(yaw_probs * idx_tensor, 1) * binwidth - angle
            
            pitch_pred_rad = np.radians(pitch_pred_deg.cpu().numpy())
            yaw_pred_rad = np.radians(yaw_pred_deg.cpu().numpy())
            
            # Calculate errors
            for p_pred, y_pred, p_gt, y_gt in zip(pitch_pred_rad, yaw_pred_rad, gt_pitch, gt_yaw):
                # 3D angular error
                pred_vec = gaze_to_3d([p_pred, y_pred])
                gt_vec = gaze_to_3d([p_gt, y_gt])
                error_deg = angular_error(pred_vec, gt_vec)
                errors.append(error_deg)
                
                # Individual axis errors
                pitch_error = np.degrees(abs(p_pred - p_gt))
                yaw_error = np.degrees(abs(y_pred - y_gt))
                pitch_errors.append(pitch_error)
                yaw_errors.append(yaw_error)
                
                # Store for analysis
                predictions.append([np.degrees(p_pred), np.degrees(y_pred)])
                ground_truths.append([np.degrees(p_gt), np.degrees(y_gt)])
    
    # Convert to numpy arrays
    errors = np.array(errors)
    pitch_errors = np.array(pitch_errors)
    yaw_errors = np.array(yaw_errors)
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"TEST RESULTS")
    print(f"{'='*80}\n")
    
    print(f"Total test samples: {len(errors):,}")
    if out_of_range_count > 0:
        print(f"⚠ Samples outside model range: {out_of_range_count:,} ({out_of_range_count/len(errors)*100:.1f}%)")
        print(f"  (Model range: ±{angle}°, these were clipped during training)")
    print()
    
    print("3D Angular Error (degrees):")
    print(f"  Mean:   {errors.mean():.4f}°")
    print(f"  Median: {np.median(errors):.4f}°")
    print(f"  Std:    {errors.std():.4f}°")
    print(f"  Min:    {errors.min():.4f}°")
    print(f"  Max:    {errors.max():.4f}°")
    print()
    
    print("Pitch Error (degrees):")
    print(f"  Mean:   {pitch_errors.mean():.4f}°")
    print(f"  Median: {np.median(pitch_errors):.4f}°")
    print(f"  Std:    {pitch_errors.std():.4f}°")
    print()
    
    print("Yaw Error (degrees):")
    print(f"  Mean:   {yaw_errors.mean():.4f}°")
    print(f"  Median: {np.median(yaw_errors):.4f}°")
    print(f"  Std:    {yaw_errors.std():.4f}°")
    print()
    
    # Accuracy at different thresholds
    print("Accuracy at Different Thresholds:")
    for threshold in [1, 2, 3, 5, 10]:
        accuracy = (errors <= threshold).mean() * 100
        print(f"  Within {threshold}°:  {accuracy:.2f}%")
    print()
    
    # Percentiles
    print("Error Percentiles:")
    for percentile in [25, 50, 75, 90, 95, 99]:
        value = np.percentile(errors, percentile)
        print(f"  {percentile}th percentile: {value:.4f}°")
    print()
    
    # Screen mapping context
    print("Practical Interpretation (for screen gaze tracking):")
    print(f"  At 60cm from a 24\" monitor:")
    mean_cm = 60 * np.tan(np.radians(errors.mean()))
    print(f"    Mean error ≈ {mean_cm:.1f} cm on screen")
    print(f"    95th percentile ≈ {np.percentile(errors, 95) * 60 * np.tan(np.radians(1)) * 100:.1f} cm on screen")
    print()
    
    # Visualization
    plot_results(errors, pitch_errors, yaw_errors, predictions, ground_truths, bins, angle)
    
    return {
        'mean_error': errors.mean(),
        'median_error': np.median(errors),
        'std_error': errors.std(),
        'errors': errors,
        'pitch_errors': pitch_errors,
        'yaw_errors': yaw_errors,
        'predictions': predictions,
        'ground_truths': ground_truths,
        'bins': bins,
        'angle': angle
    }


def plot_results(errors, pitch_errors, yaw_errors, predictions, ground_truths, bins, angle):
    """Create visualization plots for test results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Error distribution histogram
    axes[0, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(errors.mean(), color='red', linestyle='--', 
                       label=f'Mean: {errors.mean():.2f}°')
    axes[0, 0].axvline(np.median(errors), color='green', linestyle='--',
                       label=f'Median: {np.median(errors):.2f}°')
    axes[0, 0].set_xlabel('Angular Error (degrees)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Angular Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Pitch errors
    axes[0, 1].hist(pitch_errors, bins=50, edgecolor='black', alpha=0.7, color='blue')
    axes[0, 1].axvline(pitch_errors.mean(), color='red', linestyle='--',
                       label=f'Mean: {pitch_errors.mean():.2f}°')
    axes[0, 1].set_xlabel('Pitch Error (degrees)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Pitch Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Yaw errors
    axes[0, 2].hist(yaw_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 2].axvline(yaw_errors.mean(), color='red', linestyle='--',
                       label=f'Mean: {yaw_errors.mean():.2f}°')
    axes[0, 2].set_xlabel('Yaw Error (degrees)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Yaw Error Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # 4. Pitch: Predicted vs Ground Truth
    axes[1, 0].scatter(ground_truths[:, 0], predictions[:, 0], alpha=0.3, s=1)
    lim = max(abs(ground_truths[:, 0].min()), abs(ground_truths[:, 0].max()),
              abs(predictions[:, 0].min()), abs(predictions[:, 0].max()))
    axes[1, 0].plot([-lim, lim], [-lim, lim], 'r--', label='Perfect prediction')
    axes[1, 0].axvline(-angle, color='orange', linestyle=':', alpha=0.5, label=f'Model range: ±{angle}°')
    axes[1, 0].axvline(angle, color='orange', linestyle=':', alpha=0.5)
    axes[1, 0].axhline(-angle, color='orange', linestyle=':', alpha=0.5)
    axes[1, 0].axhline(angle, color='orange', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('Ground Truth Pitch (degrees)')
    axes[1, 0].set_ylabel('Predicted Pitch (degrees)')
    axes[1, 0].set_title('Pitch: Prediction vs Ground Truth')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].axis('equal')
    
    # 5. Yaw: Predicted vs Ground Truth
    axes[1, 1].scatter(ground_truths[:, 1], predictions[:, 1], alpha=0.3, s=1)
    lim = max(abs(ground_truths[:, 1].min()), abs(ground_truths[:, 1].max()),
              abs(predictions[:, 1].min()), abs(predictions[:, 1].max()))
    axes[1, 1].plot([-lim, lim], [-lim, lim], 'r--', label='Perfect prediction')
    axes[1, 1].axvline(-angle, color='orange', linestyle=':', alpha=0.5, label=f'Model range: ±{angle}°')
    axes[1, 1].axvline(angle, color='orange', linestyle=':', alpha=0.5)
    axes[1, 1].axhline(-angle, color='orange', linestyle=':', alpha=0.5)
    axes[1, 1].axhline(angle, color='orange', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('Ground Truth Yaw (degrees)')
    axes[1, 1].set_ylabel('Predicted Yaw (degrees)')
    axes[1, 1].set_title('Yaw: Prediction vs Ground Truth')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].axis('equal')
    
    # 6. Cumulative error distribution
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    axes[1, 2].plot(sorted_errors, cumulative, linewidth=2)
    axes[1, 2].set_xlabel('Angular Error (degrees)')
    axes[1, 2].set_ylabel('Cumulative Percentage (%)')
    axes[1, 2].set_title('Cumulative Error Distribution')
    axes[1, 2].grid(alpha=0.3)
    axes[1, 2].axhline(95, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].axvline(np.percentile(errors, 95), color='red', 
                       linestyle='--', alpha=0.5,
                       label=f'95th percentile: {np.percentile(errors, 95):.2f}°')
    axes[1, 2].legend()
    
    plt.suptitle(f'Model Test Results ({bins} bins, ±{angle}° range)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plots saved to 'test_results.png'")
    plt.show()


if __name__ == "__main__":
    # Test your model - will auto-detect configuration
    model_path = "output/mpiigaze_resnet18_1759874086/best_model.pt"
    
    results = test_model(
        model_path=model_path,
        dataset="mpiigaze",
        arch="resnet18",
        data_dir="MPIIGaze"
    )
    
    print(f"\n{'='*80}")
    print("Testing complete!")
    print(f"Mean Angular Error: {results['mean_error']:.4f}°")
    print(f"{'='*80}\n")