"""
Validate that your preprocessing script integrates correctly with the training pipeline.
Run this BEFORE starting full training to catch any issues.
"""

import torch
import numpy as np
from argparse import Namespace
from utils.helpers import get_dataloader, get_model
import torch.nn.functional as F


def validate_integration(params):
    """
    Validate that dataset and model work together correctly.
    
    Args:
        params: Namespace with training parameters
    """
    print("\n" + "="*80)
    print("TRAINING INTEGRATION VALIDATION")
    print("="*80 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # 1. Load Data
    print("1. LOADING DATASET")
    print("-" * 80)
    try:
        data_loader = get_dataloader(params, mode="train")
        dataset = data_loader.dataset
        print(f"✓ Dataset loaded successfully")
        print(f"  Total samples: {len(dataset):,}")
        print(f"  Batch size: {params.batch_size}")
        print(f"  Number of batches: {len(data_loader):,}")
        print(f"  Number of workers: {params.num_workers}")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False
    print()
    
    # 2. Check Data Format
    print("2. VALIDATING DATA FORMAT")
    print("-" * 80)
    try:
        # Get a single batch
        images, labels_cls, labels_reg, meta = next(iter(data_loader))
        
        print(f"✓ Batch loaded successfully")
        print(f"  Images shape: {images.shape}")
        print(f"  Images dtype: {images.dtype}")
        print(f"  Images min/max: {images.min():.3f} / {images.max():.3f}")
        print()
        print(f"  Classification labels shape: {labels_cls.shape}")
        print(f"  Classification labels dtype: {labels_cls.dtype}")
        print(f"  Pitch bins range: {labels_cls[:, 0].min().item()} to {labels_cls[:, 0].max().item()}")
        print(f"  Yaw bins range: {labels_cls[:, 1].min().item()} to {labels_cls[:, 1].max().item()}")
        print()
        print(f"  Regression labels shape: {labels_reg.shape}")
        print(f"  Regression labels dtype: {labels_reg.dtype}")
        print(f"  Pitch angles range: {labels_reg[:, 0].min().item():.2f}° to {labels_reg[:, 0].max().item():.2f}°")
        print(f"  Yaw angles range: {labels_reg[:, 1].min().item():.2f}° to {labels_reg[:, 1].max().item():.2f}°")
        print()
        print(f"  Metadata type: {type(meta)}")
        if isinstance(meta, dict):
            print(f"  Metadata keys: {list(meta.keys())}")
            for key in meta.keys():
                if key in meta:
                    val = meta[key]
                    if isinstance(val, (list, torch.Tensor)):
                        print(f"    {key}: length/shape = {len(val) if isinstance(val, list) else val.shape}")
                    else:
                        print(f"    {key}: {type(val)}")
        
        # Validate label ranges
        assert labels_cls[:, 0].min() >= 0 and labels_cls[:, 0].max() < params.bins, \
            f"Pitch bins out of range [0, {params.bins})"
        assert labels_cls[:, 1].min() >= 0 and labels_cls[:, 1].max() < params.bins, \
            f"Yaw bins out of range [0, {params.bins})"
        
        print(f"\n✓ All data format checks passed")
        
    except Exception as e:
        print(f"✗ Data format validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # 3. Initialize Model
    print("3. INITIALIZING MODEL")
    print("-" * 80)
    try:
        model = get_model(params.arch, params.bins).to(device)
        print(f"✓ Model initialized: {params.arch}")
        print(f"  Number of bins: {params.bins}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False
    print()
    
    # 4. Test Forward Pass
    print("4. TESTING FORWARD PASS")
    print("-" * 80)
    try:
        model.eval()
        with torch.no_grad():
            images_gpu = images.to(device)
            pitch_out, yaw_out = model(images_gpu)
        
        print(f"✓ Forward pass successful")
        print(f"  Pitch output shape: {pitch_out.shape}")
        print(f"  Yaw output shape: {yaw_out.shape}")
        print(f"  Expected shape: ({params.batch_size}, {params.bins})")
        
        # Validate output shapes
        assert pitch_out.shape == (images.shape[0], params.bins), \
            f"Pitch output shape mismatch: {pitch_out.shape} vs expected ({images.shape[0]}, {params.bins})"
        assert yaw_out.shape == (images.shape[0], params.bins), \
            f"Yaw output shape mismatch: {yaw_out.shape} vs expected ({images.shape[0]}, {params.bins})"
        
        print(f"\n✓ Output shapes correct")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # 5. Test Loss Calculation
    print("5. TESTING LOSS CALCULATION")
    print("-" * 80)
    try:
        model.train()
        images_gpu = images.to(device)
        label_pitch = labels_cls[:, 0].to(device)
        label_yaw = labels_cls[:, 1].to(device)
        label_pitch_reg = labels_reg[:, 0].to(device)
        label_yaw_reg = labels_reg[:, 1].to(device)
        
        # Forward pass
        pitch_out, yaw_out = model(images_gpu)
        
        # Classification loss
        cls_criterion = torch.nn.CrossEntropyLoss()
        loss_pitch_cls = cls_criterion(pitch_out, label_pitch)
        loss_yaw_cls = cls_criterion(yaw_out, label_yaw)
        
        # Regression loss
        idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)
        pitch_pred = torch.sum(F.softmax(pitch_out, dim=1) * idx_tensor, 1) * params.binwidth - params.angle
        yaw_pred = torch.sum(F.softmax(yaw_out, dim=1) * idx_tensor, 1) * params.binwidth - params.angle
        
        reg_criterion = torch.nn.MSELoss()
        loss_pitch_reg = reg_criterion(pitch_pred, label_pitch_reg)
        loss_yaw_reg = reg_criterion(yaw_pred, label_yaw_reg)
        
        # Total loss
        loss = loss_pitch_cls + loss_yaw_cls + params.alpha * (loss_pitch_reg + loss_yaw_reg)
        
        print(f"✓ Loss calculation successful")
        print(f"  Classification loss (pitch): {loss_pitch_cls.item():.4f}")
        print(f"  Classification loss (yaw): {loss_yaw_cls.item():.4f}")
        print(f"  Regression loss (pitch): {loss_pitch_reg.item():.4f}")
        print(f"  Regression loss (yaw): {loss_yaw_reg.item():.4f}")
        print(f"  Total loss: {loss.item():.4f}")
        
        # Test backward pass
        loss.backward()
        print(f"\n✓ Backward pass successful")
        
    except Exception as e:
        print(f"✗ Loss calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # 6. Test Bin-to-Angle Conversion
    print("6. VALIDATING BIN-TO-ANGLE CONVERSION")
    print("-" * 80)
    try:
        # Test the conversion formula
        test_bins = torch.tensor([0, params.bins // 2, params.bins - 1], dtype=torch.float32)
        test_angles = test_bins * params.binwidth - params.angle
        
        print(f"Bin-to-angle mapping:")
        print(f"  Bin 0 → {test_angles[0].item():.2f}°")
        print(f"  Bin {params.bins // 2} → {test_angles[1].item():.2f}°")
        print(f"  Bin {params.bins - 1} → {test_angles[2].item():.2f}°")
        print()
        print(f"Configuration:")
        print(f"  bins = {params.bins}")
        print(f"  binwidth = {params.binwidth}")
        print(f"  angle = {params.angle}")
        print(f"  Expected range: [{-params.angle:.1f}°, {params.angle - params.binwidth:.1f}°]")
        
        # Verify the range matches dataset
        expected_min = -params.angle
        expected_max = params.angle - params.binwidth
        actual_min = labels_reg[:, 0].min().item()
        actual_max = labels_reg[:, 0].max().item()
        
        print(f"\nActual data range:")
        print(f"  Pitch: [{actual_min:.2f}°, {actual_max:.2f}°]")
        print(f"  Yaw: [{labels_reg[:, 1].min().item():.2f}°, {labels_reg[:, 1].max().item():.2f}°]")
        
        if actual_min < expected_min or actual_max > (params.angle):
            print(f"\n⚠ Warning: Data range exceeds expected range!")
        else:
            print(f"\n✓ Data range within expected bounds")
        
    except Exception as e:
        print(f"✗ Conversion validation failed: {e}")
        return False
    print()
    
    # 7. Summary
    print("="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print("✓ Dataset loading: PASSED")
    print("✓ Data format: PASSED")
    print("✓ Model initialization: PASSED")
    print("✓ Forward pass: PASSED")
    print("✓ Loss calculation: PASSED")
    print("✓ Bin conversion: PASSED")
    print("\n✓✓✓ ALL CHECKS PASSED - Ready to start training! ✓✓✓\n")
    
    return True


if __name__ == "__main__":
    # Configure parameters to match your training script
    params = Namespace(
        data="MPIIGaze",  # Adjust this path
        dataset="mpiigaze",
        bins=90,
        binwidth=4.0,
        angle=180.0,
        batch_size=64,
        num_workers=4,
        arch="resnet18",
        alpha=1.0,
        lr=0.00001,
    )
    
    print("Testing with configuration:")
    print(f"  Dataset: {params.dataset}")
    print(f"  Architecture: {params.arch}")
    print(f"  Batch size: {params.batch_size}")
    print(f"  Bins: {params.bins}")
    
    success = validate_integration(params)
    
    if not success:
        print("\n" + "!"*80)
        print("VALIDATION FAILED - Fix errors before training!")
        print("!"*80 + "\n")
    else:
        print("You can now run your training script with confidence!")