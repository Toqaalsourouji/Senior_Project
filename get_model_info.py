"""
Model Architecture Inspector for Gaze Estimation
Run this before training to understand your model's structure.
"""

import torch
import torch.nn as nn
from torchsummary import torchsummary
from utils.helpers import get_model

def inspect_model(arch='resnet18', bins=90):
    """
    Inspect the gaze estimation model architecture.
    
    Args:
        arch: Architecture name (resnet18/34/50, mobilenetv2, mobileone_s0-s4)
        bins: Number of bins for gaze angle classification
    """
    print(f"\n{'='*80}")
    print(f"MODEL ARCHITECTURE INSPECTION: {arch.upper()}")
    print(f"{'='*80}\n")
    
    # Initialize model
    model = get_model(arch, bins)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 1. Model Structure
    print("1. MODEL STRUCTURE:")
    print("-" * 80)
    print(model)
    print()
    
    # 2. Total Parameters
    print("2. PARAMETER COUNT:")
    print("-" * 80)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")
    print()
    
    # 3. Layer-by-layer breakdown
    print("3. LAYER-BY-LAYER BREAKDOWN:")
    print("-" * 80)
    for name, param in model.named_parameters():
        print(f"{name:60s} | Shape: {str(list(param.shape)):30s} | Params: {param.numel():,}")
    print()
    
    # 4. Output heads
    print("4. OUTPUT ARCHITECTURE:")
    print("-" * 80)
    print(f"Number of bins: {bins}")
    print(f"Pitch output: {bins} classes (bins)")
    print(f"Yaw output: {bins} classes (bins)")
    print(f"Total output neurons: {bins * 2}")
    print()
    
    # 5. Input/Output Summary
    print("5. INPUT/OUTPUT SUMMARY:")
    print("-" * 80)
    print("Expected input shape: (batch_size, 3, H, W)")
    print("  - Typically: (batch_size, 3, 224, 224) for ResNet")
    print("  - Typically: (batch_size, 3, 224, 224) for MobileNet")
    print()
    print("Output shape:")
    print(f"  - Pitch: (batch_size, {bins}) - classification logits")
    print(f"  - Yaw: (batch_size, {bins}) - classification logits")
    print()
    
    # 6. Test forward pass
    print("6. TEST FORWARD PASS:")
    print("-" * 80)
    try:
        # Common input sizes for gaze estimation
        test_input = torch.randn(1, 3, 224, 224).to(device)
        model.eval()
        with torch.no_grad():
            pitch_out, yaw_out = model(test_input)
        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {list(test_input.shape)}")
        print(f"  Pitch output shape: {list(pitch_out.shape)}")
        print(f"  Yaw output shape: {list(yaw_out.shape)}")
        print()
        
        # Show sample predictions
        pitch_probs = torch.softmax(pitch_out, dim=1)
        yaw_probs = torch.softmax(yaw_out, dim=1)
        print(f"  Pitch probabilities sum: {pitch_probs.sum().item():.6f}")
        print(f"  Yaw probabilities sum: {yaw_probs.sum().item():.6f}")
        print(f"  Predicted pitch bin: {pitch_probs.argmax().item()}")
        print(f"  Predicted yaw bin: {yaw_probs.argmax().item()}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
    print()
    
    # 7. Memory Requirements
    print("7. MEMORY REQUIREMENTS (estimated):")
    print("-" * 80)
    batch_size = 64  # Your default batch size
    input_size = (3, 224, 224)
    
    # Input memory
    input_memory = batch_size * 3 * 224 * 224 * 4 / (1024**2)
    # Model memory
    model_memory = total_params * 4 / (1024**2)
    # Gradient memory (roughly same as model)
    grad_memory = trainable_params * 4 / (1024**2)
    # Optimizer state (Adam uses 2x parameters for momentum and velocity)
    optimizer_memory = trainable_params * 2 * 4 / (1024**2)
    # Activation memory (rough estimate)
    activation_memory = batch_size * 512 * 4 / (1024**2)  # Rough estimate
    
    total_memory = input_memory + model_memory + grad_memory + optimizer_memory + activation_memory
    
    print(f"Batch size: {batch_size}")
    print(f"Input memory: ~{input_memory:.2f} MB")
    print(f"Model weights: ~{model_memory:.2f} MB")
    print(f"Gradients: ~{grad_memory:.2f} MB")
    print(f"Optimizer states: ~{optimizer_memory:.2f} MB")
    print(f"Activations (approx): ~{activation_memory:.2f} MB")
    print(f"Total (approx): ~{total_memory:.2f} MB")
    print()
    
    # 8. Training Configuration
    print("8. TRAINING CONFIGURATION SUMMARY:")
    print("-" * 80)
    print("Loss Functions:")
    print("  - Classification: CrossEntropyLoss (for binned predictions)")
    print("  - Regression: MSELoss (for continuous angle predictions)")
    print("  - Combined: cls_loss + alpha * reg_loss")
    print()
    print("Optimizer: Adam")
    print("  - Default LR: 0.00001 (1e-5)")
    print()
    print("Data Augmentation: (check your get_dataloader function)")
    print()
    
    return model, total_params, trainable_params


if __name__ == "__main__":
    # Test with different architectures
    architectures = ['resnet18']  # Add more: 'resnet34', 'resnet50', 'mobilenetv2'
    
    for arch in architectures:
        try:
            model, total, trainable = inspect_model(arch=arch, bins=90)
            print(f"\n✓ {arch}: {total:,} total params, {trainable:,} trainable\n")
        except Exception as e:
            print(f"\n✗ {arch} failed: {e}\n")