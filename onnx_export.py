import os
import argparse
import torch
from config import data_config
from utils.helpers import get_model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Gaze Estimation Model ONNX Export')
    parser.add_argument(
        '-w', '--weight',
        default='output/mpiigaze_resnet18_1759321797/best_model.pt',
        type=str,
        help='Trained state_dict file path to open'
    )
    parser.add_argument(
        '-n', '--model',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50', 'mobilenetv2', 'mobileone_s0'],
        help='Backbone network architecture to use'
    )
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        default='mpiigaze',  # Changed from gaze360
        choices=list(data_config.keys()),
        help='Dataset name for bin configuration'
    )
    parser.add_argument(
        '--input-size',
        type=int,
        default=224,  # Added this parameter
        help='Input image size (224 for MPIIGaze)'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Enable dynamic batch size and input dimensions for ONNX export'
    )
    return parser.parse_args()

@torch.no_grad()
def onnx_export(params):
    print(f"\n{'='*80}")
    print("ONNX MODEL EXPORT")
    print(f"{'='*80}\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load checkpoint and detect configuration
    print(f"Loading model from: {params.weight}")
    checkpoint = torch.load(params.weight, map_location=device)
    
    # Detect bins from checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        bins = checkpoint.get('bins')
        state_dict = checkpoint['model_state_dict']
        print("✓ Found configuration in checkpoint")
    else:
        # Try to infer from model structure
        state_dict = checkpoint
        if 'fc_pitch.weight' in state_dict:
            bins = state_dict['fc_pitch.weight'].shape[0]
            print(f"✓ Detected {bins} bins from model structure")
        else:
            # Fallback to config
            bins = data_config[params.dataset]['bins']
            print(f"⚠ Using bins from config: {bins}")
    
    print(f"\nModel Configuration:")
    print(f"  Architecture: {params.model}")
    print(f"  Bins: {bins}")
    print(f"  Input size: {params.input_size}x{params.input_size}")
    
    # Initialize model
    print("\nInitializing model...")
    model = get_model(params.model, bins)
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Generate ONNX output filename
    fname = os.path.splitext(os.path.basename(params.weight))[0]
    onnx_model = f'{fname}.onnx'
    
    # Dummy input with CORRECT size
    dummy_input = torch.randn(1, 3, params.input_size, params.input_size).to(device)
    print(f"\nDummy input shape: {list(dummy_input.shape)}")
    
    # Test forward pass first
    print("Testing forward pass...")
    try:
        pitch_out, yaw_out = model(dummy_input)
        print(f"✓ Forward pass successful")
        print(f"  Pitch output: {list(pitch_out.shape)}")
        print(f"  Yaw output: {list(yaw_out.shape)}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return
    
    # Handle dynamic axes
    dynamic_axes = None
    if params.dynamic:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'pitch': {0: 'batch_size'},
            'yaw': {0: 'batch_size'}
        }
        print("\n✓ Using dynamic batch size")
    else:
        print(f"\n✓ Using fixed batch size: 1")
    
    # Export model
    print(f"\nExporting to ONNX: {onnx_model}")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_model,
            export_params=True,
            opset_version=17,  # Changed from 20 for better compatibility
            do_constant_folding=True,
            input_names=['input'],
            output_names=['pitch', 'yaw'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        print(f"✓ Model exported successfully to {onnx_model}")
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return
    
    # Verify the exported model
    print("\nVerifying ONNX model...")
    try:
        import onnx
        onnx_model_check = onnx.load(onnx_model)
        onnx.checker.check_model(onnx_model_check)
        print("✓ ONNX model is valid")
        
        # Print model info
        print(f"\nModel Info:")
        print(f"  Inputs: {[inp.name for inp in onnx_model_check.graph.input]}")
        print(f"  Outputs: {[out.name for out in onnx_model_check.graph.output]}")
        print(f"  File size: {os.path.getsize(onnx_model) / (1024*1024):.2f} MB")
        
    except ImportError:
        print("⚠ onnx package not installed, skipping verification")
        print("  Install with: pip install onnx")
    except Exception as e:
        print(f"⚠ Verification failed: {e}")
    
    print(f"\n{'='*80}")
    print("EXPORT COMPLETE")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    args = parse_arguments()
    onnx_export(args)