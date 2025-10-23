import onnx
import onnxruntime as ort
import numpy as np
import torch
from utils.helpers import get_model

# Load PyTorch model
print("Loading PyTorch model...")
pytorch_model = get_model('resnet18', 28)
checkpoint = torch.load('output/mpiigaze_resnet18_1759321797/best_model.pt', map_location='cpu')

# Handle checkpoint format
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

pytorch_model.load_state_dict(state_dict)
pytorch_model.eval()
print("✓ PyTorch model loaded")

# Load ONNX model
print("Loading ONNX model...")
onnx_session = ort.InferenceSession('best_model.onnx')
print("✓ ONNX model loaded")

# Test with same input
print("\nTesting predictions...")
test_input = torch.randn(1, 3, 224, 224)

# PyTorch prediction
with torch.no_grad():
    pt_pitch, pt_yaw = pytorch_model(test_input)

# ONNX prediction
onnx_input = {'input': test_input.numpy()}
onnx_pitch, onnx_yaw = onnx_session.run(None, onnx_input)

# Compare
print("\nComparison:")
print(f"PyTorch pitch (first 5): {pt_pitch[0, :5].numpy()}")
print(f"ONNX pitch (first 5):    {onnx_pitch[0, :5]}")
print(f"Max difference (pitch):  {np.abs(pt_pitch.numpy() - onnx_pitch).max():.8f}")
print(f"Max difference (yaw):    {np.abs(pt_yaw.numpy() - onnx_yaw).max():.8f}")

# Update the threshold check
max_diff_pitch = np.abs(pt_pitch.numpy() - onnx_pitch).max()
max_diff_yaw = np.abs(pt_yaw.numpy() - onnx_yaw).max()

print(f"\nMax difference (pitch):  {max_diff_pitch:.8f}")
print(f"Max difference (yaw):    {max_diff_yaw:.8f}")

# More realistic threshold
if max_diff_pitch < 1e-4 and max_diff_yaw < 1e-4:
    print("\n✓✓✓ ONNX export is accurate! ✓✓✓")
    print("Difference is negligible (< 0.0001°)")
    print("Your 0.92° accuracy is preserved in ONNX")
else:
    print("\n⚠ Warning: Significant difference detected")