"""Model definition: a standard MONAI Attention U-Net (2D)."""
import torch
from monai.networks.nets import AttentionUnet

IMG_SIZE = 256  # images are resized to 256x256 (must be divisible by 16)


def build_model() -> torch.nn.Module:
    return AttentionUnet(
        spatial_dims=2,
        in_channels=3,           # RGB
        out_channels=1,          # lesion vs background (binary)
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        dropout=0.1,
    )


def load_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    model = build_model()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()