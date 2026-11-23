"""Memory-efficient encoding of augmented image views."""
import torch
from torch.utils.checkpoint import checkpoint


class CheckpointedViewEncoder(torch.nn.Module):
    """Checkpoint each view using the full image batch."""
    def __init__(self, encoder, image_batch_size):
        super().__init__()
        if image_batch_size < 1:
            raise ValueError('Image batch size must be positive')
        if any(p.requires_grad for p in encoder.parameters()):
            raise ValueError('View checkpointing requires a frozen source encoder')
        self.encoder = encoder
        self.image_batch_size = image_batch_size

    def forward(self, images):
        if (len(images) <= self.image_batch_size or not torch.is_grad_enabled()
                or not images.requires_grad):
            return self.encoder(images)
        if len(images) % self.image_batch_size:
            raise ValueError('Augmentation views must contain complete global image batches')
        outputs = [checkpoint(self.encoder, view, use_reentrant=False, preserve_rng_state=True)
                   for view in images.split(self.image_batch_size)]
        return {key: torch.cat([out[key] for out in outputs], dim=0) for key in outputs[0]}
