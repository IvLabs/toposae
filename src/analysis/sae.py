"""SAE Analysis for Superposition (H2).

Trains a Sparse Autoencoder on model activations to recover the feature basis
and measure feature superposition.

Reference: Research plan Section 3.3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder with L1 sparsity penalty.
    
    Expands N-dimensional activations to M = expansion_factor * N features,
    then reconstructs. L1 penalty on hidden activations enforces sparsity.
    """
    
    def __init__(self, input_dim: int, expansion_factor: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = input_dim * expansion_factor
        
        self.encoder = nn.Linear(input_dim, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, input_dim, bias=False)
        
        # Tie decoder weights to encoder transpose (optional, often helps stability)
        # self.decoder.weight = nn.Parameter(self.encoder.weight.t())
        
        # Initialize
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.weight)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (reconstruction, hidden_activations)."""
        hidden = F.relu(self.encoder(x))
        reconstruction = self.decoder(hidden)
        return reconstruction, hidden
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent features."""
        return F.relu(self.encoder(x))


def train_sae(
    activations: torch.Tensor,
    config: Dict,
    val_activations: torch.Tensor = None,
    device: str = 'cpu'
) -> Dict:
    """Train a Sparse Autoencoder on activations.
    
    Args:
        activations: Training activations, shape (num_samples, input_dim).
        config: Dict with SAE hyperparameters:
            - expansion_factor: int (default 8)
            - l1_penalty: float (default 0.001)
            - lr: float (default 1e-3)
            - epochs: int (default 100)
            - batch_size: int (default 256)
        val_activations: Optional held-out activations for evaluation.
        device: Device to train on.
    
    Returns:
        Dict with trained SAE model and training metrics.
    """
    expansion_factor = config.get('expansion_factor', 8)
    l1_penalty = config.get('l1_penalty', 0.001)
    lr = config.get('lr', 1e-3)
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 256)
    
    input_dim = activations.shape[1]
    sae = SparseAutoencoder(input_dim, expansion_factor).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    # Training
    losses = []
    l1_losses = []
    recon_losses = []
    
    for epoch in range(epochs):
        sae.train()
        epoch_loss = 0
        epoch_l1 = 0
        epoch_recon = 0
        n_batches = 0
        
        # Shuffle and batch
        perm = torch.randperm(activations.shape[0])
        shuffled = activations[perm]
        
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i:i+batch_size].to(device)
            
            reconstruction, hidden = sae(batch)
            
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(reconstruction, batch)
            
            # L1 sparsity penalty on hidden activations
            l1_loss = hidden.abs().mean()
            
            # Total loss
            loss = recon_loss + l1_penalty * l1_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_l1 += l1_loss.item()
            epoch_recon += recon_loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        avg_l1 = epoch_l1 / n_batches
        avg_recon = epoch_recon / n_batches
        losses.append(avg_loss)
        l1_losses.append(avg_l1)
        recon_losses.append(avg_recon)
        
        if (epoch + 1) % 20 == 0:
            print(f"  SAE Epoch {epoch+1}/{epochs}: "
                  f"loss={avg_loss:.6f}, recon={avg_recon:.6f}, "
                  f"l1={avg_l1:.6f}")
    
    # Evaluate
    metrics = evaluate_sae(sae, val_activations if val_activations is not None else activations, device)
    metrics['training_loss'] = losses
    metrics['training_l1'] = l1_losses
    metrics['training_recon'] = recon_losses
    
    return {'model': sae, 'metrics': metrics}


def evaluate_sae(
    sae: SparseAutoencoder,
    test_activations: torch.Tensor,
    device: str = 'cpu'
) -> Dict:
    """Evaluate SAE on held-out activations.
    
    Args:
        sae: Trained SparseAutoencoder.
        test_activations: Held-out activation tensor, shape (num_samples, input_dim).
        device: Device to evaluate on.
    
    Returns:
        Dict with:
            - l0_norm: Mean number of active features per sample
            - reconstruction_loss: MSE between input and reconstruction
            - dead_feature_fraction: Fraction of features never active
            - feature_usage: Histogram of feature activation frequencies
    """
    sae.eval()
    test_activations = test_activations.to(device)
    
    with torch.no_grad():
        reconstruction, hidden = sae(test_activations)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, test_activations).item()
        
        # L0 norm: count non-zero features per sample, then average
        l0_norm = (hidden > 0).float().sum(dim=1).mean().item()
        
        # Dead features: features that are never active across all samples
        feature_active = (hidden > 0).any(dim=0)  # shape: (latent_dim,)
        dead_fraction = (~feature_active).float().mean().item()
        
        # Feature usage: fraction of samples where each feature is active
        feature_usage = (hidden > 0).float().mean(dim=0)  # shape: (latent_dim,)
    
    return {
        'l0_norm': l0_norm,
        'reconstruction_loss': recon_loss,
        'dead_feature_fraction': dead_fraction,
        'feature_usage': feature_usage.cpu(),
    }


def collect_all_layers(model, dataloader, depth: int, device='cpu') -> dict:
    """Collect CLS token activations from ALL transformer blocks in one forward pass.

    Returns dict {layer_idx: Tensor(num_samples, hidden_dim)}.
    """
    from tqdm import tqdm
    layer_acts = {i: [] for i in range(depth)}
    hooks = []

    for i in range(depth):
        def make_hook(idx):
            def hook_fn(module, input, output):
                layer_acts[idx].append(output[:, 0, :].detach().cpu())
            return hook_fn
        hooks.append(model.blocks[i].register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Collecting layers", leave=False, ncols=80):
            model(images.to(device))

    for h in hooks:
        h.remove()

    return {i: torch.cat(layer_acts[i], dim=0) for i in range(depth)}


def collect_residual_stream(model, dataloader, layer_idx: int, device='cpu'):
    """Collect CLS token activations from a specific transformer layer.
    
    Extracts only the CLS token on-the-fly to keep memory low (~65 MB for 126K
    images) instead of buffering the full sequence (~4.2 GB).

    Args:
        model: TinyViT model.
        dataloader: DataLoader with images.
        layer_idx: Which block to extract from.
        device: Device to run on.

    Returns:
        Tensor of shape (num_samples, hidden_dim) — CLS token activations.
    """
    cls_activations = []
    hooks = []

    def hook_fn(module, input, output):
        # output shape: (B, N, D) — extract CLS token only
        cls_activations.append(output[:, 0, :].detach().cpu())

    hook = model.blocks[layer_idx].register_forward_hook(hook_fn)
    hooks.append(hook)

    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            _ = model(images.to(device))

    for hook in hooks:
        hook.remove()

    return torch.cat(cls_activations, dim=0)  # (num_samples, hidden_dim)
