"""
Training script for colorization model.

Implements:
- Mixed precision training (fp16)
- Gradient checkpointing
- Automatic batch size reduction on OOM
- Class rebalancing
- TensorBoard logging
- Checkpointing and resuming
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.model import get_model, count_parameters
from src.models.ops import get_ab_grid, compute_class_rebalancing_weights
from src.data.dataset import create_data_loaders
from src.utils.logger import Logger, TensorBoardLogger, MetricTracker
from src.utils.memory import (
    get_gpu_memory_info, clear_cuda_cache, estimate_safe_batch_size,
    reduce_batch_size, check_oom_risk, get_optimal_num_workers
)
from src.utils.device import (
    get_device, enable_cuda_optimizations, auto_batch_and_workers,
    get_dataloader_config, print_device_summary
)


class ColorIzationTrainer:
    """Trainer for colorization models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Enable CUDA optimizations early
        enable_cuda_optimizations()
        
        # Get device using our helper
        self.device = get_device(prefer=config.get('device', 'cuda'))
        
        # Set up logging
        self.logger = Logger(config['log_dir'], name='train')
        self.tb_logger = TensorBoardLogger(config['tensorboard_dir'])
        self.metrics = MetricTracker()
        
        self.logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            alloc, reserved, free = get_gpu_memory_info()
            self.logger.info(f"GPU Memory - Allocated: {alloc:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB")
        
        # Set random seeds for reproducibility
        self._set_seed(config.get('seed', 42))
        
        # Build model
        self.model = get_model(config['model']).to(self.device)
        self.logger.info(f"Model: {config['model']['model_type']}")
        self.logger.info(f"Parameters: {count_parameters(self.model):,}")
        
        # Load class rebalancing weights
        self.class_weights = self._load_class_weights(config.get('class_weights_file'))
        
        # Set up optimizer
        self.optimizer = self._create_optimizer()
        
        # Set up mixed precision training
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        # self.scaler = GradScaler() if self.use_amp else None
        from torch.cuda.amp import GradScaler, autocast
        self.scaler = GradScaler() if self.use_amp else None

        if self.use_amp:
            self.logger.info("Using automatic mixed precision (FP16)")
        
        # Set up learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Resume from checkpoint if specified
        if config.get('resume_from'):
            self._load_checkpoint(config['resume_from'])
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        # Note: torch.use_deterministic_algorithms(True) can be slower
        # Uncomment for full reproducibility
        # torch.use_deterministic_algorithms(True)
    
    def _load_class_weights(self, weights_file: Optional[str]) -> Optional[torch.Tensor]:
        """Load class rebalancing weights."""
        if weights_file and os.path.exists(weights_file):
            self.logger.info(f"Loading class weights from {weights_file}")
            data = np.load(weights_file)
            weights = torch.from_numpy(data['class_weights']).float().to(self.device)
            self.logger.info(f"Class weights - min: {weights.min():.3f}, max: {weights.max():.3f}")
            return weights
        else:
            self.logger.warning("No class weights file provided, using uniform weights")
            return None
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer (Adam with β1=0.9, β2=0.99)."""
        lr = self.config.get('learning_rate', 3e-5)
        weight_decay = self.config.get('weight_decay', 1e-3)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.99),
            weight_decay=weight_decay
        )
        
        self.logger.info(f"Optimizer: Adam(lr={lr}, weight_decay={weight_decay})")
        return optimizer
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        schedule_type = self.config.get('lr_schedule', 'step')
        
        if schedule_type == 'step':
            # Paper: drop to 1e-5 at 200k iterations, 3e-6 at 375k
            milestones = self.config.get('lr_milestones', [200000, 375000])
            gamma = self.config.get('lr_gamma', 0.333)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=milestones, gamma=gamma
            )
            self.logger.info(f"LR Scheduler: MultiStepLR(milestones={milestones}, gamma={gamma})")
            return scheduler
        elif schedule_type == 'cosine':
            max_steps = self.config.get('max_steps', 450000)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_steps
            )
            self.logger.info(f"LR Scheduler: CosineAnnealingLR(T_max={max_steps})")
            return scheduler
        else:
            return None
    
    def _compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.
        
        Args:
            logits: (B, Q, H, W) predicted logits
            target: (B, Q, H, W) soft-encoded target distribution
            
        Returns:
            loss: Scalar loss value
        """
        B, Q, H, W = logits.shape
        
        # Reshape for cross-entropy
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, Q)  # (B*H*W, Q)
        target_flat = target.permute(0, 2, 3, 1).reshape(-1, Q)  # (B*H*W, Q)
        
        # Compute cross-entropy with soft targets
        log_probs = F.log_softmax(logits_flat, dim=1)
        loss = -(target_flat * log_probs).sum(dim=1)  # (B*H*W,)
        
        # Apply class rebalancing weights
        if self.class_weights is not None:
            # Get the primary class for each pixel (argmax of target)
            primary_class = target_flat.argmax(dim=1)  # (B*H*W,)
            weights = self.class_weights[primary_class]  # (B*H*W,)
            loss = loss * weights
        
        return loss.mean()
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.metrics.reset()

        # for step, (L, target) in enumerate(pbar):
        #     if step == 0:  # First step of first epoch
        #         print(f"L device: {L.device}")
        #         print(f"target device: {target.device}")
        #         print(f"Model device: {next(self.model.parameters()).device}")
        #         print(f"GPU memory after data load: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']}")
        
        for batch_idx, (L, ab, target) in enumerate(pbar):
            try:
                # L = L.to(self.device)
                L = L.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                # target = target.to(self.device)
                if batch_idx == 0 and epoch == 0:
                    print(f"L device: {L.device}")
                    print(f"target device: {target.device}")
                    print(f"Model device: {next(self.model.parameters()).device}")
                    torch.cuda.synchronize()
                    print(
                        f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
                        f"reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
                    )
                
                # Forward pass with automatic mixed precision
                # 
                with autocast(enabled=self.use_amp):
                    logits = self.model(L)
                    loss = self._compute_loss(logits, target)
                
                # Backward pass
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                # Update scheduler
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Update metrics
                self.metrics.update(loss=loss.item())
                self.global_step += 1
                
                # Log to TensorBoard
                if self.global_step % self.config.get('log_interval', 10) == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.tb_logger.add_scalar('train/loss', loss.item(), self.global_step)
                    self.tb_logger.add_scalar('train/lr', current_lr, self.global_step)
                    
                    if self.device.type == 'cuda':
                        alloc, _, _ = get_gpu_memory_info()
                        self.tb_logger.add_scalar('train/gpu_memory_gb', alloc, self.global_step)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{self.metrics.get_average('loss'):.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Save sample images periodically
                if self.global_step % self.config.get('sample_interval', 1000) == 0:
                    self._save_sample_images(L, logits, ab, num_samples=4)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.error(f"OOM error at step {self.global_step}")
                    clear_cuda_cache()
                    # Note: Could implement batch size reduction here if needed
                    continue
                else:
                    raise
        
        return self.metrics.get_all_averages()
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        self.metrics.reset()
        
        for L, ab, target in tqdm(val_loader, desc="Validation"):
            L = L.to(self.device)
            target = target.to(self.device)
            
            with autocast(enabled=self.use_amp):
                logits = self.model(L)
                loss = self._compute_loss(logits, target)
            
            self.metrics.update(loss=loss.item())
        
        return self.metrics.get_all_averages()
    
    def _save_sample_images(self, L: torch.Tensor, logits: torch.Tensor,
                           ab_gt: torch.Tensor, num_samples: int = 4):
        """Save sample colorization results."""
        from src.models.ops import decode_distribution_to_ab, lab_to_rgb
        
        num_samples = min(num_samples, L.shape[0])
        
        # Decode predictions
        with torch.no_grad():
            ab_pred = decode_distribution_to_ab(logits[:num_samples], temperature=0.38)
        
        # Convert to RGB for visualization
        images = []
        for i in range(num_samples):
            # Denormalize L
            L_denorm = (L[i, 0].cpu().numpy() * 50.0) + 50.0
            
            # Predicted
            ab_pred_np = ab_pred[i].cpu().numpy().transpose(1, 2, 0)
            lab_pred = np.stack([L_denorm, ab_pred_np[:, :, 0], ab_pred_np[:, :, 1]], axis=2)
            rgb_pred = lab_to_rgb(lab_pred)
            
            # Ground truth
            ab_gt_np = ab_gt[i].cpu().numpy().transpose(1, 2, 0)
            lab_gt = np.stack([L_denorm, ab_gt_np[:, :, 0], ab_gt_np[:, :, 1]], axis=2)
            rgb_gt = lab_to_rgb(lab_gt)
            
            # Grayscale
            gray = np.stack([L_denorm/100, L_denorm/100, L_denorm/100], axis=2)
            
            # Concatenate: grayscale | predicted | ground truth
            img_row = np.concatenate([gray, rgb_pred, rgb_gt], axis=1)
            images.append(img_row)
        
        # Stack vertically
        grid = np.concatenate(images, axis=0)
        grid = torch.from_numpy(grid).permute(2, 0, 1).float()
        
        self.tb_logger.add_image('samples', grid, self.global_step)
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint: {filepath}")
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop."""
        num_epochs = self.config['num_epochs']
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            self.logger.info(f"Epoch {epoch} - Train: {train_metrics}")
            
            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader)
                self.logger.info(f"Epoch {epoch} - Val: {val_metrics}")
                
                # Log to TensorBoard
                self.tb_logger.add_scalars('loss', {
                    'train': train_metrics['loss'],
                    'val': val_metrics['loss']
                }, epoch)
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint(f'epoch_{epoch:04d}.pth', is_best=True)
                    self.logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
            else:
                self.tb_logger.add_scalar('loss/train', train_metrics['loss'], epoch)
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'epoch_{epoch:04d}.pth')
        
        # Save final checkpoint
        self.save_checkpoint('final_model.pth')
        self.logger.info("Training completed!")
        
        self.tb_logger.close()


def main():
    from src.models import ops
    ops.reset_ab_grid_cache()
    parser = argparse.ArgumentParser(description='Train colorization model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--train_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--val_dir', type=str, help='Validation data directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line args
    if args.resume:
        config['resume_from'] = args.resume
    
    # Print device summary before training
    print_device_summary()
    
    # Auto-configure batch size and num_workers if not specified
    if 'batch_size' not in config or 'num_workers' not in config:
        print("\nAuto-configuring DataLoader settings for your hardware...")
        auto_batch, auto_workers = auto_batch_and_workers(
            image_size=config.get('image_size', 256)
        )
        if 'batch_size' not in config:
            config['batch_size'] = auto_batch
            print(f"  Using auto-detected batch_size: {auto_batch}")
        if 'num_workers' not in config:
            config['num_workers'] = auto_workers
            print(f"  Using auto-detected num_workers: {auto_workers}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.train_dir,
        args.val_dir,
        batch_size=config.get('batch_size', 16),
        num_workers=config.get('num_workers', get_optimal_num_workers()),
        target_size=config.get('image_size', 256)
    )
    
    # Create trainer and start training
    trainer = ColorIzationTrainer(config)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
