"""
Training script for modern colorization with transformer encoder.

Implements staged training with encoder freezing schedule and mixed precision.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from src.models.modern_colorizer import ModernColorizer
from src.models.loss import CombinedColorizationLoss
from src.data.dataset import ColorizationDataset
from src.models.ops import ABColorQuantizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModernColorizationTrainer:
    """Trainer for modern colorization model."""
    
    def __init__(
        self,
        model: ModernColorizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        log_dir: str,
        checkpoint_dir: str,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Loss function
        class_weights = self._load_class_weights(config.get('class_weights_path'))
        self.criterion = CombinedColorizationLoss(
            mode=config['mode'],
            use_perceptual=config.get('use_perceptual', True),
            perceptual_weight=config.get('perceptual_weight', 0.1),
            class_weights=class_weights,
        ).to(device)
        
        # Optimizer - separate learning rates for encoder and decoder
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())
        
        self.optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': config.get('encoder_lr', 1e-5)},
            {'params': decoder_params, 'lr': config.get('decoder_lr', 1e-4)},
        ], betas=(0.9, 0.999), weight_decay=config.get('weight_decay', 1e-4))
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config.get('min_lr', 1e-7)
        )
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Quantizer for classification mode
        if config['mode'] == 'classification':
            self.quantizer = ABColorQuantizer()
        else:
            self.quantizer = None
        
        logger.info("Trainer initialized:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Mode: {config['mode']}")
        logger.info(f"  Mixed precision: {self.use_amp}")
        logger.info(f"  Encoder LR: {config.get('encoder_lr', 1e-5)}")
        logger.info(f"  Decoder LR: {config.get('decoder_lr', 1e-4)}")
    
    def _load_class_weights(self, path: Optional[str]) -> Optional[torch.Tensor]:
        """Load class weights from color statistics."""
        if path is None or not Path(path).exists():
            logger.warning("No class weights provided, using uniform weights")
            return None
        
        try:
            data = np.load(path, allow_pickle=True)
            class_weights = torch.from_numpy(data['class_weights']).float()
            logger.info(f"Loaded class weights from {path}")
            return class_weights
        except Exception as e:
            logger.error(f"Failed to load class weights: {e}")
            return None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_main_loss = 0.0
        total_perceptual_loss = 0.0
        num_batches = 0
        
        # Apply freeze schedule
        freeze_until_epoch = self.config.get('freeze_encoder_until', 10)
        if epoch < freeze_until_epoch:
            if epoch == 0:
                self.model.freeze_encoder()
                logger.info(f"Encoder frozen for epochs 0-{freeze_until_epoch-1}")
        elif epoch == freeze_until_epoch:
            self.model.unfreeze_encoder()
            logger.info(f"Encoder unfrozen starting epoch {epoch}")
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            L = batch['L'].to(self.device)
            ab_target = batch['ab'].to(self.device)
            
            # Get target bins for classification
            if self.config['mode'] == 'classification':
                bins_target = batch['bins'].to(self.device)
            else:
                bins_target = None
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(L, return_logits=True)
                
                if self.config['mode'] == 'classification':
                    pred = outputs['logits']
                else:
                    pred = outputs['ab']
                
                # Compute loss
                losses = self.criterion(pred, ab_target, L, bins_target)
                loss = losses['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Track metrics
            total_loss += loss.item()
            
            main_loss_key = 'classification_loss' if self.config['mode'] == 'classification' else 'regression_loss'
            if main_loss_key in losses:
                total_main_loss += losses[main_loss_key].item()
            
            if 'perceptual_loss' in losses:
                total_perceptual_loss += losses['perceptual_loss'].item()
            
            num_batches += 1
            self.global_step += 1
            
            # Log to tensorboard
            if self.global_step % self.config.get('log_interval', 10) == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                if main_loss_key in losses:
                    self.writer.add_scalar(f'train/{main_loss_key}', losses[main_loss_key].item(), self.global_step)
                if 'perceptual_loss' in losses:
                    self.writer.add_scalar('train/perceptual_loss', losses['perceptual_loss'].item(), self.global_step)
            
            # Progress logging
            if batch_idx % self.config.get('log_interval', 10) == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )
        
        epoch_time = time.time() - start_time
        
        metrics = {
            'loss': total_loss / num_batches,
            'main_loss': total_main_loss / num_batches,
            'perceptual_loss': total_perceptual_loss / num_batches if total_perceptual_loss > 0 else 0.0,
            'epoch_time': epoch_time,
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_main_loss = 0.0
        total_perceptual_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            L = batch['L'].to(self.device)
            ab_target = batch['ab'].to(self.device)
            
            if self.config['mode'] == 'classification':
                bins_target = batch['bins'].to(self.device)
            else:
                bins_target = None
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(L, return_logits=True)
                
                if self.config['mode'] == 'classification':
                    pred = outputs['logits']
                else:
                    pred = outputs['ab']
                
                losses = self.criterion(pred, ab_target, L, bins_target)
                loss = losses['loss']
            
            total_loss += loss.item()
            
            main_loss_key = 'classification_loss' if self.config['mode'] == 'classification' else 'regression_loss'
            if main_loss_key in losses:
                total_main_loss += losses[main_loss_key].item()
            
            if 'perceptual_loss' in losses:
                total_perceptual_loss += losses['perceptual_loss'].item()
            
            num_batches += 1
        
        metrics = {
            'loss': total_loss / num_batches,
            'main_loss': total_main_loss / num_batches,
            'perceptual_loss': total_perceptual_loss / num_batches if total_perceptual_loss > 0 else 0.0,
        }
        
        # Log to tensorboard
        self.writer.add_scalar('val/loss', metrics['loss'], epoch)
        self.writer.add_scalar('val/main_loss', metrics['main_loss'], epoch)
        if metrics['perceptual_loss'] > 0:
            self.writer.add_scalar('val/perceptual_loss', metrics['perceptual_loss'], epoch)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'metrics': metrics,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
        
        # Keep only last N checkpoints
        keep_last = self.config.get('keep_last_checkpoints', 3)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > keep_last:
            for old_checkpoint in checkpoints[:-keep_last]:
                old_checkpoint.unlink()
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.config['num_epochs']}")
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            logger.info(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            logger.info(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Main: {train_metrics['main_loss']:.4f}, "
                f"Perceptual: {train_metrics['perceptual_loss']:.4f}, "
                f"Time: {train_metrics['epoch_time']:.2f}s"
            )
            
            # Validate
            val_metrics = self.validate(epoch)
            logger.info(
                f"Val - Loss: {val_metrics['loss']:.4f}, "
                f"Main: {val_metrics['main_loss']:.4f}, "
                f"Perceptual: {val_metrics['perceptual_loss']:.4f}"
            )
            
            # Learning rate schedule
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/learning_rate', current_lr, epoch)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            if (epoch + 1) % self.config.get('save_interval', 5) == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            self.current_epoch = epoch + 1
        
        logger.info("\n" + "="*60)
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("="*60)
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train modern colorization model")
    
    # Data
    parser.add_argument('--train_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Validation data directory')
    parser.add_argument('--color_stats', type=str, default='data/color_stats.npz', help='Color statistics file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    
    # Model
    parser.add_argument('--encoder_size', type=str, default='base', choices=['tiny', 'base', 'swin'], help='Encoder size')
    parser.add_argument('--encoder_model', type=str, default=None, help='Specific HF model name')
    parser.add_argument('--norm_mode', type=str, default='spade', choices=['spade', 'adain'], help='Normalization mode')
    parser.add_argument('--block_type', type=str, default='convnext', choices=['convnext', 'residual'], help='Decoder block type')
    parser.add_argument('--mode', type=str, default='classification', choices=['classification', 'regression'], help='Output mode')
    parser.add_argument('--use_checkpointing', action='store_true', help='Enable gradient checkpointing')
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--encoder_lr', type=float, default=1e-5, help='Encoder learning rate')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='Decoder learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--freeze_encoder_until', type=int, default=10, help='Freeze encoder until this epoch')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use mixed precision')
    
    # Loss
    parser.add_argument('--use_perceptual', action='store_true', default=True, help='Use perceptual loss')
    parser.add_argument('--perceptual_weight', type=float, default=0.1, help='Perceptual loss weight')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='logs/modern', help='Log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/modern', help='Checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=5, help='Checkpoint save interval')
    parser.add_argument('--keep_last_checkpoints', type=int, default=3, help='Keep last N checkpoints')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = ColorizationDataset(
        root_dir=args.train_dir,
        color_stats_path=args.color_stats if args.mode == 'classification' else None,
        mode='train',
    )
    
    val_dataset = ColorizationDataset(
        root_dir=args.val_dir,
        color_stats_path=args.color_stats if args.mode == 'classification' else None,
        mode='val',
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = ModernColorizer(
        encoder_model=args.encoder_model,
        encoder_size=args.encoder_size,
        encoder_pretrained=True,
        encoder_freeze_blocks=0,
        norm_mode=args.norm_mode,
        block_type=args.block_type,
        mode=args.mode,
        num_classes=313,
        use_checkpointing=args.use_checkpointing,
    )
    
    # Print parameter counts
    param_counts = model.count_parameters()
    logger.info(f"Total parameters: {param_counts['total']:,}")
    logger.info(f"Encoder parameters: {param_counts['encoder']:,}")
    logger.info(f"Decoder parameters: {param_counts['decoder']:,}")
    
    # Training config
    config = vars(args)
    config['class_weights_path'] = args.color_stats if args.mode == 'classification' else None
    
    # Save config
    config_path = Path(args.log_dir) / 'config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create trainer and train
    trainer = ModernColorizationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
