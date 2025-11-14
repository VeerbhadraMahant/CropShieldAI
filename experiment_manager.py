"""
Experiment Management System for CropShield AI

Simple grid search for quick hyperparameter sweeps on laptop.
Supports local grid search with automatic experiment tracking.

Features:
- Grid search over learning rate, weight decay, dropout
- Automatic experiment tracking (config + results)
- Quick 3-epoch sweeps for rapid iteration
- JSON logging for easy analysis
- Support for resuming experiments
"""

import os
import json
import time
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
from tqdm import tqdm

from models.model_factory import get_model
from fast_dataset import make_loaders
from train import Trainer


class ExperimentManager:
    """
    Manages hyperparameter sweep experiments.
    
    Features:
    - Automatic experiment ID generation
    - JSON config and results saving
    - Experiment resume support
    - Best model tracking
    """
    
    def __init__(self, base_dir: str = 'experiments'):
        """
        Initialize experiment manager.
        
        Args:
            base_dir: Base directory for all experiments
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiments directory structure
        self.runs_dir = self.base_dir / 'runs'
        self.runs_dir.mkdir(exist_ok=True)
        
        self.best_dir = self.base_dir / 'best'
        self.best_dir.mkdir(exist_ok=True)
        
        # Load or create sweep registry
        self.registry_path = self.base_dir / 'sweep_registry.json'
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load sweep registry or create new one."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {'sweeps': []}
    
    def _save_registry(self):
        """Save sweep registry."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, indent=2, fp=f)
    
    def create_experiment_id(self, sweep_name: str = None) -> str:
        """
        Generate unique experiment ID.
        
        Args:
            sweep_name: Optional name for the sweep
            
        Returns:
            Unique experiment ID (e.g., 'sweep_20251110_143022')
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if sweep_name:
            return f"{sweep_name}_{timestamp}"
        return f"sweep_{timestamp}"
    
    def save_config(self, exp_id: str, config: Dict):
        """
        Save experiment configuration.
        
        Args:
            exp_id: Experiment ID
            config: Configuration dictionary
        """
        exp_dir = self.runs_dir / exp_id
        exp_dir.mkdir(exist_ok=True)
        
        config_path = exp_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, indent=2, fp=f)
    
    def save_results(self, exp_id: str, results: Dict):
        """
        Save experiment results.
        
        Args:
            exp_id: Experiment ID
            results: Results dictionary
        """
        exp_dir = self.runs_dir / exp_id
        results_path = exp_dir / 'summary.json'
        
        with open(results_path, 'w') as f:
            json.dump(results, indent=2, fp=f)
    
    def load_results(self, exp_id: str) -> Optional[Dict]:
        """
        Load experiment results.
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            Results dictionary or None if not found
        """
        results_path = self.runs_dir / exp_id / 'summary.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
        return None
    
    def register_sweep(self, sweep_id: str, config: Dict):
        """
        Register a new sweep in the registry.
        
        Args:
            sweep_id: Sweep ID
            config: Sweep configuration
        """
        self.registry['sweeps'].append({
            'sweep_id': sweep_id,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'status': 'running'
        })
        self._save_registry()
    
    def update_sweep_status(self, sweep_id: str, status: str, best_exp: str = None):
        """
        Update sweep status.
        
        Args:
            sweep_id: Sweep ID
            status: New status ('running', 'completed', 'failed')
            best_exp: Best experiment ID (optional)
        """
        for sweep in self.registry['sweeps']:
            if sweep['sweep_id'] == sweep_id:
                sweep['status'] = status
                if best_exp:
                    sweep['best_experiment'] = best_exp
                break
        self._save_registry()
    
    def get_best_experiment(self, metric: str = 'val_acc', mode: str = 'max') -> Tuple[str, float]:
        """
        Find best experiment across all runs.
        
        Args:
            metric: Metric to optimize ('val_acc', 'val_loss', etc.)
            mode: 'max' or 'min'
            
        Returns:
            (best_exp_id, best_metric_value)
        """
        best_exp = None
        best_value = float('-inf') if mode == 'max' else float('inf')
        
        for exp_dir in self.runs_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            results = self.load_results(exp_dir.name)
            if results and metric in results:
                value = results[metric]
                
                if mode == 'max' and value > best_value:
                    best_value = value
                    best_exp = exp_dir.name
                elif mode == 'min' and value < best_value:
                    best_value = value
                    best_exp = exp_dir.name
        
        return best_exp, best_value
    
    def summarize_sweep(self, sweep_id: str = None) -> Dict:
        """
        Generate summary of sweep results.
        
        Args:
            sweep_id: Optional sweep ID to filter
            
        Returns:
            Summary dictionary
        """
        experiments = []
        
        for exp_dir in self.runs_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            if sweep_id and not exp_dir.name.startswith(sweep_id):
                continue
            
            config_path = exp_dir / 'config.json'
            results_path = exp_dir / 'summary.json'
            
            if config_path.exists() and results_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                experiments.append({
                    'exp_id': exp_dir.name,
                    'config': config,
                    'results': results
                })
        
        # Sort by validation accuracy (descending)
        experiments.sort(key=lambda x: x['results'].get('val_acc', 0), reverse=True)
        
        return {
            'total_experiments': len(experiments),
            'experiments': experiments
        }


def grid_search(
    param_grid: Dict[str, List],
    model_type: str = 'custom',
    num_epochs: int = 3,
    batch_size: int = 32,
    num_workers: int = 8,
    augmentation_mode: str = 'moderate',
    sweep_name: str = None,
    device: str = None,
    save_best_only: bool = True
) -> Dict:
    """
    Run grid search over hyperparameter combinations.
    
    Args:
        param_grid: Dictionary of parameter lists to search
            Example: {
                'learning_rate': [1e-4, 3e-4, 1e-5],
                'weight_decay': [0, 1e-5, 1e-4],
                'dropout': [0.2, 0.4]  # Not used yet
            }
        model_type: Model architecture ('custom', 'efficientnet_b0')
        num_epochs: Number of epochs per experiment (default: 3 for quick sweeps)
        batch_size: Batch size
        num_workers: DataLoader workers
        augmentation_mode: Augmentation mode
        sweep_name: Optional name for sweep
        device: Device to use (None for auto-detect)
        save_best_only: Only save checkpoint for best model
        
    Returns:
        Summary dictionary with all results
    """
    
    # Initialize experiment manager
    exp_manager = ExperimentManager()
    
    # Create sweep ID
    sweep_id = exp_manager.create_experiment_id(sweep_name or 'grid_search')
    
    print(f"\n{'='*80}")
    print(f"🔬 GRID SEARCH EXPERIMENT")
    print(f"{'='*80}")
    print(f"Sweep ID: {sweep_id}")
    print(f"Model: {model_type}")
    print(f"Epochs per run: {num_epochs}")
    print(f"{'='*80}\n")
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"📊 Parameter Grid:")
    for name, values in param_grid.items():
        print(f"   {name}: {values}")
    print(f"\n🔢 Total combinations: {len(combinations)}")
    print(f"{'='*80}\n")
    
    # Register sweep
    exp_manager.register_sweep(sweep_id, {
        'param_grid': param_grid,
        'model_type': model_type,
        'num_epochs': num_epochs,
        'batch_size': batch_size
    })
    
    # Load data once (reuse for all experiments)
    print("📦 Loading dataset...")
    train_loader, val_loader, test_loader, class_names, info = make_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation_mode=augmentation_mode
    )
    print(f"✅ Dataset loaded: {info['total_images']:,} images\n")
    
    # Device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Run experiments
    results = []
    best_val_acc = 0.0
    best_exp_id = None
    
    for idx, combination in enumerate(combinations, 1):
        # Create config for this run
        config = dict(zip(param_names, combination))
        config['model_type'] = model_type
        config['num_epochs'] = num_epochs
        config['batch_size'] = batch_size
        config['augmentation_mode'] = augmentation_mode
        
        # Create experiment ID
        exp_id = f"{sweep_id}_run{idx:03d}"
        
        print(f"\n{'='*80}")
        print(f"🧪 Experiment {idx}/{len(combinations)}")
        print(f"{'='*80}")
        print(f"Exp ID: {exp_id}")
        print(f"Config: {config}")
        print(f"{'='*80}\n")
        
        # Save config
        exp_manager.save_config(exp_id, config)
        
        try:
            # Create model with current hyperparameters
            model, optimizer, criterion, scheduler, device = get_model(
                model_type=model_type,
                num_classes=len(class_names),
                learning_rate=config.get('learning_rate', 1e-4),
                weight_decay=config.get('weight_decay', 1e-4),
                scheduler_type=config.get('scheduler', 'step'),
                verbose=False
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_classes=len(class_names),
                accumulation_steps=config.get('accumulation_steps', 1),
                use_amp=config.get('use_amp', True),
                clip_grad_norm=config.get('clip_grad_norm', None),
                checkpoint_dir=str(exp_manager.runs_dir / exp_id / 'checkpoints'),
                log_dir=str(exp_manager.runs_dir / exp_id),
                model_name=exp_id
            )
            
            # Train for quick sweep
            start_time = time.time()
            history = trainer.train(
                max_epochs=num_epochs,
                save_every=num_epochs + 1,  # Don't save during training
                early_stopping_patience=None  # No early stopping for quick sweep
            )
            train_time = time.time() - start_time
            
            # Get final metrics
            final_train_loss = history['train_loss'][-1]
            final_train_acc = history['train_acc'][-1]
            final_val_loss = history['val_loss'][-1]
            final_val_acc = history['val_acc'][-1]
            
            # Save results
            exp_results = {
                'exp_id': exp_id,
                'config': config,
                'train_loss': final_train_loss,
                'train_acc': final_train_acc,
                'val_loss': final_val_loss,
                'val_acc': final_val_acc,
                'train_time_seconds': train_time,
                'history': history,
                'status': 'completed'
            }
            
            exp_manager.save_results(exp_id, exp_results)
            results.append(exp_results)
            
            # Track best
            if final_val_acc > best_val_acc:
                best_val_acc = final_val_acc
                best_exp_id = exp_id
                
                # Save best model
                if save_best_only:
                    best_model_path = exp_manager.best_dir / f"{sweep_id}_best.pth"
                    torch.save({
                        'exp_id': exp_id,
                        'config': config,
                        'model_state_dict': model.state_dict(),
                        'val_acc': final_val_acc,
                        'val_loss': final_val_loss
                    }, best_model_path)
                    print(f"\n🏆 New best model saved! Val Acc: {final_val_acc:.2f}%")
            
            print(f"\n✅ Experiment complete!")
            print(f"   Train Loss: {final_train_loss:.4f}, Train Acc: {final_train_acc:.2f}%")
            print(f"   Val Loss:   {final_val_loss:.4f}, Val Acc:   {final_val_acc:.2f}%")
            print(f"   Time: {train_time:.1f}s")
            
        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            exp_results = {
                'exp_id': exp_id,
                'config': config,
                'status': 'failed',
                'error': str(e)
            }
            exp_manager.save_results(exp_id, exp_results)
            results.append(exp_results)
    
    # Update sweep status
    exp_manager.update_sweep_status(sweep_id, 'completed', best_exp_id)
    
    # Generate summary
    print(f"\n{'='*80}")
    print(f"📊 SWEEP SUMMARY")
    print(f"{'='*80}")
    
    # Sort by validation accuracy
    completed_results = [r for r in results if r.get('status') == 'completed']
    completed_results.sort(key=lambda x: x.get('val_acc', 0), reverse=True)
    
    print(f"\n🏆 Top 5 Configurations:")
    for i, result in enumerate(completed_results[:5], 1):
        print(f"\n{i}. {result['exp_id']}")
        print(f"   Val Acc: {result['val_acc']:.2f}%")
        print(f"   Config: lr={result['config']['learning_rate']}, "
              f"wd={result['config']['weight_decay']}")
    
    if best_exp_id:
        print(f"\n🎯 Best experiment: {best_exp_id}")
        print(f"   Val Acc: {best_val_acc:.2f}%")
    
    print(f"\n📁 Results saved to: {exp_manager.runs_dir / sweep_id}")
    print(f"{'='*80}\n")
    
    # Create final summary
    summary = {
        'sweep_id': sweep_id,
        'param_grid': param_grid,
        'total_experiments': len(combinations),
        'completed': len(completed_results),
        'failed': len(results) - len(completed_results),
        'best_exp_id': best_exp_id,
        'best_val_acc': best_val_acc,
        'results': results
    }
    
    # Save sweep summary
    summary_path = exp_manager.runs_dir / f"{sweep_id}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, indent=2, fp=f)
    
    return summary


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_quick_sweep():
    """
    Example: Quick hyperparameter sweep (3 epochs per run).
    
    Typical laptop timing:
    - 3 epochs × 18 combinations ≈ 15-20 minutes
    """
    
    param_grid = {
        'learning_rate': [1e-4, 3e-4, 1e-5],
        'weight_decay': [0, 1e-5, 1e-4],
        'scheduler': ['step', 'plateau']
    }
    
    summary = grid_search(
        param_grid=param_grid,
        model_type='custom',
        num_epochs=3,  # Quick sweep: 3 epochs only
        batch_size=32,
        sweep_name='lr_wd_sweep',
        save_best_only=True
    )
    
    return summary


def example_transfer_learning_sweep():
    """
    Example: Transfer learning hyperparameter sweep.
    """
    
    param_grid = {
        'learning_rate': [1e-4, 5e-5, 1e-5],
        'weight_decay': [1e-4, 5e-5],
        'warmup_epochs': [0, 1]
    }
    
    summary = grid_search(
        param_grid=param_grid,
        model_type='efficientnet_b0',
        num_epochs=3,
        batch_size=16,
        sweep_name='efficientnet_sweep',
        save_best_only=True
    )
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter grid search')
    parser.add_argument('--model', type=str, default='custom',
                       choices=['custom', 'efficientnet_b0'],
                       help='Model type')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Epochs per experiment (default: 3 for quick sweep)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--sweep_name', type=str, default=None,
                       help='Sweep name')
    
    args = parser.parse_args()
    
    # Default param grid
    param_grid = {
        'learning_rate': [1e-4, 3e-4, 1e-5],
        'weight_decay': [0, 1e-5, 1e-4]
    }
    
    print(f"\n🚀 Starting grid search...")
    print(f"   Model: {args.model}")
    print(f"   Epochs per run: {args.epochs}")
    print(f"   Param grid: {param_grid}\n")
    
    summary = grid_search(
        param_grid=param_grid,
        model_type=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        sweep_name=args.sweep_name
    )
    
    print(f"\n✅ Grid search complete!")
    print(f"   Best Val Acc: {summary['best_val_acc']:.2f}%")
    print(f"   Best Exp: {summary['best_exp_id']}")
