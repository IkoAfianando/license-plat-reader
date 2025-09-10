"""
Custom Model Training Script for License Plate Reader
Train YOLOv8 models specifically for license plate detection
"""

import os
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

try:
    import torch
    from ultralytics import YOLO
    import matplotlib.pyplot as plt
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("❌ Required dependencies not available. Install with:")
    print("   pip install torch ultralytics matplotlib")

from models.model_manager import ModelManager
from data.data_manager import DataManager

logger = logging.getLogger(__name__)

class LPRModelTrainer:
    """Custom model trainer for license plate detection"""
    
    def __init__(self, 
                 base_models_dir: str = "models",
                 data_dir: str = "data"):
        """Initialize trainer with model and data directories"""
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError("Required dependencies not available")
        
        self.model_manager = ModelManager(base_models_dir)
        self.data_manager = DataManager(data_dir)
        
        # Training configurations
        self.training_configs = {
            'nano': {
                'base_model': 'yolov8n.pt',
                'epochs': 100,
                'batch_size': 32,
                'image_size': 640,
                'description': 'Nano model - fastest inference'
            },
            'small': {
                'base_model': 'yolov8s.pt', 
                'epochs': 150,
                'batch_size': 16,
                'image_size': 640,
                'description': 'Small model - balanced speed/accuracy'
            },
            'medium': {
                'base_model': 'yolov8m.pt',
                'epochs': 200,
                'batch_size': 8,
                'image_size': 640,
                'description': 'Medium model - high accuracy'
            },
            'large': {
                'base_model': 'yolov8l.pt',
                'epochs': 250,
                'batch_size': 4,
                'image_size': 640,
                'description': 'Large model - highest accuracy'
            }
        }
    
    def prepare_dataset(self, 
                       dataset_name: str,
                       validation_split: float = 0.2) -> str:
        """
        Prepare dataset for training
        
        Args:
            dataset_name: Name of the dataset
            validation_split: Portion of data for validation
            
        Returns:
            Path to dataset.yaml file
        """
        print(f"📂 Preparing dataset: {dataset_name}")
        
        # Validate dataset exists
        validation = self.data_manager.validate_dataset(dataset_name)
        if not validation['valid']:
            raise ValueError(f"Dataset '{dataset_name}' is not valid: {validation}")
        
        print(f"✅ Dataset validation passed")
        print(f"   Images: {validation['images']['valid']}/{validation['images']['total']}")
        print(f"   Annotations: {validation['annotations']['valid']}/{validation['annotations']['total']}")
        
        # Export dataset in YOLO format with train/val split
        train_split = 1.0 - validation_split
        export_path = self.data_manager.export_dataset(
            dataset_name=dataset_name,
            export_format="yolo",
            train_split=train_split,
            val_split=validation_split,
            test_split=0.0
        )
        
        dataset_yaml_path = os.path.join(export_path, "dataset.yaml")
        
        if not os.path.exists(dataset_yaml_path):
            raise FileNotFoundError(f"Dataset YAML not created: {dataset_yaml_path}")
        
        print(f"✅ Dataset prepared: {export_path}")
        return dataset_yaml_path
    
    def train_model(self, 
                   dataset_yaml: str,
                   model_name: str,
                   config_name: str = 'small',
                   custom_config: Optional[Dict] = None,
                   resume: bool = False) -> Dict[str, Any]:
        """
        Train custom license plate detection model
        
        Args:
            dataset_yaml: Path to dataset configuration
            model_name: Name for the trained model
            config_name: Predefined config name (nano, small, medium, large)
            custom_config: Custom training configuration
            resume: Resume training from checkpoint
            
        Returns:
            Training results dictionary
        """
        if not os.path.exists(dataset_yaml):
            raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
        
        # Get training configuration
        if custom_config:
            config = custom_config
        elif config_name in self.training_configs:
            config = self.training_configs[config_name].copy()
        else:
            available = ', '.join(self.training_configs.keys())
            raise ValueError(f"Config '{config_name}' not available. Available: {available}")
        
        print(f"🚀 Starting training: {model_name}")
        print(f"📋 Configuration: {config_name}")
        print(f"   Base model: {config['base_model']}")
        print(f"   Epochs: {config['epochs']}")
        print(f"   Batch size: {config['batch_size']}")
        print(f"   Image size: {config['image_size']}")
        
        # Ensure base model is available
        base_model_path = self.model_manager.pretrained_dir / config['base_model']
        if not base_model_path.exists():
            print(f"📥 Downloading base model: {config['base_model']}")
            self.model_manager.download_model(config['base_model'])
        
        # Set up training directories
        training_dir = self.model_manager.custom_dir / f"training_{model_name}"
        training_dir.mkdir(exist_ok=True)
        
        # Save training configuration
        config_path = training_dir / "training_config.json"
        training_metadata = {
            'model_name': model_name,
            'config_name': config_name,
            'dataset_yaml': dataset_yaml,
            'config': config,
            'started': datetime.now().isoformat(),
            'resume': resume
        }
        
        with open(config_path, 'w') as f:
            json.dump(training_metadata, f, indent=2)
        
        try:
            # Initialize model
            model = YOLO(str(base_model_path))
            
            # Set up training parameters
            training_args = {
                'data': dataset_yaml,
                'epochs': config['epochs'],
                'imgsz': config['image_size'],
                'batch': config['batch_size'],
                'name': model_name,
                'project': str(self.model_manager.custom_dir),
                'device': 'auto',  # Auto-detect GPU/CPU
                'workers': 8,
                'patience': 50,  # Early stopping
                'save': True,
                'save_period': 10,  # Save checkpoint every 10 epochs
                'cache': False,  # Don't cache images (save RAM)
                'rect': True,   # Rectangular training
                'resume': resume,
                'amp': True,    # Automatic Mixed Precision
                'fraction': 1.0,  # Use full dataset
                'profile': False,
                'freeze': None,  # Don't freeze layers
                'lr0': 0.01,    # Initial learning rate
                'lrf': 0.01,    # Final learning rate
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,     # Box loss gain
                'cls': 0.5,     # Class loss gain
                'dfl': 1.5,     # DFL loss gain
                'pose': 12.0,   # Pose loss gain
                'kobj': 1.0,    # Keypoint object loss gain
                'label_smoothing': 0.0,
                'nbs': 64,      # Nominal batch size
                'hsv_h': 0.015, # HSV hue augmentation
                'hsv_s': 0.7,   # HSV saturation augmentation  
                'hsv_v': 0.4,   # HSV value augmentation
                'degrees': 0.0, # Rotation degrees
                'translate': 0.1, # Translation
                'scale': 0.5,   # Scale
                'shear': 0.0,   # Shear
                'perspective': 0.0, # Perspective
                'flipud': 0.0,  # Vertical flip probability
                'fliplr': 0.5,  # Horizontal flip probability
                'mosaic': 1.0,  # Mosaic augmentation probability
                'mixup': 0.0,   # Mixup augmentation probability
                'copy_paste': 0.0, # Copy-paste augmentation probability
                'auto_augment': 'randaugment',
                'erasing': 0.4, # Random erasing probability
                'crop_fraction': 1.0, # Crop fraction
            }
            
            print("🔥 Training started...")
            print(f"   Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
            
            # Start training
            results = model.train(**training_args)
            
            # Get training results
            training_result_dir = self.model_manager.custom_dir / model_name
            best_model_path = training_result_dir / "weights" / "best.pt"
            last_model_path = training_result_dir / "weights" / "last.pt"
            
            # Copy best model to main models directory
            final_model_path = self.model_manager.custom_dir / f"{model_name}.pt"
            if best_model_path.exists():
                import shutil
                shutil.copy2(best_model_path, final_model_path)
                print(f"✅ Best model saved: {final_model_path}")
            
            # Extract training metrics
            results_dict = {}
            if hasattr(results, 'results_dict'):
                results_dict = results.results_dict
            
            # Update training metadata
            training_metadata.update({
                'completed': datetime.now().isoformat(),
                'final_model_path': str(final_model_path),
                'training_directory': str(training_result_dir),
                'results': results_dict,
                'status': 'completed'
            })
            
            with open(config_path, 'w') as f:
                json.dump(training_metadata, f, indent=2)
            
            # Register model with ModelManager
            if final_model_path.exists():
                model_hash = self.model_manager._calculate_file_hash(str(final_model_path))
                self.model_manager.registry['models'][model_name] = {
                    'type': 'custom',
                    'path': str(final_model_path),
                    'size': final_model_path.stat().st_size,
                    'hash': model_hash,
                    'training_metadata': training_metadata
                }
                self.model_manager.save_registry()
            
            print(f"✅ Training completed successfully!")
            print(f"   Model: {final_model_path}")
            print(f"   Training directory: {training_result_dir}")
            
            return training_metadata
            
        except KeyboardInterrupt:
            print("\\n⏹️  Training interrupted by user")
            training_metadata.update({
                'status': 'interrupted',
                'interrupted': datetime.now().isoformat()
            })
            with open(config_path, 'w') as f:
                json.dump(training_metadata, f, indent=2)
            raise
            
        except Exception as e:
            print(f"\\n❌ Training failed: {e}")
            training_metadata.update({
                'status': 'failed',
                'error': str(e),
                'failed': datetime.now().isoformat()
            })
            with open(config_path, 'w') as f:
                json.dump(training_metadata, f, indent=2)
            raise
    
    def evaluate_model(self, 
                      model_path: str,
                      test_dataset_yaml: str) -> Dict[str, Any]:
        """
        Evaluate trained model on test dataset
        
        Args:
            model_path: Path to trained model
            test_dataset_yaml: Path to test dataset configuration
            
        Returns:
            Evaluation results
        """
        print(f"📊 Evaluating model: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if not os.path.exists(test_dataset_yaml):
            raise FileNotFoundError(f"Test dataset not found: {test_dataset_yaml}")
        
        try:
            # Load model
            model = YOLO(model_path)
            
            # Run validation
            results = model.val(
                data=test_dataset_yaml,
                imgsz=640,
                batch=16,
                conf=0.001,  # Low confidence for comprehensive evaluation
                iou=0.6,
                max_det=300,
                half=False,
                device='auto',
                dnn=False,
                plots=True,
                verbose=True
            )
            
            # Extract metrics
            evaluation_results = {
                'model_path': model_path,
                'test_dataset': test_dataset_yaml,
                'evaluated': datetime.now().isoformat()
            }
            
            if hasattr(results, 'results_dict'):
                evaluation_results['metrics'] = results.results_dict
            
            # Print key metrics
            if 'metrics' in evaluation_results:
                metrics = evaluation_results['metrics']
                print(f"\\n📈 Evaluation Results:")
                
                if 'metrics/mAP50(B)' in metrics:
                    print(f"   mAP@0.5: {metrics['metrics/mAP50(B)']:.3f}")
                if 'metrics/mAP50-95(B)' in metrics:
                    print(f"   mAP@0.5-0.95: {metrics['metrics/mAP50-95(B)']:.3f}")
                if 'metrics/precision(B)' in metrics:
                    print(f"   Precision: {metrics['metrics/precision(B)']:.3f}")
                if 'metrics/recall(B)' in metrics:
                    print(f"   Recall: {metrics['metrics/recall(B)']:.3f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def compare_models(self, 
                      model_paths: list,
                      test_dataset_yaml: str) -> Dict[str, Any]:
        """
        Compare multiple models on same test dataset
        
        Args:
            model_paths: List of model paths to compare
            test_dataset_yaml: Path to test dataset
            
        Returns:
            Comparison results
        """
        print(f"⚖️  Comparing {len(model_paths)} models")
        
        comparison_results = {
            'test_dataset': test_dataset_yaml,
            'models': {},
            'comparison': datetime.now().isoformat()
        }
        
        for model_path in model_paths:
            if not os.path.exists(model_path):
                print(f"⚠️  Skipping missing model: {model_path}")
                continue
            
            model_name = Path(model_path).stem
            print(f"\\n📊 Evaluating {model_name}...")
            
            try:
                results = self.evaluate_model(model_path, test_dataset_yaml)
                comparison_results['models'][model_name] = results
                
            except Exception as e:
                print(f"❌ Failed to evaluate {model_name}: {e}")
                comparison_results['models'][model_name] = {'error': str(e)}
        
        # Create comparison summary
        summary = {'best_model': None, 'metrics_comparison': {}}
        
        best_map = 0
        for model_name, results in comparison_results['models'].items():
            if 'error' in results:
                continue
            
            metrics = results.get('metrics', {})
            map50 = metrics.get('metrics/mAP50(B)', 0)
            
            if map50 > best_map:
                best_map = map50
                summary['best_model'] = model_name
            
            summary['metrics_comparison'][model_name] = {
                'mAP50': map50,
                'precision': metrics.get('metrics/precision(B)', 0),
                'recall': metrics.get('metrics/recall(B)', 0)
            }
        
        comparison_results['summary'] = summary
        
        # Print comparison summary
        print(f"\\n🏆 Model Comparison Results:")
        if summary['best_model']:
            print(f"   Best model: {summary['best_model']} (mAP@0.5: {best_map:.3f})")
        
        print(f"\\n📊 Detailed Comparison:")
        for model_name, metrics in summary['metrics_comparison'].items():
            print(f"   {model_name}:")
            print(f"     mAP@0.5: {metrics['mAP50']:.3f}")
            print(f"     Precision: {metrics['precision']:.3f}")
            print(f"     Recall: {metrics['recall']:.3f}")
        
        return comparison_results


def main():
    """Command line interface for model training"""
    parser = argparse.ArgumentParser(description="Train custom license plate detection model")
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name to train on')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Name for the trained model')
    
    # Optional arguments
    parser.add_argument('--config', type=str, default='small',
                       choices=['nano', 'small', 'medium', 'large'],
                       help='Training configuration (default: small)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model after training')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation data split (default: 0.2)')
    
    args = parser.parse_args()
    
    print("🤖 License Plate Reader - Custom Model Training")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = LPRModelTrainer()
        
        # Prepare dataset
        dataset_yaml = trainer.prepare_dataset(
            dataset_name=args.dataset,
            validation_split=args.validation_split
        )
        
        # Prepare training configuration
        custom_config = None
        if args.epochs or args.batch_size:
            custom_config = trainer.training_configs[args.config].copy()
            if args.epochs:
                custom_config['epochs'] = args.epochs
            if args.batch_size:
                custom_config['batch_size'] = args.batch_size
        
        # Train model
        training_results = trainer.train_model(
            dataset_yaml=dataset_yaml,
            model_name=args.model_name,
            config_name=args.config,
            custom_config=custom_config,
            resume=args.resume
        )
        
        print(f"\\n✅ Training completed successfully!")
        print(f"   Model: {training_results.get('final_model_path')}")
        
        # Evaluate if requested
        if args.evaluate and training_results.get('final_model_path'):
            print(f"\\n📊 Running evaluation...")
            eval_results = trainer.evaluate_model(
                training_results['final_model_path'],
                dataset_yaml
            )
            
            print(f"✅ Evaluation completed")
        
        print(f"\\n🎯 Next Steps:")
        print(f"   1. Test the model: python src/offline/standalone_detector.py --model {args.model_name}.pt")
        print(f"   2. Compare models: python scripts/compare_implementations.py")
        print(f"   3. Deploy to production: ./deployment/scripts/deploy.sh")
        
    except KeyboardInterrupt:
        print("\\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()