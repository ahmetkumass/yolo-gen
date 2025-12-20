#!/usr/bin/env python3
"""
Train YOLO + VLM

Usage:
    python train.py --data path/to/dataset.yaml
    python train.py --data path/to/dataset.yaml --model yolov8s.pt --epochs 100
    python train.py --data path/to/dataset.yaml --vlm
    python train.py --config configs/default.yaml
"""

import argparse
import gc
import yaml
from pathlib import Path
from datetime import datetime

import torch


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="YoloGen Training")

    # Config
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    parser.add_argument("--data", type=str, help="Path to dataset.yaml")

    # YOLO
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="")

    # VLM
    parser.add_argument("--vlm", action="store_true", help="Enable VLM training")
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--vlm-epochs", type=int, default=3)
    parser.add_argument("--vlm-precision", type=str, default="4bit")
    parser.add_argument("--vlm-max-samples", type=int, default=None, help="Max training samples for VLM")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-vlm-data", action="store_true")

    # Output
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--name", type=str)

    # Visualization
    parser.add_argument("--visualize", action="store_true", default=True, help="Generate visualizations after training")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false", help="Skip visualizations")

    args = parser.parse_args()

    # Load config if provided
    vlm_dataset_config = None
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        args.data = args.data or cfg.get('data')
        yolo_cfg = cfg.get('yolo', {})
        vlm_cfg = cfg.get('vlm', {})
        vlm_dataset_config = cfg.get('vlm_dataset')  # VLM dataset generation settings
        args.model = yolo_cfg.get('model', args.model)
        args.epochs = yolo_cfg.get('epochs', args.epochs)
        args.batch = yolo_cfg.get('batch', args.batch)
        args.vlm = vlm_cfg.get('enabled', args.vlm)
        args.vlm_model = vlm_cfg.get('model', args.vlm_model)
        args.vlm_epochs = vlm_cfg.get('epochs', args.vlm_epochs)
        args.vlm_precision = vlm_cfg.get('precision', args.vlm_precision)
        args.vlm_max_samples = vlm_cfg.get('max_samples', args.vlm_max_samples)
        # VLM training params
        args.vlm_batch_size = vlm_cfg.get('batch_size', 1)
        args.vlm_lr = vlm_cfg.get('lr', 2e-5)
        args.vlm_gradient_accumulation = vlm_cfg.get('gradient_accumulation', 4)
        args.vlm_lora_r = vlm_cfg.get('lora_r', 64)
        args.vlm_lora_alpha = vlm_cfg.get('lora_alpha', 16)
        args.vlm_lora_dropout = vlm_cfg.get('lora_dropout', 0.05)
        args.vlm_gradient_checkpointing = vlm_cfg.get('gradient_checkpointing', True)
        # Image size (affects GPU memory significantly)
        args.vlm_min_pixels = vlm_cfg.get('min_pixels', 256 * 28 * 28)
        args.vlm_max_pixels = vlm_cfg.get('max_pixels', 1280 * 28 * 28)

    if not args.data:
        parser.error("--data or --config required")

    # Resolve paths
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent / data_path

    # Output directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(__file__).parent / "runs"

    if args.name:
        name = args.name
    else:
        name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")

    output_dir = save_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  YoloGen Training")
    print("=" * 60)
    print(f"Dataset: {data_path}")
    print(f"Output: {output_dir}")
    print(f"YOLO: {args.model}, {args.epochs} epochs, batch={args.batch}")
    if args.vlm:
        print(f"VLM: {args.vlm_model}")
        print(f"     epochs={args.vlm_epochs}, batch={getattr(args, 'vlm_batch_size', 1)}, lr={getattr(args, 'vlm_lr', 2e-5)}")
        print(f"     lora_r={getattr(args, 'vlm_lora_r', 64)}, max_samples={args.vlm_max_samples}")
    else:
        print(f"VLM: Disabled")

    results = {}

    # ==================== YOLO TRAINING ====================
    if not args.skip_yolo:
        print("\n" + "=" * 60)
        print("  STAGE 1: YOLO Training")
        print("=" * 60)

        from yologen.core.trainer import YOLOTrainer

        trainer = YOLOTrainer(model=args.model, device=args.device)
        yolo_results = trainer.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            save_dir=str(output_dir),
            name="yolo",
        )
        results['yolo_weights'] = yolo_results['weights']
        print(f"YOLO weights: {results['yolo_weights']}")

        # Export ONNX
        try:
            onnx_path = trainer.export(format="onnx", imgsz=args.imgsz)
            print(f"ONNX exported: {onnx_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}")

        del trainer
        clear_gpu()

    # ==================== VLM DATASET ====================
    if args.vlm and not args.skip_vlm_data:
        print("\n" + "=" * 60)
        print("  STAGE 2: VLM Dataset Generation")
        print("=" * 60)

        from yologen.data.vlm_dataset import VLMDatasetGenerator

        generator = VLMDatasetGenerator(str(data_path), vlm_config=vlm_dataset_config)
        stats = generator.generate(force=True)
        vlm_data_dir = generator.data_path / 'vlm'
        print(f"Generated {stats.get('qa_pairs', 0)} Q&A pairs")
        print(f"Output: {vlm_data_dir}")

    # ==================== VLM TRAINING ====================
    if args.vlm:
        print("\n" + "=" * 60)
        print("  STAGE 3: VLM Training")
        print("=" * 60)

        # Find VLM data
        with open(data_path) as f:
            ds_cfg = yaml.safe_load(f)
        ds_root = Path(ds_cfg.get('path', data_path.parent))
        if not ds_root.is_absolute():
            ds_root = data_path.parent / ds_root
        vlm_data_dir = ds_root / 'vlm'

        if not (vlm_data_dir / 'train.jsonl').exists():
            print(f"VLM data not found at {vlm_data_dir}")
        else:
            from yologen.core.vlm_trainer import VLMTrainer

            trainer = VLMTrainer(
                model=args.vlm_model,
                precision=args.vlm_precision,
                lora_r=getattr(args, 'vlm_lora_r', 64),
                lora_alpha=getattr(args, 'vlm_lora_alpha', 16),
                lora_dropout=getattr(args, 'vlm_lora_dropout', 0.05),
                gradient_checkpointing=getattr(args, 'vlm_gradient_checkpointing', True),
                min_pixels=getattr(args, 'vlm_min_pixels', 256 * 28 * 28),
                max_pixels=getattr(args, 'vlm_max_pixels', 1280 * 28 * 28),
            )
            vlm_results = trainer.train(
                data=str(vlm_data_dir),
                epochs=args.vlm_epochs,
                batch_size=getattr(args, 'vlm_batch_size', 1),
                lr=getattr(args, 'vlm_lr', 2e-5),
                gradient_accumulation=getattr(args, 'vlm_gradient_accumulation', 4),
                save_dir=str(output_dir),
                name="vlm",
                max_samples=args.vlm_max_samples,
            )
            results['vlm_adapter'] = vlm_results['best_adapter']
            results['vlm_loss_history'] = vlm_results.get('loss_history', [])
            results['vlm_output_dir'] = vlm_results.get('output_dir')
            print(f"VLM adapter: {results['vlm_adapter']}")

            del trainer
            clear_gpu()

    # ==================== VISUALIZATION ====================
    if args.visualize and results.get('yolo_weights'):
        print("\n" + "=" * 60)
        print("  STAGE 4: Generating Visualizations")
        print("=" * 60)

        from yologen.utils.visualization import create_training_report

        # Find YOLO training directory
        yolo_dir = Path(results['yolo_weights']).parent.parent

        # Find validation images
        with open(data_path) as f:
            ds_cfg = yaml.safe_load(f)
        ds_root = Path(ds_cfg.get('path', data_path.parent))
        if not ds_root.is_absolute():
            ds_root = data_path.parent / ds_root
        val_images = ds_root / ds_cfg.get('val', 'images/val')

        # Generate report
        report = create_training_report(
            results_dir=yolo_dir,
            model_path=results['yolo_weights'],
            val_images_dir=val_images,
            save_dir=output_dir / "visualizations",
        )
        results['visualizations'] = report

    # ==================== VLM VISUALIZATION ====================
    if args.visualize and results.get('vlm_adapter'):
        print("\n" + "=" * 60)
        print("  STAGE 5: VLM Visualizations")
        print("=" * 60)

        from yologen.utils.visualization import plot_vlm_training_loss

        # Plot VLM loss curve
        if results.get('vlm_loss_history'):
            vlm_vis_dir = output_dir / "visualizations" / "vlm"
            vlm_vis_dir.mkdir(parents=True, exist_ok=True)

            loss_plot = plot_vlm_training_loss(
                results['vlm_loss_history'],
                vlm_vis_dir / "vlm_training_loss.png",
            )
            if loss_plot:
                print(f"VLM loss curve: {loss_plot}")
                if 'visualizations' not in results:
                    results['visualizations'] = {}
                results['visualizations']['vlm_loss'] = loss_plot

    # ==================== SUMMARY ====================
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"Output: {output_dir}")
    if results.get('yolo_weights'):
        print(f"YOLO: {results['yolo_weights']}")
    if results.get('vlm_adapter'):
        print(f"VLM: {results['vlm_adapter']}")

    # Print visualization locations
    if results.get('visualizations'):
        print("\nVisualizations:")
        for name, path in results['visualizations'].items():
            print(f"  {name}: {path}")

    print("\nInference:")
    print(f"  python predict.py --weights {results.get('yolo_weights', 'path/to/best.pt')} --source image.jpg")
    if results.get('vlm_adapter'):
        print(f"  python predict.py --weights {results['yolo_weights']} --source image.jpg --vlm --vlm-adapter {results['vlm_adapter']}")


if __name__ == "__main__":
    main()
