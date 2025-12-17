#!/usr/bin/env python3
"""
YoloGen Visualization Tool

Generate visualizations from training results.

Usage:
    # Visualize training results
    python visualize.py --results runs/exp_xxx/yolo

    # Visualize with prediction samples
    python visualize.py --results runs/exp_xxx/yolo --val-images data/car_detection/images/val

    # Just run predictions on images
    python visualize.py --weights best.pt --source images/ --save-dir outputs/
"""

import argparse
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="YoloGen Visualization")

    # Mode 1: Visualize training results
    parser.add_argument("--results", type=str, help="Path to training results directory")
    parser.add_argument("--val-images", type=str, help="Path to validation images for prediction samples")

    # Mode 2: Run predictions
    parser.add_argument("--weights", type=str, help="Path to YOLO weights for predictions")
    parser.add_argument("--source", type=str, help="Path to images for prediction")

    # Common
    parser.add_argument("--save-dir", type=str, help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--num-images", type=int, default=16, help="Number of images to visualize")
    parser.add_argument("--grid", type=str, default="4x4", help="Grid size (e.g., 4x4)")

    args = parser.parse_args()

    # Parse grid size
    grid_rows, grid_cols = map(int, args.grid.split('x'))

    # Set output directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(__file__).parent / "runs" / "visualize" / f"exp_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  YoloGen Visualization")
    print("=" * 60)

    # Mode 1: Training results visualization
    if args.results:
        print(f"\nVisualizing training results: {args.results}")

        from yologen.utils.visualization import create_training_report

        results_dir = Path(args.results)
        val_images = Path(args.val_images) if args.val_images else None

        # Auto-detect weights
        weights = results_dir / "weights" / "best.pt"
        if not weights.exists():
            weights = None

        report = create_training_report(
            results_dir=results_dir,
            model_path=weights,
            val_images_dir=val_images,
            save_dir=save_dir,
        )

        print("\nGenerated visualizations:")
        for name, path in report.items():
            print(f"  {name}: {path}")

    # Mode 2: Prediction visualization
    elif args.weights and args.source:
        print(f"\nGenerating predictions: {args.source}")
        print(f"Using weights: {args.weights}")

        from yologen.utils.visualization import visualize_predictions

        pred_path = visualize_predictions(
            model_path=args.weights,
            images_dir=args.source,
            save_dir=save_dir,
            num_images=args.num_images,
            conf=args.conf,
            imgsz=args.imgsz,
            grid_size=(grid_rows, grid_cols),
        )

        if pred_path:
            print(f"\nPrediction grid: {pred_path}")
            print(f"Individual predictions: {save_dir / 'predictions'}")

    # Mode 3: Compare ground truth vs predictions
    elif args.weights:
        print("Please provide --source for predictions or --results for training visualization")
        return

    else:
        parser.print_help()
        return

    print(f"\nOutput saved to: {save_dir}")


if __name__ == "__main__":
    main()
