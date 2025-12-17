"""
YoloGen CLI

Command-line interface for training and inference.
"""

import argparse
import sys
from pathlib import Path


def train_cli():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="YoloGen Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train YOLO only
  yologen-train --data dataset.yaml --model yolov8n.pt --epochs 100

  # Train YOLO + VLM
  yologen-train --data dataset.yaml --model yolov8n.pt --epochs 100 --vlm

  # Use config file
  yologen-train --config config.yaml
        """
    )

    parser.add_argument("--config", type=str, help="Path to config.yaml")
    parser.add_argument("--data", type=str, help="Path to dataset.yaml")

    # YOLO
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--epochs", type=int, default=100, help="YOLO epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="", help="Device")

    # VLM
    parser.add_argument("--vlm", action="store_true", help="Enable VLM training")
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--vlm-epochs", type=int, default=3, help="VLM epochs")
    parser.add_argument("--vlm-precision", type=str, default="4bit", choices=["4bit", "8bit", "fp16"])

    # Output
    parser.add_argument("--save-dir", type=str, help="Output directory")
    parser.add_argument("--name", type=str, help="Experiment name")

    args = parser.parse_args()

    if not args.config and not args.data:
        parser.error("Either --config or --data is required")

    # Import here to avoid slow startup
    from yologen.core.trainer import YOLOTrainer
    from yologen.core.vlm_trainer import VLMTrainer
    from yologen.data.vlm_dataset import VLMDatasetGenerator

    # Train YOLO
    print("=" * 60)
    print("  YoloGen Training")
    print("=" * 60)

    yolo_trainer = YOLOTrainer(model=args.model, device=args.device)
    yolo_results = yolo_trainer.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        save_dir=args.save_dir,
        name=args.name,
    )
    print(f"YOLO weights: {yolo_results['weights']}")

    # Train VLM
    if args.vlm:
        print("\nGenerating VLM dataset...")
        generator = VLMDatasetGenerator(args.data)
        generator.generate(force=True)

        print("\nTraining VLM...")
        vlm_trainer = VLMTrainer(
            model=args.vlm_model,
            precision=args.vlm_precision,
        )
        vlm_results = vlm_trainer.train(
            data=str(generator.data_path / 'vlm'),
            epochs=args.vlm_epochs,
        )
        print(f"VLM adapter: {vlm_results['best_adapter']}")

    print("\nDone!")


def predict_cli():
    """CLI entry point for prediction."""
    parser = argparse.ArgumentParser(description="YoloGen Prediction")

    parser.add_argument("--weights", type=str, required=True, help="Model weights")
    parser.add_argument("--source", type=str, required=True, help="Image/folder path")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--device", type=str, default="", help="Device")
    parser.add_argument("--save", action="store_true", help="Save results")
    parser.add_argument("--show", action="store_true", help="Show results")

    # VLM
    parser.add_argument("--vlm", action="store_true", help="Enable VLM Q&A")
    parser.add_argument("--vlm-adapter", type=str, help="VLM adapter path")
    parser.add_argument("--vlm-question", type=str, default="What is in the red marked area?")

    args = parser.parse_args()

    from yologen.core.predictor import YOLOPredictor, UnifiedPredictor

    if args.vlm:
        predictor = UnifiedPredictor(
            yolo_weights=args.weights,
            vlm_adapter=args.vlm_adapter,
        )
        results = predictor.predict(
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            vlm_question=args.vlm_question,
            save=args.save,
        )
    else:
        predictor = YOLOPredictor(weights=args.weights, device=args.device)
        results = predictor.predict(
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
        )

    # Print results
    for r in results:
        print(f"\n{r['path']}:")
        for det in r['detections']:
            line = f"  {det['class_name']}: {det['confidence']:.2f}"
            if det.get('vlm_answer'):
                line += f" | VLM: {det['vlm_answer'][:50]}..."
            print(line)


def generate_cli():
    """CLI entry point for VLM dataset generation."""
    parser = argparse.ArgumentParser(description="Generate VLM Dataset")

    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force regeneration")

    args = parser.parse_args()

    from yologen.data.vlm_dataset import VLMDatasetGenerator

    generator = VLMDatasetGenerator(args.data)
    stats = generator.generate(output_dir=args.output, force=args.force)

    print(f"\nGenerated: {stats.get('qa_pairs', 0)} Q&A pairs")


if __name__ == "__main__":
    train_cli()
