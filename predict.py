#!/usr/bin/env python3
"""
YoloGen Prediction

Usage:
    # YOLO only
    python predict.py --weights best.pt --source image.jpg

    # YOLO + VLM
    python predict.py --weights best.pt --source image.jpg --vlm --vlm-adapter vlm_adapter/

    # Save results
    python predict.py --weights best.pt --source folder/ --save
"""

import argparse
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="YoloGen Prediction")

    # Required
    parser.add_argument("--weights", type=str, required=True, help="YOLO weights")
    parser.add_argument("--source", type=str, required=True, help="Image/folder path")

    # Detection settings
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections")
    parser.add_argument("--device", type=str, default="", help="Device")

    # VLM settings
    parser.add_argument("--vlm", action="store_true", help="Enable VLM Q&A")
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--vlm-adapter", type=str, help="VLM adapter path")
    parser.add_argument("--vlm-precision", type=str, default="4bit")
    parser.add_argument("--vlm-question", type=str, default="What is in the red marked area?")

    # Output
    parser.add_argument("--save", action="store_true", help="Save annotated images")
    parser.add_argument("--save-txt", action="store_true", help="Save results to txt")
    parser.add_argument("--save-dir", type=str, help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display results")

    args = parser.parse_args()

    print("=" * 60)
    print("  YoloGen Prediction")
    print("=" * 60)
    print(f"Weights: {args.weights}")
    print(f"Source: {args.source}")
    print(f"VLM: {'Enabled' if args.vlm else 'Disabled'}")

    # Set output directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(__file__).parent / "runs" / "predict" / f"exp_{timestamp}"

    if args.vlm:
        from yologen.core.predictor import UnifiedPredictor

        predictor = UnifiedPredictor(
            yolo_weights=args.weights,
            vlm_model=args.vlm_model,
            vlm_adapter=args.vlm_adapter,
            vlm_precision=args.vlm_precision,
            device=args.device,
        )
        results = predictor.predict(
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            vlm_question=args.vlm_question,
            save=args.save,
            save_dir=str(save_dir) if args.save else None,
        )
    else:
        from yologen.core.predictor import YOLOPredictor

        predictor = YOLOPredictor(weights=args.weights, device=args.device)
        results = predictor.predict(
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            save=args.save,
            save_dir=str(save_dir) if args.save else None,
        )

    # Print results
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)

    total_detections = 0
    for r in results:
        print(f"\n{Path(r['path']).name}:")
        for det in r['detections']:
            total_detections += 1
            bbox = det['bbox']
            line = f"  [{det['class_name']}] conf={det['confidence']:.2f} bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]"
            print(line)
            if det.get('vlm_answer'):
                answer = det['vlm_answer'][:60] + "..." if len(det['vlm_answer']) > 60 else det['vlm_answer']
                print(f"    VLM: {answer}")

    print(f"\nTotal: {len(results)} images, {total_detections} detections")

    if args.save:
        print(f"Results saved to: {save_dir}")

    # Save to txt
    if args.save_txt:
        txt_path = save_dir / "results.txt"
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(txt_path, 'w') as f:
            for r in results:
                f.write(f"# {r['path']}\n")
                for det in r['detections']:
                    bbox = det['bbox']
                    f.write(f"{det['class_id']} {det['confidence']:.4f} {bbox[0]:.1f} {bbox[1]:.1f} {bbox[2]:.1f} {bbox[3]:.1f}\n")
                    if det.get('vlm_answer'):
                        f.write(f"# VLM: {det['vlm_answer']}\n")
                f.write("\n")
        print(f"Results saved to: {txt_path}")


if __name__ == "__main__":
    main()
