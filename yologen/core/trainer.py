"""
YOLO Trainer using Ultralytics

Provides high-level API for training YOLO models with best practices.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

from ultralytics import YOLO


class YOLOTrainer:
    """
    YOLO Trainer with Ultralytics backend.

    Example:
        trainer = YOLOTrainer(model="yolov8n.pt")
        trainer.train(data="dataset.yaml", epochs=100)
        trainer.export(format="onnx")
    """

    def __init__(
        self,
        model: str = "yolov8n.pt",
        device: str = "",
    ):
        """
        Initialize YOLO trainer.

        Args:
            model: Model variant or path to weights
                   Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
            device: Device to use (cuda, cpu, mps, or device id)
        """
        self.model_name = model
        self.device = device
        self.model = YOLO(model)
        self.results = None
        self.best_weights = None

    def train(
        self,
        data: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        patience: int = 50,
        save_dir: str = None,
        name: str = None,
        # Augmentation
        mosaic: float = 1.0,
        mixup: float = 0.0,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        fliplr: float = 0.5,
        erasing: float = 0.4,
        close_mosaic: int = 10,
        # Optimizer
        optimizer: str = "auto",
        lr0: float = 0.01,
        lrf: float = 0.01,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train YOLO model.

        Args:
            data: Path to dataset.yaml
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size (-1 for auto)
            patience: Early stopping patience
            save_dir: Directory to save results
            name: Experiment name
            mosaic: Mosaic augmentation probability
            mixup: MixUp augmentation probability
            hsv_h/s/v: HSV augmentation parameters
            degrees: Rotation degrees
            translate: Translation factor
            scale: Scale factor
            fliplr: Horizontal flip probability
            erasing: Random erasing probability
            close_mosaic: Disable mosaic in last N epochs
            optimizer: Optimizer (auto, SGD, Adam, AdamW)
            lr0: Initial learning rate
            lrf: Final learning rate factor
            **kwargs: Additional Ultralytics arguments

        Returns:
            Training results dictionary
        """
        # Set output directory
        if save_dir is None:
            save_dir = Path(__file__).parent.parent.parent / "runs" / "detect"

        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"train_{timestamp}"

        # Train
        self.results = self.model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device if self.device else None,
            patience=patience,
            project=str(save_dir),
            name=name,
            # Augmentation
            mosaic=mosaic,
            mixup=mixup,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            fliplr=fliplr,
            erasing=erasing,
            close_mosaic=close_mosaic,
            # Optimizer
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            # Best practices
            save=True,
            plots=True,
            **kwargs,
        )

        # Store best weights path
        self.best_weights = Path(save_dir) / name / "weights" / "best.pt"

        return {
            "weights": str(self.best_weights),
            "results": self.results.results_dict if self.results else None,
        }

    def export(
        self,
        format: str = "onnx",
        imgsz: int = 640,
        simplify: bool = True,
        **kwargs,
    ) -> str:
        """
        Export model to deployment format.

        Args:
            format: Export format (onnx, torchscript, tensorrt, etc.)
            imgsz: Input image size
            simplify: Simplify ONNX graph
            **kwargs: Additional export arguments

        Returns:
            Path to exported model
        """
        if self.best_weights and self.best_weights.exists():
            model = YOLO(str(self.best_weights))
        else:
            model = self.model

        return model.export(
            format=format,
            imgsz=imgsz,
            simplify=simplify,
            **kwargs,
        )

    def validate(self, data: str = None, **kwargs) -> Dict[str, Any]:
        """
        Validate model on dataset.

        Args:
            data: Path to dataset.yaml (uses training data if None)
            **kwargs: Additional validation arguments

        Returns:
            Validation metrics
        """
        if self.best_weights and self.best_weights.exists():
            model = YOLO(str(self.best_weights))
        else:
            model = self.model

        results = model.val(data=data, **kwargs)
        return results.results_dict


def train_yolo(
    data: str,
    model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
    **kwargs,
) -> Dict[str, Any]:
    """
    Train YOLO model (convenience function).

    Args:
        data: Path to dataset.yaml
        model: Model variant
        epochs: Number of epochs
        imgsz: Image size
        batch: Batch size
        device: Device to use
        **kwargs: Additional training arguments

    Returns:
        Training results
    """
    trainer = YOLOTrainer(model=model, device=device)
    return trainer.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        **kwargs,
    )
