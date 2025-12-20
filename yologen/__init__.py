"""
YoloGen - Unified YOLO + VLM Training Framework

A production-ready framework for training YOLO object detectors
and Vision-Language Models on the same dataset.

Features:
- YOLO training with Ultralytics (pretrained COCO weights)
- VLM fine-tuning with QLoRA (Qwen-VL)
- Automatic Q&A dataset generation from detection labels
- Unified inference pipeline

Example:
    # Train YOLO
    from yologen import train_yolo
    train_yolo(data="dataset.yaml", model="yolov8n.pt", epochs=100)

    # Train VLM
    from yologen import train_vlm
    train_vlm(data="vlm_data/", epochs=3)

    # Inference
    from yologen import predict
    results = predict(weights="best.pt", source="image.jpg")

MIT License
"""

__version__ = "0.1.0"
__author__ = "YoloGen Contributors"
__license__ = "MIT"

from yologen.core.trainer import YOLOTrainer, train_yolo
from yologen.core.vlm_trainer import VLMTrainer, train_vlm
from yologen.core.predictor import YOLOPredictor, VLMPredictor, UnifiedPredictor, predict
from yologen.data.vlm_dataset import VLMDatasetGenerator, generate_vlm_dataset

__all__ = [
    # Version
    "__version__",
    # Trainers
    "YOLOTrainer",
    "VLMTrainer",
    "train_yolo",
    "train_vlm",
    # Predictors
    "YOLOPredictor",
    "VLMPredictor",
    "UnifiedPredictor",
    "predict",
    # Data
    "VLMDatasetGenerator",
    "generate_vlm_dataset",
]
