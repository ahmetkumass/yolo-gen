"""
YOLO and Unified Predictors

Inference engines for object detection and combined YOLO + VLM.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import cv2
import numpy as np
from ultralytics import YOLO


class YOLOPredictor:
    """
    YOLO Predictor for object detection.

    Example:
        predictor = YOLOPredictor(weights="best.pt")
        results = predictor.predict("image.jpg")
    """

    def __init__(
        self,
        weights: str,
        device: str = "",
    ):
        """
        Initialize YOLO predictor.

        Args:
            weights: Path to model weights
            device: Device to use
        """
        self.model = YOLO(weights)
        self.device = device
        self.class_names = self.model.names

    def predict(
        self,
        source: Union[str, np.ndarray, List],
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        save: bool = False,
        save_dir: str = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Run prediction on source.

        Args:
            source: Image path, numpy array, or list of sources
            imgsz: Inference image size
            conf: Confidence threshold
            iou: IoU threshold for NMS
            max_det: Maximum detections per image
            save: Save annotated results
            save_dir: Directory to save results
            **kwargs: Additional arguments

        Returns:
            List of detection results per image
        """
        # Set output directory
        if save and save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path(__file__).parent.parent.parent / "runs" / "detect" / f"predict_{timestamp}"

        # Run prediction
        results = self.model.predict(
            source=source,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=self.device if self.device else None,
            save=save,
            project=str(Path(save_dir).parent) if save_dir else None,
            name=str(Path(save_dir).name) if save_dir else None,
            **kwargs,
        )

        # Format results
        all_results = []
        for result in results:
            detections = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "class_id": cls_id,
                    "class_name": self.class_names[cls_id],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].cpu().numpy().tolist(),
                })

            all_results.append({
                "path": result.path,
                "detections": detections,
            })

        return all_results


class VLMPredictor:
    """
    VLM-only Predictor for Vision-Language Model inference.

    Use this when you have bounding boxes from another source
    or want to ask questions about specific image regions.

    Example:
        predictor = VLMPredictor(vlm_adapter="vlm_adapter/")
        answer = predictor.predict(
            image="image.jpg",
            bbox=[100, 100, 300, 300],
            question="What is in the red box?"
        )
    """

    def __init__(
        self,
        vlm_model: str = "Qwen/Qwen3-VL-4B-Instruct",
        vlm_adapter: str = None,
        vlm_precision: str = "4bit",
        device: str = "",
        box_color: tuple = None,
        box_thickness: int = None,
    ):
        """
        Initialize VLM-only predictor.

        Args:
            vlm_model: VLM model name (Qwen2.5-VL or Qwen3-VL)
            vlm_adapter: Path to VLM adapter (fine-tuned weights)
            vlm_precision: VLM precision (4bit, 8bit, fp16)
            device: Device to use
            box_color: RGB tuple for box color (None = load from training config)
            box_thickness: Box line thickness (None = load from training config)
        """
        self.vlm = None
        self.vlm_model = vlm_model
        self.vlm_adapter = vlm_adapter
        self.vlm_precision = vlm_precision
        self.device = device

        # VLM settings (None = will be loaded from config, otherwise use provided)
        self._box_color_override = box_color
        self._box_thickness_override = box_thickness
        self.box_thickness = box_thickness if box_thickness is not None else 3
        self.box_color = box_color if box_color is not None else (255, 0, 0)
        self.system_prompt = None

    def _load_vlm(self):
        """Load VLM model lazily."""
        if self.vlm is not None:
            return

        parent_path = Path(__file__).parent.parent.parent.parent
        if str(parent_path) not in sys.path:
            sys.path.insert(0, str(parent_path))

        # Load box settings from VLM config if available
        if self.vlm_adapter:
            self._load_vlm_config()

        try:
            from yologen.models.vlm.qwen import create_qwen_vlm

            self.vlm = create_qwen_vlm(
                model_name=self.vlm_model,
                load_in_4bit=(self.vlm_precision == "4bit"),
                load_in_8bit=(self.vlm_precision == "8bit"),
                use_lora=False,
                gradient_checkpointing=False,
            )
            self.vlm.load_model()

            if self.vlm_adapter:
                self.vlm.load_adapter(self.vlm_adapter)

        except ImportError as e:
            raise ImportError(
                f"VLM dependencies not installed: {e}\n"
                "Install with: pip install transformers accelerate peft bitsandbytes qwen-vl-utils"
            )

    def _load_vlm_config(self):
        """Load VLM config for consistent box settings (respects constructor overrides)."""
        import json

        adapter_path = Path(self.vlm_adapter)

        search_paths = [
            adapter_path / "config.json",
            adapter_path.parent / "config.json",
            adapter_path.parent.parent / "vlm" / "config.json",
        ]

        for config_path in search_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    # Only load from config if not overridden in constructor
                    if self._box_thickness_override is None:
                        self.box_thickness = config.get('box_thickness', self.box_thickness)
                    if self._box_color_override is None:
                        box_color = config.get('box_color', list(self.box_color))
                        self.box_color = tuple(box_color) if isinstance(box_color, list) else box_color
                    self.system_prompt = config.get('system_prompt')
                    print(f"Loaded VLM config: box_thickness={self.box_thickness}, box_color={self.box_color}")
                    return
                except Exception:
                    continue

    def predict(
        self,
        image: str,
        bbox: List[float] = None,
        question: str = "What is in the red marked area?",
    ) -> str:
        """
        Run VLM inference on an image region.

        Args:
            image: Image path
            bbox: Bounding box [x1, y1, x2, y2] (optional, if None asks about whole image)
            question: Question to ask VLM

        Returns:
            VLM answer string
        """
        self._load_vlm()

        try:
            answer = self.vlm.generate(
                image=image,
                question=question,
                bbox=bbox,
                box_thickness=self.box_thickness,
                box_color=self.box_color,
                system_prompt=self.system_prompt,
            )
            return answer
        except Exception as e:
            return f"Error: {str(e)}"

    def predict_batch(
        self,
        image: str,
        bboxes: List[List[float]],
        question: str = "What is in the red marked area?",
    ) -> List[str]:
        """
        Run VLM inference on multiple regions of an image.

        Args:
            image: Image path
            bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            question: Question to ask VLM

        Returns:
            List of VLM answers
        """
        answers = []
        for bbox in bboxes:
            answer = self.predict(image=image, bbox=bbox, question=question)
            answers.append(answer)
        return answers


class UnifiedPredictor:
    """
    Unified YOLO + VLM Predictor.

    Combines object detection with Vision-Language Model Q&A.

    Example:
        predictor = UnifiedPredictor(
            yolo_weights="best.pt",
            vlm_adapter="vlm_adapter/"
        )
        results = predictor.predict("image.jpg", question="What is in the red box?")
    """

    def __init__(
        self,
        yolo_weights: str,
        vlm_model: str = "Qwen/Qwen3-VL-4B-Instruct",
        vlm_adapter: str = None,
        vlm_precision: str = "4bit",
        device: str = "",
        box_color: tuple = None,
        box_thickness: int = None,
    ):
        """
        Initialize unified predictor.

        Args:
            yolo_weights: Path to YOLO weights
            vlm_model: VLM model name (Qwen2.5-VL or Qwen3-VL)
            vlm_adapter: Path to VLM adapter
            vlm_precision: VLM precision (4bit, 8bit, fp16)
            device: Device to use
            box_color: RGB tuple for box color (None = load from training config)
            box_thickness: Box line thickness (None = load from training config)
        """
        self.yolo = YOLO(yolo_weights)
        self.class_names = self.yolo.names
        self.device = device

        self.vlm = None
        self.vlm_model = vlm_model
        self.vlm_adapter = vlm_adapter
        self.vlm_precision = vlm_precision

        # VLM settings (None = will be loaded from config, otherwise use provided)
        self._box_color_override = box_color
        self._box_thickness_override = box_thickness
        self.box_thickness = box_thickness if box_thickness is not None else 3
        self.box_color = box_color if box_color is not None else (255, 0, 0)
        self.system_prompt = None

    def _load_vlm(self):
        """Load VLM model lazily."""
        if self.vlm is not None:
            return

        parent_path = Path(__file__).parent.parent.parent.parent
        if str(parent_path) not in sys.path:
            sys.path.insert(0, str(parent_path))

        # Load box settings from VLM config if available
        if self.vlm_adapter:
            self._load_vlm_config()

        try:
            from yologen.models.vlm.qwen import create_qwen_vlm

            self.vlm = create_qwen_vlm(
                model_name=self.vlm_model,
                load_in_4bit=(self.vlm_precision == "4bit"),
                load_in_8bit=(self.vlm_precision == "8bit"),
                use_lora=False,  # Don't apply LoRA during loading
                gradient_checkpointing=False,
            )
            self.vlm.load_model()

            if self.vlm_adapter:
                # Load adapter via PeftModel
                self.vlm.load_adapter(self.vlm_adapter)

        except ImportError as e:
            raise ImportError(
                f"VLM dependencies not installed: {e}\n"
                "Install with: pip install transformers accelerate peft bitsandbytes qwen-vl-utils"
            )

    def _load_vlm_config(self):
        """Load VLM config for consistent box settings (respects constructor overrides)."""
        import json

        adapter_path = Path(self.vlm_adapter)

        # Try to find config.json in parent directories
        # Structure: runs/exp/vlm/best/ -> look for vlm/config.json or ../config.json
        search_paths = [
            adapter_path / "config.json",
            adapter_path.parent / "config.json",
            adapter_path.parent.parent / "vlm" / "config.json",
        ]

        for config_path in search_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    # Only load from config if not overridden in constructor
                    if self._box_thickness_override is None:
                        self.box_thickness = config.get('box_thickness', self.box_thickness)
                    if self._box_color_override is None:
                        box_color = config.get('box_color', list(self.box_color))
                        self.box_color = tuple(box_color) if isinstance(box_color, list) else box_color
                    self.system_prompt = config.get('system_prompt')
                    print(f"Loaded VLM config: box_thickness={self.box_thickness}, box_color={self.box_color}")
                    return
                except Exception:
                    continue

    def predict(
        self,
        source: str,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        use_vlm: bool = True,
        vlm_question: str = "What is in the red marked area?",
        save: bool = False,
        save_dir: str = None,
    ) -> List[Dict]:
        """
        Run unified prediction.

        Args:
            source: Image path or folder
            imgsz: Image size
            conf: Confidence threshold
            iou: IoU threshold
            use_vlm: Enable VLM Q&A
            vlm_question: Question to ask VLM
            save: Save results
            save_dir: Output directory

        Returns:
            List of results per image
        """
        # Set output directory
        if save:
            if save_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = Path(__file__).parent.parent.parent / "runs" / "unified" / f"predict_{timestamp}"
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Get source files
        source_path = Path(source)
        if source_path.is_file():
            image_files = [source_path]
        elif source_path.is_dir():
            image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png")) + \
                          list(source_path.glob("*.jpeg"))
        else:
            raise FileNotFoundError(f"Source not found: {source}")

        # Load VLM if needed
        if use_vlm:
            self._load_vlm()

        all_results = []

        for img_path in image_files:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Run YOLO
            yolo_results = self.yolo.predict(
                source=str(img_path),
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=self.device if self.device else None,
                verbose=False,
            )

            # Process detections
            detections = []
            for result in yolo_results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy()

                    det = {
                        "class_id": cls_id,
                        "class_name": self.class_names[cls_id],
                        "confidence": float(box.conf[0]),
                        "bbox": xyxy.tolist(),
                        "vlm_answer": None,
                    }

                    # Run VLM Q&A
                    if use_vlm and self.vlm is not None:
                        try:
                            answer = self.vlm.generate(
                                image=str(img_path),
                                question=vlm_question,
                                bbox=xyxy.tolist(),
                                box_thickness=self.box_thickness,
                                box_color=self.box_color,
                                system_prompt=self.system_prompt,
                            )
                            det["vlm_answer"] = answer
                        except Exception as e:
                            det["vlm_answer"] = f"Error: {str(e)[:50]}"

                    detections.append(det)

            # Save annotated image + results text
            if save:
                img_annotated = self._draw_detections(img_rgb, detections)
                save_path = save_dir / f"{img_path.stem}_result.jpg"
                cv2.imwrite(str(save_path), cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))

                # Save VLM results as text file
                txt_path = save_dir / f"{img_path.stem}_result.txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"Image: {img_path.name}\n")
                    f.write("=" * 50 + "\n\n")
                    for i, det in enumerate(detections):
                        f.write(f"Detection {i+1}:\n")
                        f.write(f"  Class: {det['class_name']}\n")
                        f.write(f"  Confidence: {det['confidence']:.2f}\n")
                        f.write(f"  BBox: {det['bbox']}\n")
                        if det.get('vlm_answer'):
                            f.write(f"  VLM Answer: {det['vlm_answer']}\n")
                        f.write("\n")

            all_results.append({
                "path": str(img_path),
                "detections": detections,
            })

        return all_results

    def _draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw detection boxes and labels on image."""
        img = image.copy()
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_id = det["class_id"]
            color = colors[cls_id % len(colors)]

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 5, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), thickness)

        return img


def predict(
    weights: str,
    source: str,
    conf: float = 0.25,
    save: bool = False,
    **kwargs,
) -> List[Dict]:
    """
    Run YOLO prediction (convenience function).

    Args:
        weights: Path to model weights
        source: Image source
        conf: Confidence threshold
        save: Save results
        **kwargs: Additional arguments

    Returns:
        List of detection results
    """
    predictor = YOLOPredictor(weights=weights)
    return predictor.predict(source=source, conf=conf, save=save, **kwargs)
