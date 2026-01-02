# YoloGen

**Train YOLO + VLM with one command. No extra labeling.**

```
Image + YOLO labels → Auto-generate VLM training data → Fine-tuned model
```

Train object detection and natural language description models from a standard YOLO dataset. VLM training data is auto-generated from YOLO labels.

## Use Cases

YOLO localizes objects (bbox) → VLM analyzes and returns structured response:

| Scenario | VLM Response |
|----------|--------------|
| **Defect Detection** | `{"defect": true, "type": "scratch", "size": "2mm"}` |
| **Weapon Detection** | `{"weapon": true, "type": "rifle"}` |
| **Vehicle Damage** | `{"damaged": true, "part": "front bumper"}` |
| **Medical Imaging** | `{"finding": true, "type": "nodule", "size": "6mm"}` |

## Why YOLO + VLM?

- **YOLO alone**: Fast but not enough for production-level accuracy
- **VLM alone**: Smart but too slow for production
- **YOLO + VLM**: Fast detection + VLM adds detailed descriptions, classification, and false positive filtering

## Built With

- [Ultralytics YOLOv8/v11](https://github.com/ultralytics/ultralytics) - State-of-the-art YOLO implementation
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) / [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) - Vision-Language Models
- [PEFT/QLoRA](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Standard YOLO format:
```
data/my_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

Example `dataset.yaml`:
```yaml
path: .  # Dataset root (relative to this file)
train: images/train
val: images/val

names:
  0: class_a
  1: class_b
```

### 3. Configure

Edit `configs/default.yaml` and set your dataset path:
```yaml
data: data/my_dataset/dataset.yaml
```

### 4. Train

```bash
python train.py --config configs/default.yaml
```

This will:
1. Train YOLO (100 epochs)
2. Generate VLM dataset (Q&A pairs with red boxes)
3. Train VLM with QLoRA (3 epochs)
4. Export ONNX
5. Generate visualizations

### 5. Predict

```bash
# YOLO only
python predict.py --weights runs/exp_xxx/yolo/weights/best.pt --source image.jpg

# YOLO + VLM
python predict.py --weights runs/exp_xxx/yolo/weights/best.pt --source image.jpg \
    --vlm --vlm-adapter runs/exp_xxx/vlm/best
```

### 6. Evaluate (Compare Base vs Fine-tuned)

```bash
jupyter notebook examples/compare_vlm.ipynb
```

Compare your fine-tuned VLM against the base model to measure improvements.

### Python API

```python
from yologen.core.predictor import YOLOPredictor, VLMPredictor, UnifiedPredictor

# YOLO only
yolo = YOLOPredictor(weights="best.pt")
results = yolo.predict("image.jpg")

# VLM only (for images with existing bounding boxes)
vlm = VLMPredictor(vlm_adapter="vlm/best")
answer = vlm.predict(image="image.jpg", bbox=[100, 100, 300, 300], question="What is this?")

# YOLO + VLM combined
predictor = UnifiedPredictor(yolo_weights="best.pt", vlm_adapter="vlm/best")
results = predictor.predict(source="image.jpg", vlm_question="What is in the red box?")
```

## Configuration

Copy and edit `configs/default.yaml`:

```yaml
# YOLO settings
yolo:
  model: yolov8n.pt
  epochs: 100
  batch: 16

# VLM settings
vlm:
  enabled: true
  # Qwen 3 VL (recommended): 2B, 4B, 8B
  # Qwen 2.5 VL: 3B, 7B
  model: Qwen/Qwen3-VL-4B-Instruct
  epochs: 3
  precision: 4bit
  max_samples: 10000
  # max_pixels: auto-calculated based on model version

# Visual grounding (red box)
vlm_dataset:
  box_color: [0, 0, 255]    # BGR Red
  box_thickness: 3
  system_prompt: |
    You are an object detection assistant.
    Identify objects in red marked areas clearly.
```

## Output Structure

```
runs/exp_20251217_xxx/
├── yolo/
│   └── weights/
│       ├── best.pt           # YOLO model
│       └── best.onnx         # ONNX export
├── vlm/
│   └── best/                 # VLM adapter (~150MB)
└── visualizations/
    ├── training_curves.png
    └── prediction_samples.png
```

## Key Features

| Feature | Description |
|---------|-------------|
| Single Config | One YAML controls everything |
| Sequential Training | YOLO → VLM automatically |
| QLoRA | 7B VLM training with 4-bit quantization |
| Visual Grounding | Red boxes link detection to VLM |
| Configurable | Colors, prompts, models all in YAML |

## Requirements

- Python 3.10+

### GPU Memory Usage

| Task | VRAM |
|------|------|
| YOLO training | 4-12 GB |
| VLM 2B-3B | ~14-18 GB |
| VLM 4B | ~18-20 GB |
| VLM 7B-8B | ~24-28 GB |

*VLM memory depends on `max_pixels` setting. Values above are for 4-bit QLoRA with default pixel settings.*

## Example Results

**Input**: Product image from assembly line

**YOLO Output**:
```
[defect] conf=0.92 bbox=[120, 340, 280, 520]
```

**VLM Output**:
```json
{"defect": true, "type": "scratch", "size": "3mm"}
```

## FAQ

**Do I need to manually write VLM training data?**
No. YoloGen automatically generates Q&A pairs from your YOLO labels. Just prepare standard YOLO format dataset.

**How many images do I need?**
Minimum ~100 images for YOLO, ~500+ recommended for better VLM results.

**Can I use only YOLO without VLM?**
Yes. Set `vlm.enabled: false` in config, or just use `predict.py` without `--vlm` flag.

**How much VRAM do I need?**
See [GPU Memory Usage](#gpu-memory-usage) table above. RTX 4090 (24GB) can train both 3B and 7B models with default settings.

**How do I customize VLM responses?**
Edit `system_prompt` and `details` under `vlm_dataset` section in your config file.

## License

MIT

Note: This project uses [Ultralytics](https://github.com/ultralytics/ultralytics) which is licensed under AGPL-3.0. See their license for details.
