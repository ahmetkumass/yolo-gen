# YoloGen

**Detect objects → Describe them in natural language. One pipeline.**

```
Input Image → YOLO finds "defect" → Model describes "2mm crack on solder joint"
```

Object detectors give you **where**, but not **what**. Vision models are too slow for production. YoloGen combines both: YOLO finds objects fast, then a local model describes them. Training data is auto-generated — no extra labeling needed.

## Real-World Scenarios

| Scenario | YOLO Detects | VLM Describes |
|----------|--------------|---------------|
| **Quality Control** | Defect location | "Surface scratch, 2mm length, located on edge" |
| **Security** | Person, vehicle | "Person near entrance, wearing dark hoodie, carrying backpack" |
| **Medical** | Lesion area | "6mm nodule with irregular borders, darker pigmentation" |
| **Retail** | Product on shelf | "2 items remaining on shelf, front row empty" |
| **Agriculture** | Crop, pest | "Small green insects on leaf underside, early cluster" |
| **False Positive Filter** | "Person" (conf: 0.6) | "This is a mannequin, not a real person" → filtered out |

## What It Does

```
Image → YOLO → "defect at [x1,y1,x2,y2]" → VLM → "Crack in solder joint, 1.5mm, critical defect"
```

- **YOLO**: Finds WHERE (fast, accurate bounding boxes)
- **VLM**: Explains WHAT (domain-specific natural language)

## Built With

- [Ultralytics YOLOv8/v11](https://github.com/ultralytics/ultralytics) - State-of-the-art YOLO implementation
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) - Vision-Language Model (more VLMs coming soon)
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

### 3. Train (One Command)

```bash
python train.py --config configs/car_detection.yaml
```

This will:
1. Train YOLO (100 epochs)
2. Generate VLM dataset (Q&A pairs with red boxes)
3. Train VLM with QLoRA (3 epochs)
4. Export ONNX
5. Generate visualizations

### 4. Predict

```bash
# YOLO only
python predict.py --weights runs/exp_xxx/yolo/weights/best.pt --source image.jpg

# YOLO + VLM
python predict.py --weights runs/exp_xxx/yolo/weights/best.pt --source image.jpg \
    --vlm --vlm-adapter runs/exp_xxx/vlm/best
```

## Configuration

Edit `configs/car_detection.yaml`:

```yaml
# YOLO settings
yolo:
  model: yolov8n.pt
  epochs: 100
  batch: 16

# VLM settings
vlm:
  enabled: true
  model: Qwen/Qwen2.5-VL-7B-Instruct
  epochs: 3
  precision: 4bit  # QLoRA

# Visual grounding (red box)
vlm_dataset:
  box_color: [0, 0, 255]  # BGR Red
  box_thickness: 3
  system_prompt: |
    You are an object detection assistant.
    Identify objects in red marked areas clearly.
```

## Output Structure

```
runs/exp_20251217_xxx/
├── yolo/
│   ├── weights/best.pt      # YOLO model
│   └── weights/best.onnx    # ONNX export
├── vlm/
│   └── best/                # VLM adapter (~150MB)
└── visualizations/
    ├── training_curves.png
    └── prediction_samples.png
```

## Key Features

| Feature | Description |
|---------|-------------|
| Single Config | One YAML controls everything |
| Sequential Training | YOLO → VLM automatically |
| QLoRA | 7B VLM fits in 8GB VRAM |
| Visual Grounding | Red boxes link detection to VLM |
| Configurable | Colors, prompts, models all in YAML |

## Requirements

- Python 3.10+
- CUDA GPU (8GB+ VRAM recommended)
- For VLM: A100/RTX 3090 or better

## Example Results

**Input**: Image with vehicles

**YOLO Output**:
```
[car] conf=0.85 bbox=[27, 192, 129, 237]
```

**VLM Output**:
```
There is a car in the red marked area. The vehicle appears to be a passenger car.
```

## FAQ

**Do I need to manually write VLM training data?**
No. YoloGen automatically generates Q&A pairs from your YOLO labels. Just prepare standard YOLO format dataset.

**How many images do I need?**
Minimum ~100 images for YOLO, ~500+ recommended for better VLM results.

**Can I use only YOLO without VLM?**
Yes. Set `vlm.enabled: false` in config, or just use `predict.py` without `--vlm` flag.

**How much VRAM do I need?**
- YOLO only: 4GB+
- YOLO + VLM (4-bit): 8GB+ (RTX 3060 works)
- YOLO + VLM (fp16): 24GB+ (A100/RTX 3090)

**How do I customize VLM responses?**
Edit `system_prompt` and `details` under `vlm_dataset` section in your config file (e.g., `configs/car_detection.yaml`).

## License

MIT
