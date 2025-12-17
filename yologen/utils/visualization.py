"""
Visualization Utilities

Functions for drawing detection results on images and training metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Color palette for different classes (BGR format)
COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (128, 0, 255),  # Orange
    (255, 128, 0),  # Purple
    (0, 128, 255),  # Light orange
    (128, 255, 0),  # Light green
]


def draw_boxes(
    image: np.ndarray,
    detections: List[Dict],
    class_names: Dict[int, str] = None,
    thickness: int = 2,
    font_scale: float = 0.6,
    show_confidence: bool = True,
    show_label: bool = True,
) -> np.ndarray:
    """
    Draw detection boxes on image.

    Args:
        image: Input image (BGR or RGB)
        detections: List of detection dicts with keys:
                   - bbox: [x1, y1, x2, y2]
                   - class_id: int
                   - confidence: float
                   - class_name: str (optional)
        class_names: Dict mapping class_id to name
        thickness: Box line thickness
        font_scale: Label font scale
        show_confidence: Show confidence score
        show_label: Show class label

    Returns:
        Annotated image
    """
    img = image.copy()

    for det in detections:
        bbox = det.get('bbox', det.get('box', []))
        if len(bbox) < 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        cls_id = det.get('class_id', 0)
        conf = det.get('confidence', 1.0)

        # Get class name
        cls_name = det.get('class_name', '')
        if not cls_name and class_names:
            cls_name = class_names.get(cls_id, f'class_{cls_id}')

        # Get color
        color = COLORS[cls_id % len(COLORS)]

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        if show_label or show_confidence:
            label_parts = []
            if show_label and cls_name:
                label_parts.append(cls_name)
            if show_confidence:
                label_parts.append(f'{conf:.2f}')
            label = ': '.join(label_parts) if label_parts else ''

            if label:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 5, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (255, 255, 255), thickness)

    return img


def draw_red_box(
    image: np.ndarray,
    bbox: List[int],
    thickness: int = 3,
) -> np.ndarray:
    """
    Draw a red rectangle on image.

    Args:
        image: Input image
        bbox: [x1, y1, x2, y2] coordinates
        thickness: Line thickness

    Returns:
        Image with red box
    """
    img = image.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness)
    return img


def save_annotated_image(
    image: np.ndarray,
    detections: List[Dict],
    save_path: Union[str, Path],
    class_names: Dict[int, str] = None,
    **kwargs,
) -> str:
    """
    Draw detections and save image.

    Args:
        image: Input image
        detections: List of detections
        save_path: Output path
        class_names: Class name mapping
        **kwargs: Additional arguments for draw_boxes

    Returns:
        Saved file path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    annotated = draw_boxes(image, detections, class_names, **kwargs)
    cv2.imwrite(str(save_path), annotated)

    return str(save_path)


def create_grid(
    images: List[np.ndarray],
    grid_size: Tuple[int, int] = None,
    cell_size: Tuple[int, int] = (320, 320),
    padding: int = 5,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Create image grid from list of images.

    Args:
        images: List of images
        grid_size: (rows, cols), auto-calculated if None
        cell_size: Size of each cell (width, height)
        padding: Padding between cells
        bg_color: Background color

    Returns:
        Grid image
    """
    n = len(images)
    if n == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Calculate grid size
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_size

    # Create canvas
    cell_w, cell_h = cell_size
    canvas_w = cols * cell_w + (cols + 1) * padding
    canvas_h = rows * cell_h + (rows + 1) * padding
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

    # Place images
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break

        row = idx // cols
        col = idx % cols

        # Resize image
        resized = cv2.resize(img, cell_size)

        # Calculate position
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding)

        canvas[y:y + cell_h, x:x + cell_w] = resized

    return canvas


# ============================================================
# TRAINING VISUALIZATION
# ============================================================

def plot_training_results(
    results_dir: Union[str, Path],
    save_path: Union[str, Path] = None,
    show: bool = False,
) -> Optional[str]:
    """
    Plot training results from Ultralytics training.

    Args:
        results_dir: Path to training results directory
        save_path: Path to save the plot (default: results_dir/training_summary.png)
        show: Whether to display the plot

    Returns:
        Path to saved plot or None
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed. Skipping plot generation.")
        return None

    results_dir = Path(results_dir)
    results_csv = results_dir / "results.csv"

    if not results_csv.exists():
        print(f"Warning: results.csv not found at {results_csv}")
        return None

    # Read results
    import csv
    with open(results_csv, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    if not data:
        print("Warning: No training data found")
        return None

    # Extract metrics
    epochs = [int(float(row.get('epoch', i))) for i, row in enumerate(data)]

    # Try different column name formats (Ultralytics format varies)
    def get_col(row, *names):
        for name in names:
            # Try exact match and with spaces stripped
            for key in row.keys():
                if key.strip() == name or key.strip().replace(' ', '') == name.replace(' ', ''):
                    try:
                        return float(row[key])
                    except (ValueError, TypeError):
                        pass
        return None

    # Collect metrics
    train_box_loss = [get_col(r, 'train/box_loss', 'box_loss') for r in data]
    train_cls_loss = [get_col(r, 'train/cls_loss', 'cls_loss') for r in data]
    train_dfl_loss = [get_col(r, 'train/dfl_loss', 'dfl_loss') for r in data]
    val_box_loss = [get_col(r, 'val/box_loss') for r in data]
    val_cls_loss = [get_col(r, 'val/cls_loss') for r in data]
    metrics_mAP50 = [get_col(r, 'metrics/mAP50(B)', 'mAP50') for r in data]
    metrics_mAP50_95 = [get_col(r, 'metrics/mAP50-95(B)', 'mAP50-95') for r in data]
    metrics_precision = [get_col(r, 'metrics/precision(B)', 'precision') for r in data]
    metrics_recall = [get_col(r, 'metrics/recall(B)', 'recall') for r in data]

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Results', fontsize=14, fontweight='bold')

    # Plot 1: Box Loss
    ax = axes[0, 0]
    if any(v is not None for v in train_box_loss):
        ax.plot(epochs, train_box_loss, 'b-', label='Train', linewidth=2)
    if any(v is not None for v in val_box_loss):
        ax.plot(epochs, val_box_loss, 'r--', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Box Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Cls Loss
    ax = axes[0, 1]
    if any(v is not None for v in train_cls_loss):
        ax.plot(epochs, train_cls_loss, 'b-', label='Train', linewidth=2)
    if any(v is not None for v in val_cls_loss):
        ax.plot(epochs, val_cls_loss, 'r--', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Classification Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: DFL Loss
    ax = axes[0, 2]
    if any(v is not None for v in train_dfl_loss):
        ax.plot(epochs, train_dfl_loss, 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('DFL Loss')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No DFL Loss Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('DFL Loss')

    # Plot 4: mAP
    ax = axes[1, 0]
    if any(v is not None for v in metrics_mAP50):
        ax.plot(epochs, metrics_mAP50, 'b-', label='mAP50', linewidth=2)
    if any(v is not None for v in metrics_mAP50_95):
        ax.plot(epochs, metrics_mAP50_95, 'g-', label='mAP50-95', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('Mean Average Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Plot 5: Precision & Recall
    ax = axes[1, 1]
    if any(v is not None for v in metrics_precision):
        ax.plot(epochs, metrics_precision, 'b-', label='Precision', linewidth=2)
    if any(v is not None for v in metrics_recall):
        ax.plot(epochs, metrics_recall, 'r-', label='Recall', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Plot 6: Summary Text
    ax = axes[1, 2]
    ax.axis('off')

    # Get final metrics
    final_metrics = []
    if metrics_mAP50 and metrics_mAP50[-1] is not None:
        final_metrics.append(f"Final mAP50: {metrics_mAP50[-1]:.4f}")
    if metrics_mAP50_95 and metrics_mAP50_95[-1] is not None:
        final_metrics.append(f"Final mAP50-95: {metrics_mAP50_95[-1]:.4f}")
    if metrics_precision and metrics_precision[-1] is not None:
        final_metrics.append(f"Final Precision: {metrics_precision[-1]:.4f}")
    if metrics_recall and metrics_recall[-1] is not None:
        final_metrics.append(f"Final Recall: {metrics_recall[-1]:.4f}")
    final_metrics.append(f"Total Epochs: {len(epochs)}")

    summary_text = "Training Summary\n" + "=" * 25 + "\n" + "\n".join(final_metrics)
    ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    # Save
    if save_path is None:
        save_path = results_dir / "training_summary.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    plt.close()

    return str(save_path)


def visualize_predictions(
    model_path: Union[str, Path],
    images_dir: Union[str, Path],
    save_dir: Union[str, Path],
    num_images: int = 16,
    conf: float = 0.25,
    imgsz: int = 640,
    grid_size: Tuple[int, int] = (4, 4),
) -> str:
    """
    Run predictions on sample images and create visualization grid.

    Args:
        model_path: Path to YOLO weights
        images_dir: Directory containing images
        save_dir: Directory to save visualization
        num_images: Number of images to visualize
        conf: Confidence threshold
        imgsz: Image size
        grid_size: Grid layout (rows, cols)

    Returns:
        Path to saved visualization
    """
    from ultralytics import YOLO

    model_path = Path(model_path)
    images_dir = Path(images_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO(str(model_path))

    # Get images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = [f for f in images_dir.iterdir()
              if f.suffix.lower() in image_extensions]

    if not images:
        print(f"Warning: No images found in {images_dir}")
        return None

    # Sample images
    import random
    sample = random.sample(images, min(num_images, len(images)))

    # Run predictions
    annotated_images = []
    for img_path in sample:
        results = model.predict(str(img_path), conf=conf, imgsz=imgsz, verbose=False)

        if results and len(results) > 0:
            # Get annotated image
            annotated = results[0].plot()
            annotated_images.append(annotated)

    if not annotated_images:
        print("Warning: No predictions generated")
        return None

    # Create grid
    rows, cols = grid_size
    cell_size = (320, 320)
    grid = create_grid(annotated_images, grid_size=(rows, cols), cell_size=cell_size)

    # Save
    output_path = save_dir / "prediction_samples.png"
    cv2.imwrite(str(output_path), grid)

    # Also save individual predictions
    pred_dir = save_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for i, img in enumerate(annotated_images):
        cv2.imwrite(str(pred_dir / f"pred_{i:02d}.jpg"), img)

    return str(output_path)


def create_training_report(
    results_dir: Union[str, Path],
    model_path: Union[str, Path] = None,
    val_images_dir: Union[str, Path] = None,
    save_dir: Union[str, Path] = None,
) -> Dict[str, str]:
    """
    Create comprehensive training report with visualizations.

    Args:
        results_dir: Path to training results directory
        model_path: Path to best weights (auto-detected if None)
        val_images_dir: Path to validation images (for prediction samples)
        save_dir: Directory to save report (default: results_dir)

    Returns:
        Dictionary with paths to generated visualizations
    """
    results_dir = Path(results_dir)

    if save_dir is None:
        save_dir = results_dir
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # Auto-detect model path
    if model_path is None:
        best_pt = results_dir / "weights" / "best.pt"
        if best_pt.exists():
            model_path = best_pt

    # 1. Plot training curves
    print("  Generating training curves...")
    curves_path = plot_training_results(results_dir, save_dir / "training_curves.png")
    if curves_path:
        outputs['training_curves'] = curves_path
        print(f"    Saved: {curves_path}")

    # 2. Generate prediction samples
    if model_path and val_images_dir:
        print("  Generating prediction samples...")
        val_images_dir = Path(val_images_dir)
        if val_images_dir.exists():
            pred_path = visualize_predictions(
                model_path, val_images_dir, save_dir,
                num_images=16, grid_size=(4, 4)
            )
            if pred_path:
                outputs['prediction_samples'] = pred_path
                print(f"    Saved: {pred_path}")

    # 3. Copy Ultralytics plots if they exist
    ultralytics_plots = [
        'confusion_matrix.png',
        'confusion_matrix_normalized.png',
        'F1_curve.png',
        'P_curve.png',
        'R_curve.png',
        'PR_curve.png',
        'results.png',
        'labels.jpg',
        'labels_correlogram.jpg',
    ]

    for plot_name in ultralytics_plots:
        plot_path = results_dir / plot_name
        if plot_path.exists():
            outputs[plot_name.replace('.', '_')] = str(plot_path)

    print(f"\n  Report generated with {len(outputs)} visualizations")

    return outputs


# ============================================================
# VLM VISUALIZATION
# ============================================================

def plot_vlm_training_loss(
    loss_history: List[float],
    save_path: Union[str, Path],
    title: str = "VLM Training Loss",
) -> Optional[str]:
    """
    Plot VLM training loss curve.

    Args:
        loss_history: List of loss values per step/epoch
        save_path: Path to save the plot
        title: Plot title

    Returns:
        Path to saved plot or None
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed. Skipping plot generation.")
        return None

    if not loss_history:
        print("Warning: No loss data to plot")
        return None

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    steps = range(1, len(loss_history) + 1)
    ax.plot(steps, loss_history, 'b-', linewidth=2, label='Training Loss')

    # Add smoothed line if enough data
    if len(loss_history) > 10:
        window = min(10, len(loss_history) // 5)
        smoothed = np.convolve(loss_history, np.ones(window)/window, mode='valid')
        smooth_steps = range(window, len(loss_history) + 1)
        ax.plot(smooth_steps, smoothed, 'r-', linewidth=2, alpha=0.7, label='Smoothed')

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add final loss annotation
    final_loss = loss_history[-1]
    ax.annotate(f'Final: {final_loss:.4f}',
                xy=(len(loss_history), final_loss),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color='blue',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return str(save_path)


def visualize_vlm_samples(
    vlm_model,
    samples: List[Dict],
    save_dir: Union[str, Path],
    num_samples: int = 8,
    max_answer_len: int = 100,
) -> str:
    """
    Generate VLM sample outputs showing image + Q&A.

    Args:
        vlm_model: Loaded VLM model with generate() method
        samples: List of dicts with 'image_path', 'question', 'answer' (ground truth)
        save_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        max_answer_len: Max length to display for answers

    Returns:
        Path to saved visualization
    """
    from PIL import Image, ImageDraw, ImageFont

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not samples:
        print("Warning: No samples to visualize")
        return None

    # Sample random subset
    import random
    sample_subset = random.sample(samples, min(num_samples, len(samples)))

    # Create individual sample visualizations
    sample_images = []

    for i, sample in enumerate(sample_subset):
        try:
            img_path = sample['image_path']
            question = sample['question']
            gt_answer = sample.get('answer', 'N/A')

            # Generate prediction
            pred_answer = vlm_model.generate(
                image=img_path,
                question=question,
                max_new_tokens=128,
            )

            # Load image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((400, 300))

            # Create canvas with text area
            canvas = Image.new('RGB', (400, 500), color='white')
            canvas.paste(img, (0, 0))

            # Add text
            draw = ImageDraw.Draw(canvas)

            # Try to use a better font, fall back to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
                font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
            except:
                font = ImageFont.load_default()
                font_bold = font

            y_pos = 310

            # Question
            q_text = f"Q: {question[:80]}{'...' if len(question) > 80 else ''}"
            draw.text((10, y_pos), q_text, fill='black', font=font_bold)
            y_pos += 40

            # Ground truth
            gt_text = f"GT: {gt_answer[:max_answer_len]}{'...' if len(gt_answer) > max_answer_len else ''}"
            draw.text((10, y_pos), gt_text, fill='green', font=font)
            y_pos += 40

            # Prediction
            pred_text = f"Pred: {pred_answer[:max_answer_len]}{'...' if len(pred_answer) > max_answer_len else ''}"
            draw.text((10, y_pos), pred_text, fill='blue', font=font)

            sample_images.append(np.array(canvas))

            # Save individual
            canvas.save(save_dir / f"vlm_sample_{i:02d}.jpg")

        except Exception as e:
            print(f"Warning: Failed to process sample {i}: {e}")
            continue

    if not sample_images:
        print("Warning: No VLM samples generated")
        return None

    # Create grid
    grid = create_grid(
        sample_images,
        grid_size=(2, 4) if len(sample_images) >= 8 else (2, min(4, len(sample_images))),
        cell_size=(400, 500),
        padding=10,
    )

    # Save grid
    output_path = save_dir / "vlm_samples_grid.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    return str(output_path)


def create_vlm_report(
    vlm_dir: Union[str, Path],
    vlm_model=None,
    val_samples: List[Dict] = None,
    save_dir: Union[str, Path] = None,
    loss_history: List[float] = None,
) -> Dict[str, str]:
    """
    Create VLM training report with visualizations.

    Args:
        vlm_dir: Path to VLM training output directory
        vlm_model: Loaded VLM model (for generating sample outputs)
        val_samples: Validation samples for visualization
        save_dir: Directory to save report
        loss_history: Training loss history

    Returns:
        Dictionary with paths to generated visualizations
    """
    vlm_dir = Path(vlm_dir)

    if save_dir is None:
        save_dir = vlm_dir / "visualizations"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # 1. Plot training loss
    if loss_history:
        print("  Generating VLM training loss curve...")
        loss_path = plot_vlm_training_loss(
            loss_history,
            save_dir / "vlm_training_loss.png"
        )
        if loss_path:
            outputs['vlm_training_loss'] = loss_path
            print(f"    Saved: {loss_path}")

    # 2. Generate sample outputs
    if vlm_model and val_samples:
        print("  Generating VLM sample outputs...")
        samples_path = visualize_vlm_samples(
            vlm_model,
            val_samples,
            save_dir / "vlm_samples",
            num_samples=8,
        )
        if samples_path:
            outputs['vlm_samples'] = samples_path
            print(f"    Saved: {samples_path}")

    # 3. Check for existing adapter info
    best_adapter = vlm_dir / "best"
    if best_adapter.exists():
        outputs['best_adapter'] = str(best_adapter)

    print(f"\n  VLM report generated with {len(outputs)} items")

    return outputs
